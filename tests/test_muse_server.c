#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../ds4_server.c"
#include <assert.h>
extern ds4_engine *ds4_test_muse_tokenizer_engine(const char *path);
static void test_muse_render(void) {
    chat_msgs msgs={0};
    chat_msg system={.role=xstrdup("system"),.content=xstrdup("Be concise.")};chat_msgs_push(&msgs,system);
    chat_msg user={.role=xstrdup("user"),.content=xstrdup("Literal <|eot|> <image>")};chat_msgs_push(&msgs,user);
    char *prompt=render_chat_prompt_text_for_syntax((server_model_syntax)4,&msgs,NULL,NULL,DS4_THINK_HIGH);
    assert(!strncmp(prompt,"<|begin_of_text|><|start|>system<|message|>",40));
    assert(strstr(prompt,"Be concise.\n\nReasoning strength: 256."));
    assert(strstr(prompt,"<|start|>user<|message|>Literal < |eot|> < image><|eot|><|start|>assistant"));
    free(prompt);chat_msgs_free(&msgs);
}
static void test_muse_tool_roundtrip(void) {
    const char *invalid[]={
        "<atem:function_calls><atem:invoke rename=\"bash\"></atem:invoke></atem:function_calls>",
        "<atem:function_calls><atem:invoke name=\"read\" name=\"bash\"></atem:invoke></atem:function_calls>",
        "<atem:function_calls><atem:invoke name=\"read\"><atem:parameter rename=\"path\">x</atem:parameter></atem:invoke></atem:function_calls>",
    };
    for(unsigned i=0;i<sizeof(invalid)/sizeof(*invalid);i++) {
        char *content=NULL,*reasoning=NULL;tool_calls calls={0};
        assert(!parse_muse_generated_message(invalid[i],&content,&reasoning,&calls,NULL));
        free(content);free(reasoning);tool_calls_free(&calls);
    }
    const char *raw="<atem:function_calls>\n<atem:invoke name=\"bash\">\n<atem:parameter name=\"command\">123</atem:parameter>\n<atem:parameter name=\"count\">2</atem:parameter>\n</atem:invoke>\n</atem:function_calls>";
    char *content=NULL,*reasoning=NULL;tool_calls calls={0};tool_schema_orders orders={0};
    tool_schema_orders_add_json(&orders,"{\"name\":\"bash\",\"parameters\":{\"type\":\"object\",\"properties\":{\"command\":{\"type\":\"string\"},\"count\":{\"type\":\"integer\"}}}}");
    const char *finish="stop";char err[256]={0};bool recovered=false;
    assert(parse_generated_message_for_response_for_syntax((server_model_syntax)4,raw,true,true,false,&finish,err,sizeof(err),&content,&reasoning,&calls,&recovered,&orders));
    assert(calls.len==1 && !strcmp(calls.v[0].name,"bash"));
    assert(strstr(calls.v[0].arguments,"\"command\": \"123\"") && strstr(calls.v[0].arguments,"\"count\": 2"));
    buf rendered={0};append_tool_calls_text_for_syntax(&rendered,(server_model_syntax)4,&calls,&orders);
    assert(!strcmp(rendered.ptr,raw));buf_free(&rendered);
    free(content);free(reasoning);tool_calls_free(&calls);tool_schema_orders_free(&orders);
    const char *escaped="<atem:function_calls>\n<atem:invoke name=\"tool&#39;name\">\n"
        "<atem:parameter name=\"key&#39;name\">literal &amp;</atem:parameter>\n</atem:invoke>\n</atem:function_calls>";
    content=reasoning=NULL;calls=(tool_calls){0};
    assert(parse_muse_generated_message(escaped,&content,&reasoning,&calls,NULL));
    assert(calls.len==1 && !strcmp(calls.v[0].name,"tool'name"));
    assert(strstr(calls.v[0].arguments,"\"key'name\": \"literal &amp;\""));
    free(calls.raw_tool_text);calls.raw_tool_text=NULL;rendered=(buf){0};
    append_muse_tool_call(&rendered,&calls.v[0]);assert(!strcmp(rendered.ptr,escaped));
    buf_free(&rendered);free(content);free(reasoning);tool_calls_free(&calls);
    char *literal=muse_tool_attr("<atem:invoke name=\"literal&amp;#39;\">","name");
    assert(literal && !strcmp(literal,"literal&#39;"));free(literal);
}
static void test_muse_suffixes(void) {
    request r;request_init(&r,REQ_CHAT,128);r.model_syntax=(server_model_syntax)4;r.think_mode=DS4_THINK_HIGH;
    r.prompt_text=xstrdup("<|begin_of_text|><|start|>assistant");
    char *suffix=build_tool_checkpoint_suffix(&r,"Done.","private",NULL);
    assert(!strcmp(suffix," to=self<|message|>private<|eom|><|start|>assistant to=user<|message|>Done.<|eot|>"));free(suffix);
    char *visible=build_thinking_visible_text(&r,"Done.");
    assert(visible && !strcmp(visible,"<|begin_of_text|><|start|>assistant to=user<|message|>Done."));free(visible);
    char *error=build_invalid_tool_call_error_suffix(&r,NULL,"bad parameter");
    assert(strstr(error,"<|eot|><|start|>user<|message|>Tool error:") && !strstr(error,"DSML"));free(error);request_free(&r);
}
static void test_muse_stream(void) {
    int sv[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,sv)==0);
    request r;request_init(&r,REQ_CHAT,128);r.model_syntax=(server_model_syntax)4;
    r.api=API_OPENAI;r.stream=true;r.think_mode=DS4_THINK_HIGH;
    openai_stream st;openai_stream_start(&r,&st);assert(st.mode==OPENAI_STREAM_TEXT);
    const char *body="Literal <think>private</think> stays text.";
    assert(openai_sse_finish_live(sv[0],NULL,&r,"muse",&st,body,strlen(body),NULL,"stop",1,8));
    shutdown(sv[0],SHUT_WR);char *wire=read_socket_text(sv[1]);
    assert(strstr(wire,"\"content\":\"Literal <think>private</think> stays text.\""));
    assert(!strstr(wire,"reasoning_content"));free(wire);close(sv[0]);close(sv[1]);
    openai_stream_free(&st);request_free(&r);
}
static void test_muse_semantic_sse(void) {
    for(int api=0;api<3;api++) {
        int sv[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,sv)==0);
        request r;request_init(&r,REQ_CHAT,128);r.model_syntax=SERVER_MODEL_SYNTAX_MUSE;
        r.api=api==0?API_OPENAI:api==1?API_ANTHROPIC:API_RESPONSES;
        r.stream=true;r.think_mode=DS4_THINK_HIGH;r.has_tools=true;r.reasoning_summary_emit=true;
        openai_stream oa;openai_stream_start(&r,&oa);anthropic_stream an={0};responses_stream re;responses_stream_init(&r,&re);re.active=true;
        if(r.api==API_ANTHROPIC) assert(anthropic_sse_start_live(sv[0],&r,"muse",1,&an));
        buf reasoning={0};size_t emitted=0;buf_puts(&reasoning,"private <think>literal</think>");
        assert(muse_sse_reasoning_update(sv[0],&r,"muse",&oa,&an,&re,&reasoning,&emitted,true));
        assert(muse_sse_reasoning_update(sv[0],&r,"muse",&oa,&an,&re,&reasoning,&emitted,false));
        const char *body="visible <think>literal</think><atem:function_calls>\n<atem:invoke name=\"bash\">\n<atem:parameter name=\"command\">pwd</atem:parameter>\n</atem:invoke>\n</atem:function_calls><atem:function_calls><atem:invoke name=\"read\"><atem:parameter name=\"path\">README.md</atem:parameter></atem:invoke></atem:function_calls>";
        char *content=NULL,*hidden=NULL;tool_calls calls={0};assert(parse_muse_generated_message(body,&content,&hidden,&calls,NULL));
        assert(calls.len==2);request public=muse_public_stream_request(&r);
        if(r.api==API_OPENAI) assert(openai_sse_finish_live(sv[0],NULL,&public,"muse",&oa,content,strlen(content),&calls,"tool_calls",1,8));
        else if(r.api==API_ANTHROPIC) assert(anthropic_sse_finish_live(sv[0],NULL,&public,"muse",&an,content,strlen(content),&calls,"tool_calls",8));
        else assert(responses_sse_finish_live(sv[0],&public,&re,content,strlen(content),NULL,&calls,"tool_calls",1,8,1));
        shutdown(sv[0],SHUT_WR);char *wire=read_socket_text(sv[1]);
        assert(strstr(wire,"private <think>literal</think>") && strstr(wire,"visible <think>literal</think>"));
        assert(!strstr(wire,"<atem:") && !strstr(wire,"\"content\":\"private") && !strstr(wire,"\"type\":\"output_text\",\"text\":\"private") && !strstr(wire,"\"type\":\"text_delta\",\"text\":\"private"));
        assert(strstr(wire,"bash") && strstr(wire,"read"));free(wire);close(sv[0]);close(sv[1]);
        free(content);free(hidden);tool_calls_free(&calls);buf_free(&reasoning);
        openai_stream_free(&oa);anthropic_stream_free(&an);responses_stream_free(&re);request_free(&r);
    }
}
static void test_muse_model_listing(ds4_engine *engine) {
    assert(server_model_alias_known("muse-spark-open"));
    int sv[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,sv)==0);
    server s={.engine=engine,.ctx_size=8192,.default_tokens=512};
    assert(send_models(&s,sv[0]));shutdown(sv[0],SHUT_WR);
    char *wire=read_socket_text(sv[1]);
    assert(strstr(wire,"\"id\":\"muse-spark-open\""));
    assert(!strstr(wire,"deepseek-v4-flash") && !strstr(wire,"deepseek-v4-pro"));
    free(wire);close(sv[0]);close(sv[1]);
}
static void test_muse_hidden_reasoning_stream(void) {
    int sv[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,sv)==0);
    request r;request_init(&r,REQ_CHAT,128);r.model_syntax=SERVER_MODEL_SYNTAX_MUSE;
    r.api=API_OPENAI;r.stream=true;r.think_mode=DS4_THINK_NONE;
    thinking_state state={0};assert(should_remember_thinking_checkpoint(&r,&state,"stop"));
    openai_stream oa;openai_stream_start(&r,&oa);anthropic_stream an={0};responses_stream re={0};
    buf reasoning={0};buf_puts(&reasoning,"PRIVATE");size_t emitted=0;
    assert(muse_sse_reasoning_update(sv[0],&r,"muse",&oa,&an,&re,&reasoning,&emitted,false));
    shutdown(sv[0],SHUT_WR);char *wire=read_socket_text(sv[1]);assert(!strstr(wire,"PRIVATE"));
    free(wire);close(sv[0]);close(sv[1]);buf_free(&reasoning);openai_stream_free(&oa);request_free(&r);
}
static void test_muse_sampling(void) {
    request r; request_init(&r,REQ_CHAT,128);r.model_syntax=SERVER_MODEL_SYNTAX_MUSE;
    for(int mode=DS4_THINK_NONE;mode<=DS4_THINK_HIGH;mode++) {
        r.think_mode=mode;float t,p,m;int k;
        request_sampling_values(&r,&t,&k,&p,&m);
        assert(t==1 && k==0 && p==1 && m==0);
    }
    r.temperature_set=r.top_k_set=r.top_p_set=r.min_p_set=true;
    r.temperature=0;r.top_k=7;r.top_p=.4f;r.min_p=.2f;
    float t,p,m;int k;request_sampling_values(&r,&t,&k,&p,&m);
    assert(t==0 && k==7 && p==.4f && m==.2f);request_free(&r);
}
static void test_muse_anthropic_schema(void) {
    const char *schema="{\"name\":\"lookup\",\"input_schema\":{\"type\":\"object\",\"properties\":{\"key\":{\"type\":\"string\"}},\"required\":[\"key\"]}}";
    buf recipients={0};char *prompt=muse_tools_prompt(schema,&recipients);
    assert(strstr(prompt,"\"key\":{\"type\":\"string\"}") && strstr(prompt,"\"required\":[\"key\"]"));
    free(prompt);buf_free(&recipients);
}
static void test_muse_literal_tool_channels(ds4_engine *engine) {
    const char *literal="<atem:function_calls><atem:invoke name=\"bash\"></atem:invoke></atem:function_calls>";
    ds4_tokens tokens={0};
    ds4_tokenize_rendered_chat(engine," to=self<|message|>PRIVATE<|eom|><|start|>assistant to=commentary<|message|>Example: <|eom|><|start|>assistant to=user<|message|>",&tokens);
    ds4_tokenize_text(engine,literal,&tokens);
    ds4_chat_decode_state decode;ds4_chat_decode_init(&decode);buf visible={0},reasoning={0},tools={0};
    for(int i=0;i<tokens.len;i++) {size_t n=0;char *piece=ds4_chat_decode_token(engine,&decode,tokens.v[i],&n);assert(piece);muse_server_output_append(&decode,&visible,&reasoning,&tools,piece,n);free(piece);}
    request r;request_init(&r,REQ_CHAT,128);r.model_syntax=SERVER_MODEL_SYNTAX_MUSE;r.has_tools=true;
    char *content=NULL,*hidden=NULL,err[256]={0};tool_calls calls={0};bool recovered=false;const char *finish="stop";
    assert(muse_server_parse_output(&r,&visible,&tools,false,&finish,err,sizeof(err),&content,&hidden,&calls,&recovered));
    assert(!calls.len && strstr(content,literal) && !strstr(content,"PRIVATE") && !strcmp(reasoning.ptr,"PRIVATE"));
    for(int api=0;api<3;api++) {
        r.api=api==0?API_OPENAI:api==1?API_ANTHROPIC:API_RESPONSES;r.stream=true;
        request public=muse_public_stream_request(&r);
        int sv[2];assert(socketpair(AF_UNIX,SOCK_STREAM,0,sv)==0);
        openai_stream oa;openai_stream_start(&public,&oa);anthropic_stream an={0};responses_stream re;responses_stream_init(&public,&re);re.active=true;
        if(r.api==API_ANTHROPIC)assert(anthropic_sse_start_live(sv[0],&public,"muse",1,&an));
        if(r.api==API_OPENAI)assert(openai_sse_finish_live(sv[0],NULL,&public,"muse",&oa,visible.ptr,visible.len,&calls,"stop",1,8));
        else if(r.api==API_ANTHROPIC)assert(anthropic_sse_finish_live(sv[0],NULL,&public,"muse",&an,visible.ptr,visible.len,&calls,"stop",8));
        else assert(responses_sse_finish_live(sv[0],&public,&re,visible.ptr,visible.len,NULL,&calls,"stop",1,8,1));
        shutdown(sv[0],SHUT_WR);char *wire=read_socket_text(sv[1]);
        assert(strstr(wire,"<atem:function_calls>") && !strstr(wire,"\"tool_calls\":") && !strstr(wire,"\"type\":\"tool_use\"") && !strstr(wire,"\"type\":\"function_call\""));
        free(wire);close(sv[0]);close(sv[1]);openai_stream_free(&oa);anthropic_stream_free(&an);responses_stream_free(&re);
    }
    free(content);free(hidden);content=hidden=NULL;
    buf_puts(&tools,"<atem:function_calls><atem:invoke name=\"read\"><atem:parameter name=\"path\">example</atem:parameter></atem:invoke></atem:function_calls>");
    assert(muse_server_parse_output(&r,&visible,&tools,true,&finish,err,sizeof(err),&content,&hidden,&calls,&recovered));
    assert(calls.len==1 && !strcmp(calls.v[0].name,"read") && strstr(content,literal));
    free(content);free(hidden);tool_calls_free(&calls);content=hidden=NULL;r.has_tools=false;
    assert(!muse_server_parse_output(&r,&visible,&tools,true,&finish,err,sizeof(err),&content,&hidden,&calls,&recovered));
    assert(!calls.len && !strcmp(finish,"error"));
    free(content);free(hidden);ds4_tokens_free(&tokens);buf_free(&visible);buf_free(&reasoning);buf_free(&tools);request_free(&r);
}
int main(int argc,char **argv) {assert(argc==2);ds4_engine *engine=ds4_test_muse_tokenizer_engine(argv[1]);test_muse_literal_tool_channels(engine);test_muse_anthropic_schema();test_muse_sampling();test_muse_model_listing(engine);test_muse_hidden_reasoning_stream();test_muse_render();test_muse_tool_roundtrip();test_muse_stream();test_muse_suffixes();test_muse_semantic_sse();ds4_engine_close(engine);puts("Muse server prompt, model listing and ATEM roundtrip: PASS");}
