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
        const char *body="visible <think>literal</think><atem:function_calls>\n<atem:invoke name=\"bash\">\n<atem:parameter name=\"command\">pwd</atem:parameter>\n</atem:invoke>\n</atem:function_calls>";
        char *content=NULL,*hidden=NULL;tool_calls calls={0};assert(parse_muse_generated_message(body,&content,&hidden,&calls,NULL));
        if(r.api==API_OPENAI) assert(openai_sse_finish_live(sv[0],NULL,&r,"muse",&oa,body,strlen(body),&calls,"tool_calls",1,8));
        else if(r.api==API_ANTHROPIC) assert(anthropic_sse_finish_live(sv[0],NULL,&r,"muse",&an,body,strlen(body),&calls,"tool_calls",8));
        else assert(responses_sse_finish_live(sv[0],&r,&re,body,strlen(body),NULL,&calls,"tool_calls",1,8,1));
        shutdown(sv[0],SHUT_WR);char *wire=read_socket_text(sv[1]);
        assert(strstr(wire,"private <think>literal</think>") && strstr(wire,"visible <think>literal</think>"));
        assert(!strstr(wire,"<atem:") && !strstr(wire,"\"content\":\"private") && !strstr(wire,"\"type\":\"output_text\",\"text\":\"private") && !strstr(wire,"\"type\":\"text_delta\",\"text\":\"private"));
        assert(strstr(wire,"bash"));free(wire);close(sv[0]);close(sv[1]);
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
int main(int argc,char **argv) {assert(argc==2);ds4_engine *engine=ds4_test_muse_tokenizer_engine(argv[1]);test_muse_model_listing(engine);test_muse_render();test_muse_tool_roundtrip();test_muse_stream();test_muse_suffixes();test_muse_semantic_sse();ds4_engine_close(engine);puts("Muse server prompt, model listing and ATEM roundtrip: PASS");}
