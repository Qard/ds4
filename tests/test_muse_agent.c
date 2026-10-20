#define DS4_AGENT_TEST
#define DS4_AGENT_TEST_NO_MAIN
#include "../ds4_agent.c"
#include <assert.h>
extern ds4_engine *ds4_test_muse_tokenizer_engine(const char *path);

static void feed(agent_worker *w, agent_stream_renderer *stream, const char *text) {
    ds4_tokens tokens = {0};ds4_tokenize_rendered_chat(w->engine, text, &tokens);
    int generated = 0;char err[160] = {0};
    for (int i = 0; i < tokens.len; i++)
        AGENT_TEST_ASSERT(!worker_finish_generated_token(w, tokens.v[i], &generated,
            now_sec(), stream, false, err, sizeof(err)));
    ds4_tokens_free(&tokens);
}

static void test_semantic_stream(agent_worker *w) {
    w->trace=tmpfile();assert(w->trace);
    agent_tail_capture tail = {.cap = 16384};
    agent_token_renderer renderer = {.engine = w->engine, .worker = w,
        .capture = &tail, .last_output_newline = true};
    agent_dsml_parser parser = {.syntax = AGENT_TOOL_SYNTAX_MUSE};
    agent_stream_renderer stream = {.renderer = &renderer, .parser = &parser,
        .syntax = AGENT_TOOL_SYNTAX_MUSE};
    feed(w, &stream, " to=self<|message|>Private planning: "
        "<atem:function_calls><atem:invoke name=\"bash\"></atem:invoke></atem:function_calls>");
    AGENT_TEST_ASSERT(renderer.in_think && parser.calls.len == 0);
    feed(w, &stream, "<|eom|><|start|>assistant to=commentary<|message|>"
        "Visible <think>literal</think>.\n<|eom|><|start|>assistant to=list<|message|>"
        "<atem:function_calls><atem:invoke name=\"list\">"
        "<atem:parameter name=\"path\">.</atem:parameter>"
        "</atem:invoke></atem:function_calls>");
    feed(w, &stream, "<|eom|><|start|>assistant to=read<|message|>"
        "<atem:function_calls><atem:invoke name=\"read\">"
        "<atem:parameter name=\"path\">README.md</atem:parameter>"
        "</atem:invoke></atem:function_calls>");
    agent_stream_text(&stream, "", 0, true);renderer_finish(&renderer);
    size_t n = 0;char *out = agent_tail_capture_take(&tail, &n);
    AGENT_TEST_ASSERT(!renderer.in_think && parser.calls.len == 2);
    if (parser.calls.len == 2) AGENT_TEST_ASSERT(!strcmp(parser.calls.v[0].name, "list") && !strcmp(parser.calls.v[1].name, "read"));
    AGENT_TEST_ASSERT(!strstr(out, "<|start|>") && !strstr(out, "to=self"));
    AGENT_TEST_ASSERT(strstr(out, "Visible <think>literal</think>.") != NULL);
    if(!ds4_think_mode_enabled(w->cfg->gen.think_mode)) AGENT_TEST_ASSERT(!strstr(out,"Private planning"));
    rewind(w->trace);char traced[32768]={0};
    assert(fread(traced,1,sizeof(traced)-1,w->trace)>0);
    AGENT_TEST_ASSERT(strstr(traced,"<|message|>")!=NULL);
    fclose(w->trace);w->trace=NULL;
    free(out);agent_dsml_parser_free(&parser);ds4_tokens_free(&w->transcript);
}

static void test_public_tool_literal(agent_worker *w) {
    const char *literal="<atem:function_calls><atem:invoke name=\"bash\"><atem:parameter name=\"command\">example</atem:parameter></atem:invoke></atem:function_calls>";
    const char *headers[]={" to=user<|message|>"," to=commentary<|message|>"};
    for(unsigned h=0;h<2;h++) {
    agent_tail_capture tail={.cap=16384};
    agent_token_renderer renderer={.engine=w->engine,.worker=w,.capture=&tail,.last_output_newline=true};
    agent_dsml_parser parser={.syntax=AGENT_TOOL_SYNTAX_MUSE};
    agent_stream_renderer stream={.renderer=&renderer,.parser=&parser,.syntax=AGENT_TOOL_SYNTAX_MUSE};
    feed(w,&stream,headers[h]);feed(w,&stream,literal);
    agent_stream_text(&stream,"",0,true);renderer_finish(&renderer);
    size_t n=0;char *out=agent_tail_capture_take(&tail,&n);
    AGENT_TEST_ASSERT(parser.calls.len==0);
    AGENT_TEST_ASSERT(strstr(out,literal)!=NULL);
    free(out);agent_dsml_parser_free(&parser);ds4_tokens_free(&w->transcript);
    }
}

static void test_tool_results(agent_worker *w) {
    agent_tool_call calls[] = {{.name = "missing_one"}, {.name = "missing_two"}};
    agent_tool_calls all = {.v = calls, .len = 2};
    agent_tool_observation obs = agent_execute_tool_observation(w, &all);
    ds4_tokens tokens = {0};ds4_vision_span *spans = NULL;char err[160] = {0};
    AGENT_TEST_ASSERT(agent_tool_observation_build(w, &obs, &tokens, &spans, err, sizeof(err)));
    size_t n = 0;char *text = ds4_kvstore_render_tokens_text(w->engine, &tokens, &n);
    AGENT_TEST_ASSERT(strstr(text, "<|start|>tool missing_one<|message|><tool_output name=\"missing_one\">\n") != NULL);
    AGENT_TEST_ASSERT(strstr(text, "</tool_output><|eot|><|start|>tool missing_two<|message|>") != NULL);
    free(text);free(spans);ds4_tokens_free(&tokens);agent_tool_observation_free(&obs);
}

static void test_history(agent_worker *w) {
    const char *text = "<|begin_of_text|><|start|>system<|message|>system secret<|eot|>"
        "<|start|>user<|message|>Old question<|eot|><|start|>assistant to=user<|message|>Old answer<|eot|>"
        "<|start|>user<|message|>Current question<|eot|>"
        "<|start|>assistant to=self<|message|>Private plan<|eom|>"
        "<|start|>assistant to=user<|message|>Visible <think>literal</think><|eot|>";
    free(w->out);w->out = NULL;w->out_len = w->out_cap = 0;
    agent_history_render_text(w, text, strlen(text), 1);
    AGENT_TEST_ASSERT(w->out && strstr(w->out, "Current question") && strstr(w->out, "Visible <think>literal</think>"));
    AGENT_TEST_ASSERT(w->out && !strstr(w->out, "Old question") && !strstr(w->out, "system secret") && !strstr(w->out, "<|start|>"));
    char *title = agent_session_title_from_text(text, strlen(text), 0);
    AGENT_TEST_ASSERT(!strcmp(title, "Old question"));free(title);
}

static void test_tail_boundary(agent_worker *w) {
    ds4_tokens tokens = {0};
    ds4_chat_append_message(w->engine, &tokens, "user", "Keep these constraints.");
    const int first = tokens.len;
    ds4_chat_append_message(w->engine, &tokens, "user", "Current question");
    ds4_chat_append_message(w->engine, &tokens, "assistant", "Longer answer to retain.");
    ds4_tokens_copy(&w->transcript, &tokens);
    w->cfg->gen.ctx_size = 4096;
    AGENT_TEST_ASSERT(agent_compact_tail_start(w, tokens.len, 0) == first);
    ds4_tokens_free(&w->transcript);ds4_tokens_free(&tokens);
}

static void test_compaction_progress(void) {
    agent_status status={.state=AGENT_WORKER_COMPACTING,.ctx_used=7101,.ctx_size=8192,
        .prefill_total=5000,.prefill_done=1024,.prefill_tps=8.0};
    char text[512];build_status_text(&status,text,sizeof(text));
    AGENT_TEST_ASSERT(strstr(text,"COMPACTING reading 1024/5000")!=NULL);
    status.generated=12;status.gen_tps=3.0;
    build_status_text(&status,text,sizeof(text));
    AGENT_TEST_ASSERT(strstr(text,"COMPACTING summary 12 tokens")!=NULL);
}

static void test_resume_prefix(agent_worker *w) {
    const char *headers[] = {"<|start|>assistant to=self<|message|>",
        "<|start|>assistant to=commentary<|message|>",
        "<|start|>assistant to=read<|message|>"};
    for (unsigned h = 0; h < sizeof(headers)/sizeof(*headers); h++) {
        ds4_tokens source = {0},prefix = {0};
        ds4_tokenize_rendered_chat(w->engine, headers[h], &source);
        int header_end = source.len;
        ds4_tokenize_text(w->engine, "partial body", &source);
        for (int cut = 1; cut < source.len; cut++) {
            agent_compact_resume_prefix(w->engine, &source, 0, cut, &prefix);
            int expected = cut < header_end ? cut : header_end;
            AGENT_TEST_ASSERT(prefix.len == expected);
            if (prefix.len == expected) AGENT_TEST_ASSERT(!memcmp(prefix.v, source.v, (size_t)expected * sizeof(int)));
            ds4_tokens_free(&prefix);
        }
        ds4_tokens_free(&source);
    }
}

int main(int argc, char **argv) {
    assert(argc == 2);
    agent_config cfg = {0};cfg.gen.think_mode = DS4_THINK_MAX;
    cfg.gen.system = "Follow the user. Keep <|start|> literal.";
    agent_worker w = {.engine = ds4_test_muse_tokenizer_engine(argv[1]), .cfg = &cfg};
    agent_generation_options defaults={.min_p=.05f};agent_apply_model_sampling_defaults(w.engine,&defaults);
    AGENT_TEST_ASSERT(defaults.temperature==1 && defaults.top_p==1 && defaults.min_p==0);
    agent_generation_options explicit={.temperature=.2f,.top_p=.4f,.min_p=.7f,
        .temperature_set=true,.top_p_set=true,.min_p_set=true};
    agent_apply_model_sampling_defaults(w.engine,&explicit);
    AGENT_TEST_ASSERT(explicit.temperature==.2f && explicit.top_p==.4f && explicit.min_p==.7f);
    pthread_mutex_init(&w.mu, NULL);
    ds4_tokens tokens = {0};agent_worker_build_system_tokens(&w, &tokens);
    size_t n = 0;char *text = ds4_kvstore_render_tokens_text(w.engine, &tokens, &n);
    AGENT_TEST_ASSERT(strstr(text, "<|start|>system<|message|>") != NULL);
    AGENT_TEST_ASSERT(strstr(text, "<atem:function_calls>") != NULL);
    AGENT_TEST_ASSERT(strstr(text, "Reasoning strength: 512.") != NULL);
    AGENT_TEST_ASSERT(strstr(text, "Keep < |start|> literal.") != NULL);
    AGENT_TEST_ASSERT(strstr(text, "DSML") == NULL);
    AGENT_TEST_ASSERT(strstr(text, "<tool_call>") == NULL);
    const char *first = strstr(text, "<|start|>system");
    AGENT_TEST_ASSERT(first && !strstr(first + 1, "<|start|>system"));
    free(text);ds4_tokens_free(&tokens);
    test_semantic_stream(&w);test_public_tool_literal(&w);
    cfg.gen.think_mode=DS4_THINK_NONE;test_semantic_stream(&w);test_public_tool_literal(&w);
    test_tool_results(&w);
    test_history(&w);
    test_tail_boundary(&w);
    test_compaction_progress();
    test_resume_prefix(&w);
    free(w.out);
    ds4_engine_close(w.engine);pthread_mutex_destroy(&w.mu);
    return agent_test_failures ? 1 : 0;
}
