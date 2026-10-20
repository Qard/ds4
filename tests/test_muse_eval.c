#define main ds4_eval_program_main
#include "../ds4_eval.c"
#undef main
#include <assert.h>
extern ds4_engine *ds4_test_muse_tokenizer_engine(const char *path);

static void feed(ds4_engine *e,ds4_chat_decode_state *state,byte_buf *answer,eval_ui *ui,const char *text) {
    ds4_tokens tokens={0};ds4_tokenize_rendered_chat(e,text,&tokens);
    for(int i=0;i<tokens.len;i++) {
        size_t len=0;char *piece=eval_decode_piece(e,state,tokens.v[i],answer,&len);assert(piece);
        eval_project_piece(ui,e,state,piece,len,false);free(piece);
    }
    ds4_tokens_free(&tokens);
}
int main(int argc,char **argv) {
    assert(argc==2);ds4_engine *e=ds4_test_muse_tokenizer_engine(argv[1]);
    ds4_chat_decode_state state;ds4_chat_decode_init(&state,DS4_THINK_HIGH);
    byte_buf answer={0};eval_ui ui={0};
    feed(e,&state,&answer,&ui," to=self<|message|>Private\nAnswer: A");
    assert(!answer.len && state.reasoning);
    assert(ui.stream.v && !strstr(ui.stream.v,"to=self"));
    assert(ui.styles.len && ui.styles.v[0]==1);
    feed(e,&state,&answer,&ui,"<|eom|><|start|>assistant to=commentary<|message|>Checking.\nAnswer: C");
    assert(!answer.len);
    feed(e,&state,&answer,&ui,"<|eom|><|start|>assistant to=user<|message|>Visible <think>literal</think>\nAnswer: B");
    assert(answer.v && !strcmp(answer.v,"Visible <think>literal</think>\nAnswer: B"));
    assert(strstr(ui.stream.v,"Visible <think>literal</think>"));
    assert(!ui.in_think && ui.styles.v[ui.styles.len-1]==0);
    ds4_tokens close={0},expected={0};eval_build_think_close(e,&close);
    ds4_tokenize_rendered_chat(e,"<|eom|><|start|>assistant to=user<|message|>",&expected);
    assert(close.len==expected.len && !memcmp(close.v,expected.v,(size_t)close.len*sizeof(int)));
    eval_config cfg={.think_mode=DS4_THINK_MAX};
    assert(eval_auto_context_size(e,&cfg,eval_core_cases,1,128,NULL,NULL)<(int)ds4_think_max_min_context());
    buf_free(&answer);ds4_chat_decode_init(&state,DS4_THINK_HIGH);
    feed(e,&state,&answer,&ui," to=self<|message|>Private plan");
    cfg.hard_limit_reply_budget=512;int position=-1,rank=0;eval_think_close_kind kind;
    for(int i=0;i<close.len;i++) {
        int token=eval_think_close_token(NULL,&cfg,&close,i==0,true,512-i,&position,&rank,&kind);
        assert(token==close.v[i] && kind==(i?EVAL_THINK_CLOSE_NONE:EVAL_THINK_CLOSE_HARD));
        size_t len;char *piece=eval_decode_piece(e,&state,token,&answer,&len);assert(piece);free(piece);
    }
    assert(position==-1 && state.body && state.final_answer && !state.reasoning && !answer.len);
    assert(eval_think_close_token(NULL,&cfg,&close,false,true,1,&position,&rank,&kind)==-1);
    ds4_tokens_free(&close);ds4_tokens_free(&expected);
    buf_free(&answer);buf_free(&ui.stream);style_free(&ui.styles);ds4_engine_close(e);
    puts("Muse evaluation recipients, answer isolation, and native forced transition: PASS");return 0;
}
