#include "../ds4.c"
#include <assert.h>
static void feed(ds4_engine *e,ds4_chat_decode_state *s,const char *text,const char *expected,bool reasoning) {
    token_vec tokens={0};tokenize_rendered_chat_vocab(&e->vocab,text,&tokens);
    char out[4096]={0};size_t used=0;
    for(int i=0;i<tokens.len;i++) {size_t n=0;char *p=ds4_chat_decode_token(e,s,tokens.v[i],&n);assert(p);assert(used+n<sizeof(out));memcpy(out+used,p,n);used+=n;free(p);}
    assert(!strcmp(out,expected));assert(s->reasoning==reasoning);token_vec_free(&tokens);
}
int main(int argc,char **argv) {
    assert(argc==2);ds4_engine e={0};g_ds4_shape=DS4_SHAPE_MUSE_SPARK;
    model_open(&e.model,argv[1],false,false);vocab_load(&e.vocab,&e.model);
    ds4_chat_decode_state s;ds4_chat_decode_init(&s,DS4_THINK_HIGH);
    feed(&e,&s," to=self<|message|>private <think>literal</think>","private <think>literal</think>",true);
    assert(!s.final_answer);
    feed(&e,&s,"<|eom|><|start|>assistant to=user<|message|>Answer <think>literal</think>","Answer <think>literal</think>",false);
    assert(s.final_answer);
    feed(&e,&s,"<|eom|><|start|>assistant to=bash<|message|><atem:function_calls>","<atem:function_calls>",false);
    assert(!s.final_answer);
    ds4_chat_decode_init(&s,DS4_THINK_NONE);
    assert(s.final_answer);
    feed(&e,&s,"Just the answer","Just the answer",false);
    ds4_chat_decode_init(&s,DS4_THINK_HIGH);
    feed(&e,&s," to=selfish<|message|>visible","visible",false);
    const char *bad_headers[]={" to=self to=user<|message|>"," to=user to=self<|message|>"," to=<|message|>"};
    for(unsigned h=0;h<sizeof(bad_headers)/sizeof(*bad_headers);h++) {
        ds4_chat_decode_init(&s,DS4_THINK_HIGH);token_vec bad={0};bool refused=false;
        tokenize_rendered_chat_vocab(&e.vocab,bad_headers[h],&bad);
        for(int i=0;i<bad.len;i++) {size_t n;char *p=ds4_chat_decode_token(&e,&s,bad.v[i],&n);if(!p) {refused=true;break;}free(p);}
        assert(refused);token_vec_free(&bad);
    }
    ds4_chat_decode_init(&s,DS4_THINK_HIGH);
    char oversized[1024];memset(oversized,'a',sizeof(oversized)-1);oversized[sizeof(oversized)-1]=0;
    token_vec tokens={0};bpe_tokenize_text(&e.vocab,oversized,&tokens);bool rejected=false;
    for(int i=0;i<tokens.len;i++) {size_t n;char *p=ds4_chat_decode_token(&e,&s,tokens.v[i],&n);if(!p) {rejected=true;break;}free(p);}
    assert(rejected);token_vec_free(&tokens);vocab_free(&e.vocab);model_close(&e.model);
    puts("Muse semantic decoding: recipient headers, literal body tags, and bounded malformed headers PASS");
}
