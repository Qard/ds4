/* Length-prefixed UTF-8 requests, lists of pre-token byte lengths in reply. */
#include "../ds4.c"
#include <assert.h>
int main(int argc,char **argv) {
    ds4_model model={0}; ds4_vocab vocab={0};
    if(argc==2) {
        g_ds4_shape=DS4_SHAPE_MUSE_SPARK;
        model_open(&model,argv[1],false,false);
        vocab_load(&vocab,&model);
        assert(vocab_token_is_generation_stop(&vocab,200008));
        assert(vocab_token_is_generation_stop(&vocab,200001));
        assert(!vocab_token_is_generation_stop(&vocab,200007));
    }
    uint32_t n;
    while(fread(&n,4,1,stdin)==1) {
        assert(n<1048576);
        char *s=malloc((size_t)n+1); uint32_t *lengths=malloc(((size_t)n+1)*4);
        assert(s && lengths && fread(s,1,n,stdin)==n); s[n]=0;
        if(argc==2) {
            token_vec tokens={0}; tokenize_rendered_chat_vocab(&vocab,s,&tokens);
            uint32_t count=tokens.len;
            assert(fwrite(&count,4,1,stdout)==1 && fwrite(tokens.v,4,count,stdout)==count);
            free(tokens.v); free(lengths); free(s); continue;
        }
        uint32_t count=0; uint64_t pos=0;
        while(pos<n) {
            uint64_t end=muse_next_piece(s,n,pos);
            assert(end>pos && end<=n);
            lengths[count++]=(uint32_t)(end-pos); pos=end;
        }
        assert(fwrite(&count,4,1,stdout)==1 && fwrite(lengths,4,count,stdout)==count);
        free(lengths); free(s);
    }
    if(argc==2) { vocab_free(&vocab); model_close(&model); }
}
