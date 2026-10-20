/* Manual checkpoint diagnostic. The observer exists only in this test build;
 * production graph submissions and public engine APIs are unchanged. */
static int checkpoint_layer(void *graph,unsigned layer,unsigned rows);
#define DS4_MUSE_LAYER_OBSERVER(g,layer,rows) checkpoint_layer(g,layer,rows)
#include "../ds4.c"
#include <assert.h>
static const char *trace_directory;
static int save_tensor(const char *name,unsigned layer,ds4_gpu_tensor *tensor,uint64_t bytes) {
    char path[4096];int n=snprintf(path,sizeof(path),"%s/layer-%02u.%s",trace_directory,layer,name);
    if(n<0 || (size_t)n>=sizeof(path) || bytes>SIZE_MAX) return 0;
    void *data=xmalloc((size_t)bytes);FILE *fp=fopen(path,"wbx");
    int ok=fp && ds4_gpu_tensor_read(tensor,0,data,bytes) && fwrite(data,1,bytes,fp)==bytes;
    if(fp && fclose(fp))ok=0;free(data);return ok;
}
static int checkpoint_layer(void *graph,unsigned layer,unsigned rows) {
    ds4_muse_gpu_graph *g=graph;
    if(!trace_directory)return 1;
    /* Commit before reading, then continue the same tensor graph. Comparing
     * final logits with an unobserved run checks submission-boundary effects. */
    return ds4_gpu_end_commands() &&
        save_tensor("residual.f32",layer,g->r,(uint64_t)rows*DS4_N_EMBD*4) &&
        save_tensor("selected.i32",layer,g->selected,(uint64_t)rows*DS4_N_EXPERT_USED*4) &&
        save_tensor("ffn_norm.f32",layer,g->xn,(uint64_t)rows*DS4_N_EMBD*4) &&
        save_tensor("router.f32",layer,g->router,(uint64_t)rows*DS4_N_EXPERT*4) &&
        save_tensor("q.f32",layer,g->q,(uint64_t)rows*DS4_N_HEAD*DS4_N_HEAD_DIM*4) &&
        save_tensor("key.f16",layer,g->kc[layer],(uint64_t)rows*DS4_N_HEAD_KV*DS4_N_HEAD_DIM*2) &&
        ds4_gpu_begin_commands();
}
int main(int argc,char **argv) {
    if(argc!=4){fprintf(stderr,"Usage: %s MODEL PROMPT_IDS OUTPUT_DIRECTORY\n",argv[0]);return 2;}
    FILE *fp=fopen(argv[2],"rb");if(!fp)return 2;
    int ids[33];size_t bytes=fread(ids,1,sizeof(ids),fp);int tail=fgetc(fp);fclose(fp);
    size_t count=bytes/4;
    if(!count || count>32 || bytes%4 || tail!=EOF){fprintf(stderr,"Provide 1..32 int32 prompt IDs\n");return 2;}
    ds4_engine_options opt={.model_path=argv[1],.backend=DS4_BACKEND_METAL,
        .context_size=4096,.prefill_chunk=32,.power_percent=100,.ssd_streaming=true,
        .ssd_streaming_cache_experts=1024,.placement_ctx_hint=4096,.placement_session_count_hint=1};
    ds4_engine *e=NULL;ds4_session *s=NULL;int rc=1;char err[512]={0};
    if(ds4_engine_open(&e,&opt))return 1;
    if(!ds4_engine_is_muse(e) || ds4_session_create(&s,e,4096))goto done;
    ds4_tokens prompt={.v=ids,.len=(int)count,.cap=(int)count};
    trace_directory=argv[3];
    if(ds4_session_sync(s,&prompt,err,sizeof(err))){fprintf(stderr,"%s\n",err);goto done;}
    rc=!save_tensor("logits.f32",DS4_N_LAYER,s->muse_graph.logits,(uint64_t)DS4_N_VOCAB*4);
 done:
    ds4_session_free(s);ds4_engine_close(e);return rc;
}
