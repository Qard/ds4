/* Whole native Muse graph at a bounded test shape, bypassing only the fixed
 * production-profile selector. Tensor binding and every GPU op are real. */
static unsigned observed_layers;
#define DS4_MUSE_LAYER_OBSERVER(g,layer,tokens) (++observed_layers != 0)
#include "../ds4.c"
#include <assert.h>

int main(int argc, char **argv) {
    assert(argc==6 || argc==7);
    bool matrix=argc==7 && atoi(argv[6])!=0;
    unsigned count=matrix?40:10, context=matrix?64:16;
    bool packed=atoi(argv[5])!=0; unsigned d=packed?256:128;
    g_ds4_shape=DS4_SHAPE_MUSE_SPARK;
    g_ds4_shape.n_layer=2; g_ds4_shape.n_embd=d; g_ds4_shape.n_vocab=32;
    g_ds4_shape.n_head=d/64; g_ds4_shape.n_head_kv=d/128;
    g_ds4_shape.n_expert=4; g_ds4_shape.n_expert_used=2;
    g_ds4_shape.n_ff_exp=d; g_ds4_shape.n_moe_latent=d; g_ds4_shape.n_swa=4;
    g_muse_swa[0]=1; g_muse_swa[1]=0;
    ds4_model m; model_open(&m,argv[1],true,false);
    ds4_weights w; weights_bind(&w,&m,false,0,UINT32_MAX,true,false);
    assert(ds4_gpu_init());
    bool streamed=atoi(argv[3])!=0;
    ds4_gpu_set_ssd_streaming(streamed);
    /* Exercise row-at-a-time, pairs, and the complete grouped batch. */
    ds4_gpu_set_streaming_expert_cache_budget(atoi(argv[3])==3?20:atoi(argv[3])==2?4:2);
    assert(ds4_gpu_set_model_map(m.map,m.size) && ds4_gpu_set_model_fd(m.fd));
    unsigned chunk=atoi(argv[4]);
    ds4_muse_gpu_graph g,last_graph;
    ds4_muse_gpu_graph invalid_graph;
    assert(!muse_graph_alloc(&invalid_graph,context,chunk,0));
    assert(!muse_graph_alloc(&invalid_graph,context,chunk,chunk+1));
    assert(muse_graph_bytes(32,3)-muse_graph_bytes(16,3)==16*(d/2)*4);
    assert(muse_graph_alloc(&g,context,chunk,chunk));
    assert(muse_graph_alloc(&last_graph,context,chunk,1));
    assert(ds4_gpu_tensor_bytes(g.logits)==(uint64_t)chunk*DS4_N_VOCAB*4);
    assert(ds4_gpu_tensor_bytes(last_graph.logits)==(uint64_t)DS4_N_VOCAB*4);
    g.matmul_prefill=last_graph.matmul_prefill=matrix;
    FILE *out=fopen(argv[2],"wb"); assert(out);
    int ids[count];for(unsigned i=0;i<count;i++)ids[i]=1+i%30;
    float logits[count*32],first_logits[32];
    for(unsigned first=0;first<count;) {
        unsigned n=MIN(chunk,count-first);
        observed_layers=0;
        assert(muse_graph_forward(&g,&m,&w,ids+first,n,logits,true));
        assert(observed_layers==DS4_N_LAYER);
        if(!first) memcpy(first_logits,logits,sizeof(first_logits));
        float last[32];
        if(n>1) {
            uint32_t position=last_graph.pos;
            assert(!muse_graph_forward(&last_graph,&m,&w,ids+first,n,last,true));
            assert(last_graph.valid && last_graph.pos==position);
        }
        assert(muse_graph_forward(&last_graph,&m,&w,ids+first,n,last,false));
        /* Reference matvecs keep strict batch invariance. The optional full-
         * FP32 matrix profile has an independent FP64 oracle below; its final
         * one-row output projection can round differently from all-row MMA. */
        if(matrix) for(unsigned i=0;i<32;i++)assert(fabsf(last[i]-logits[(n-1)*32+i])<5e-5f);
        else assert(memcmp(last,logits+(n-1)*32,sizeof(last))==0);
        assert(fwrite(logits,32*4,n,out)==n);
        first+=n;
    }
    if(streamed) {
        ds4_gpu_set_streaming_expert_cache_budget(2);
        ds4_muse_gpu_graph failure;assert(muse_graph_alloc(&failure,16,1,1));
        ds4_tensor invalid=*w.layer[0].ffn_pre_exp_norm;invalid.abs_offset=m.size;
        ds4_weights broken=w;broken.layer[0].ffn_pre_exp_norm=&invalid;
        assert(!muse_graph_forward(&failure,&m,&broken,ids,1,NULL,false));
        assert(!failure.valid && !ds4_gpu_commands_active());
        assert(ds4_gpu_stream_expert_cache_current_count()<=DS4_N_EXPERT_USED);
        muse_graph_free(&failure);
        ds4_gpu_set_streaming_expert_cache_budget(2);
        assert(muse_graph_alloc(&failure,16,1,1));
        invalid=*w.layer[0].ffn_gate_exps;invalid.type=UINT32_MAX;
        broken=w;broken.layer[0].ffn_gate_exps=&invalid;
        assert(!muse_graph_forward(&failure,&m,&broken,ids,1,NULL,false));
        assert(!failure.valid && !ds4_gpu_commands_active());
        /* Invalid latent weights fail before event submission. Invalid expert
         * format fails after prefetch starts, before expert dispatch: installed
         * entries prove pending reads were joined on the latter failure. */
        assert(ds4_gpu_stream_expert_cache_current_count()==DS4_N_EXPERT_USED);
        muse_graph_free(&failure);
        assert(muse_graph_alloc(&failure,16,1,1));float recovered[32];
        assert(muse_graph_forward(&failure,&m,&w,ids,1,recovered,false));
        for(unsigned i=0;i<32;i++) assert(fabsf(recovered[i]-first_logits[i])<2e-5f);
        muse_graph_free(&failure);
    }
    int invalid=32;
    assert(!muse_graph_forward(&g,&m,&w,&invalid,1,logits,true));
    assert(g.valid && g.pos==count);
    assert(fclose(out)==0);
    muse_graph_free(&last_graph);
    muse_graph_free(&g); ds4_gpu_cleanup(); model_close(&m);
    return 0;
}
