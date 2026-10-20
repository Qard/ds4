#include "../ds4.c"
#include <assert.h>

static void equal_logits(ds4_session *a,ds4_session *b) {
    assert(ds4_session_argmax(a)==ds4_session_argmax(b));
    for(unsigned i=0;i<DS4_N_VOCAB;i++) assert(fabsf(a->logits[i]-b->logits[i])<3e-5f);
}
static void rebuild(ds4_session *s,int *ids,int n) {
    char err[256]={0}; ds4_tokens p={.v=ids,.len=n,.cap=n};
    ds4_session_invalidate(s);
    assert(ds4_session_sync(s,&p,err,sizeof(err))==0);
}
static void payload_roundtrips(ds4_engine *e,ds4_session *a,ds4_session *b,int *ids) {
    char err[256]={0};ds4_session_snapshot snap={0},small={0};
    const uint64_t expected=13*4+8*4+32*4+(7+8)*64*4;
    assert(ds4_session_payload_bytes(a)==expected);
    assert(ds4_session_save_snapshot(a,&snap,err,sizeof(err))==0 && snap.len==expected);
    ds4_session_invalidate(b);
    assert(ds4_session_load_snapshot(b,&snap,err,sizeof(err))==0);equal_logits(a,b);
    e->prefill_chunk=1;ds4_session *c=NULL,*d=NULL;
    assert(ds4_session_create(&c,e,24)==0);
    assert(ds4_session_load_snapshot(c,&snap,err,sizeof(err))==0);equal_logits(a,c);
    assert(ds4_session_eval(a,9,err,sizeof(err))==0);
    assert(ds4_session_eval(b,9,err,sizeof(err))==0);
    assert(ds4_session_eval(c,9,err,sizeof(err))==0);equal_logits(a,b);equal_logits(a,c);
    assert(ds4_session_save_snapshot(c,&small,err,sizeof(err))==0);
    e->prefill_chunk=10;assert(ds4_session_create(&d,e,24)==0);
    assert(ds4_session_load_snapshot(d,&small,err,sizeof(err))==0);equal_logits(a,d);
    ds4_session_rewind(d,6);assert(d->muse_graph.pos==0);
    ds4_tokens prefix={.v=ids,.len=6,.cap=8};
    assert(ds4_session_sync(d,&prefix,err,sizeof(err))==0);rebuild(b,ids,6);equal_logits(d,b);
    ds4_session_free(c);ds4_session_free(d);e->prefill_chunk=3;
    assert(ds4_session_load_snapshot(a,&snap,err,sizeof(err))==0);
    assert(ds4_session_load_snapshot(b,&snap,err,sizeof(err))==0);equal_logits(a,b);
    snap.ptr[12*4]^=1;
    assert(ds4_session_load_snapshot(b,&snap,err,sizeof(err))!=0);
    assert(ds4_session_checkpoint_valid(b));equal_logits(a,b);snap.ptr[12*4]^=1;
    uint32_t bad_token=32,original=0;
    memcpy(&original,snap.ptr+13*4,4);memcpy(snap.ptr+13*4,&bad_token,4);
    assert(ds4_session_load_snapshot(b,&snap,err,sizeof(err))!=0);
    assert(ds4_session_checkpoint_valid(b));equal_logits(a,b);
    memcpy(snap.ptr+13*4,&original,4);
    uint32_t nan_bits=0x7fc00000;
    memcpy(&original,snap.ptr+(13+8)*4,4);memcpy(snap.ptr+(13+8)*4,&nan_bits,4);
    assert(ds4_session_load_snapshot(b,&snap,err,sizeof(err))!=0);
    assert(ds4_session_checkpoint_valid(b));equal_logits(a,b);
    memcpy(snap.ptr+(13+8)*4,&original,4);
    FILE *fp=tmpfile();assert(fp);
    assert(fwrite(snap.ptr,1,snap.len-1,fp)==snap.len-1);rewind(fp);
    assert(ds4_session_load_payload(a,fp,snap.len,err,sizeof(err))!=0);
    assert(!ds4_session_checkpoint_valid(a));fclose(fp);rebuild(a,ids,8);equal_logits(a,b);
    fp=tmpfile();uint32_t trailer=0x1234abcd,readback=0;
    assert(fp && fwrite(snap.ptr,1,snap.len,fp)==snap.len && fwrite(&trailer,4,1,fp)==1);rewind(fp);
    assert(ds4_session_load_payload(b,fp,snap.len,err,sizeof(err))==0);
    assert(ftell(fp)==(long)snap.len && fread(&readback,4,1,fp)==1 && readback==trailer);
    equal_logits(a,b);fclose(fp);
    ds4_session_snapshot_free(&snap);ds4_session_snapshot_free(&small);
}
static void collect(void *ud,int token) {token_vec_push(ud,token);}
static void generation_roundtrip(ds4_engine *e,ds4_session *s,int *ids) {
    char err[256]={0};ds4_tokens prompt={.v=ids,.len=6,.cap=8};
    rebuild(s,ids,6);int expected[3];
    for(int i=0;i<3;i++) {expected[i]=ds4_session_argmax(s);assert(ds4_session_eval(s,expected[i],err,sizeof(err))==0);}
    token_vec got={0};
    assert(ds4_engine_generate_argmax(e,&prompt,3,16,collect,NULL,&got,NULL,NULL)==0);
    assert(got.len==3 && memcmp(got.v,expected,sizeof(expected))==0);token_vec_free(&got);
}
static bool cancel_layer(void *ud) { return ++*(int *)ud >= 4; }
static bool cancel(void *ud) {return *(bool *)ud;}
static void progress(void *ud,const char *event,int pos,int total) {
    if(!strcmp(event,"prefill_chunk") && pos<total) *(bool *)ud=true;
}
int main(int argc,char **argv) {
    assert(argc==3); bool streamed=atoi(argv[2])!=0;
    g_ds4_shape=DS4_SHAPE_MUSE_SPARK;
    g_ds4_shape.n_layer=2; g_ds4_shape.n_embd=128; g_ds4_shape.n_vocab=32;
    g_ds4_shape.n_head=2; g_ds4_shape.n_head_kv=1; g_ds4_shape.n_expert=4;
    g_ds4_shape.n_expert_used=2; g_ds4_shape.n_ff_exp=128;
    g_ds4_shape.n_moe_latent=128; g_ds4_shape.n_swa=4;
    g_muse_swa[0]=1;g_muse_swa[1]=0;
    ds4_engine e={0}; e.backend=DS4_BACKEND_METAL; e.metal_ready=true;
    e.power_percent=100; e.prefill_chunk=3; e.ssd_streaming=streamed; e.ssd_streaming_cache_experts=2;
    e.vocab.eos_id=e.vocab.endoftext_id=-1;
    model_open(&e.model,argv[1],true,false);
    weights_bind(&e.weights,&e.model,false,0,UINT32_MAX,true,false);
    assert(ds4_gpu_init()); ds4_gpu_set_ssd_streaming(streamed);
    ds4_gpu_set_streaming_expert_cache_budget(2);
    assert(ds4_gpu_set_model_map(e.model.map,e.model.size) && ds4_gpu_set_model_fd(e.model.fd));
    ds4_session *a=NULL,*b=NULL;
    assert(ds4_session_create(&a,&e,16)==0 && ds4_session_create(&b,&e,16)==0);
    assert(ds4_session_set_power(a,50)!=0 && ds4_engine_set_power(&e,50)!=0);
    assert(ds4_session_power(a)==100 && ds4_session_set_power(a,100)==0);
    assert(ds4_session_set_directional_steering_ffn(a,1)!=0);
    int ids[]={1,2,3,4,5,6,7,8}; char err[256]={0};
    ds4_tokens prompt={.v=ids,.len=6,.cap=8};
    assert(ds4_session_sync(a,&prompt,err,sizeof(err))==0);
    ds4_tokens seed={.v=ids,.len=1,.cap=8};
    assert(ds4_session_sync(b,&seed,err,sizeof(err))==0);
    for(int i=1;i<6;i++) assert(ds4_session_eval(b,ids[i],err,sizeof(err))==0);
    equal_logits(a,b);
    assert(ds4_session_eval(a,7,err,sizeof(err))==0);
    assert(ds4_session_eval_argmax(b,7,err,sizeof(err))>=0); equal_logits(a,b);
    prompt.len=8; assert(ds4_session_sync(a,&prompt,err,sizeof(err))==0);
    assert(ds4_session_eval(b,8,err,sizeof(err))==0); equal_logits(a,b);
    payload_roundtrips(&e,a,b,ids);
    ds4_session_rewind(a,6); assert(ds4_session_argmax(a)==-1);
    prompt.len=6; assert(ds4_session_sync(a,&prompt,err,sizeof(err))==0);
    rebuild(b,ids,6);equal_logits(a,b);
    for(int t=21;t<=22;t++) {
        assert(ds4_session_eval(a,t,err,sizeof(err))==0);
        assert(ds4_session_eval(b,t,err,sizeof(err))==0);equal_logits(a,b);
    }
    ds4_session_rewind(a,1);assert(ds4_session_argmax(a)==-1);
    assert(ds4_session_eval(a,19,err,sizeof(err))==0);
    rebuild(b,ids,1);assert(ds4_session_eval(b,19,err,sizeof(err))==0);equal_logits(a,b);
    ds4_session_invalidate(a); bool stopped=false;
    ds4_session_set_cancel(a,cancel,&stopped);ds4_session_set_progress(a,progress,&stopped);
    assert(ds4_session_sync(a,&prompt,err,sizeof(err))==DS4_SESSION_SYNC_INTERRUPTED);
    assert(ds4_session_pos(a)==3 && ds4_session_checkpoint_valid(a));
    ds4_session_set_cancel(a,NULL,NULL);ds4_session_set_progress(a,NULL,NULL);
    assert(ds4_session_sync(a,&prompt,err,sizeof(err))==0);rebuild(b,ids,6);equal_logits(a,b);
    ds4_session_invalidate(a);int checks=0;
    ds4_session_set_cancel(a,cancel_layer,&checks);
    assert(ds4_session_sync(a,&prompt,err,sizeof(err))==DS4_SESSION_SYNC_INTERRUPTED);
    assert(checks==4 && !ds4_session_checkpoint_valid(a));
    ds4_session_set_cancel(a,NULL,NULL);
    assert(ds4_session_sync(a,&prompt,err,sizeof(err))==0);equal_logits(a,b);
    int invalid=32;ds4_tokens bad={.v=&invalid,.len=1,.cap=1};
    assert(ds4_session_sync(a,&bad,err,sizeof(err))!=0);
    assert(ds4_session_pos(a)==6);equal_logits(a,b);
    ds4_decode_item batch[2]={{.session=a,.token=15},{.session=b,.token=16}};
    assert(ds4_sessions_eval_batch(batch,2,err,sizeof(err))==0);
    assert(ds4_session_pos(a)==7 && ds4_session_pos(b)==7);
    generation_roundtrip(&e,a,ids);
    ds4_session_free(a);ds4_session_free(b);
    assert(e.muse_session_bytes==0);
    ds4_gpu_cleanup();model_close(&e.model);
    puts("Muse sessions: prefix reuse, rewrite, rewind, KV payload restore, cancellation, and batch isolation PASS");
}
