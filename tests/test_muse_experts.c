#define _DARWIN_C_SOURCE
#include "ds4_muse_gpu.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

int main(void) {
    assert(!setenv("DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY","1",1));
    enum { D=64, F=64, E=4, S=2, T=3, BANK=E*D*F*4, SIZE=4*BANK };
    FILE *file = tmpfile();
    assert(file && ftruncate(fileno(file),SIZE)==0);
    float *map = mmap(NULL,SIZE,PROT_READ|PROT_WRITE,MAP_SHARED,fileno(file),0);
    assert(map != MAP_FAILED);
    for (unsigned i=0;i<3*BANK/4;i++) map[i]=((int)((i*13+i/(D*F))%19)-9)*.015625f;
    float *norm = (float *)((unsigned char *)map+3*BANK);
    for (unsigned i=0;i<D;i++) norm[i]=.75f+i*.01f;
    assert(msync(map,SIZE,MS_SYNC)==0 && ds4_gpu_init());
    assert(ds4_gpu_set_model_map(map,SIZE) && ds4_gpu_set_model_fd(fileno(file)));
    float input[T*D], weights[T*S]={.2f,.8f,.3f,.7f,.4f,.6f};
    int32_t ids[T*S]={0,1,1,2,0,3};
    for (unsigned i=0;i<T*D;i++) input[i]=((int)(i%17)-8)*.0625f;
    ds4_gpu_tensor *x=ds4_gpu_tensor_alloc(sizeof(input)), *selected=ds4_gpu_tensor_alloc(sizeof(ids));
    ds4_gpu_tensor *mid=ds4_gpu_tensor_alloc(T*S*F*4), *part=ds4_gpu_tensor_alloc(T*S*D*4);
    ds4_gpu_tensor *w=ds4_gpu_tensor_alloc(sizeof(weights)), *out=ds4_gpu_tensor_alloc(T*D*4);
    assert(x&&selected&&mid&&part&&w&&out);
    assert(ds4_gpu_tensor_write(x,0,input,sizeof(input)) && ds4_gpu_tensor_write(selected,0,ids,sizeof(ids)));
    assert(ds4_gpu_tensor_write(w,0,weights,sizeof(weights)));
    double expected[T*D];
    for(unsigned t=0;t<T;t++) for(unsigned d=0;d<D;d++) expected[t*D+d]=0;
    for(unsigned t=0;t<T;t++) for(unsigned s=0;s<S;s++) {
        unsigned e=ids[t*S+s]; double hidden[F], raw[D], ss=0;
        for(unsigned r=0;r<F;r++) {
            double g=0,u=0;
            for(unsigned d=0;d<D;d++) {
                g+=map[(e*F+r)*D+d]*input[t*D+d];
                u+=map[BANK/4+(e*F+r)*D+d]*input[t*D+d];
            }
            hidden[r]=g/(1+exp(-g))*u;
        }
        for(unsigned d=0;d<D;d++) {
            raw[d]=0;
            for(unsigned r=0;r<F;r++) raw[d]+=map[2*BANK/4+(e*D+d)*F+r]*hidden[r];
            raw[d]*=norm[d]; ss+=raw[d]*raw[d];
        }
        double inv=1/sqrt(ss/D+1e-8);
        for(unsigned d=0;d<D;d++) expected[t*D+d]+=weights[t*S+s]*raw[d]*inv;
    }
    for(int streamed=0;streamed<2;streamed++) {
        ds4_gpu_set_ssd_streaming(streamed);
        ds4_gpu_set_streaming_expert_cache_budget(S);
        ds4_gpu_set_streaming_expert_cache_expert_bytes(D*F*4*3);
        assert(ds4_gpu_begin_commands());
        if(streamed) {
            const ds4_gpu_stream_expert_table table={.model_map=map,.model_size=SIZE,
                .layer=0,.n_total_expert=E,.gate_offset=0,.up_offset=BANK,.down_offset=2*BANK,
                .gate_expert_bytes=D*F*4,.down_expert_bytes=D*F*4};
            ds4_gpu_stream_expert_table bad=table;bad.down_offset=SIZE;
            assert(!ds4_gpu_muse_prefetch_experts_tensor(&bad,selected,S,0));
            int32_t invalid[S]={-1,0};
            assert(ds4_gpu_tensor_write(selected,0,invalid,sizeof(invalid)));
            assert(!ds4_gpu_muse_prefetch_experts_tensor(&table,selected,S,0));
            int32_t stale[S]={3,2};
            assert(ds4_gpu_tensor_write(selected,0,stale,sizeof(stale)));
            assert(ds4_gpu_muse_prefetch_experts_tensor(&table,selected,S,0));
            assert(ds4_gpu_stream_expert_cache_current_count()<=S);
            /* Abandon that selection and use a different one. The pending
             * reads must be joined without substituting stale routing IDs. */
            assert(ds4_gpu_tensor_write(selected,0,ids,sizeof(ids)));
            assert(ds4_gpu_muse_input_norm_tensor(mid,x,map,SIZE,3*BANK,T,D,1e-8f));
        }
        assert(ds4_gpu_muse_experts_tensor(mid,part,x,selected,map,SIZE,0,BANK,2*BANK,0,0,T,E,S,D,F,0,NULL,NULL));
        assert(ds4_gpu_muse_input_norm_tensor(part,part,map,SIZE,3*BANK,T*S,D,1e-8f));
        assert(ds4_gpu_qwen4_moe_reduce_tensor(out,part,w,NULL,NULL,NULL,NULL,T,S,S,D,0));
        assert(ds4_gpu_end_commands());
        assert(ds4_gpu_muse_prefetch_finish());
        float actual[T*D]; assert(ds4_gpu_tensor_read(out,0,actual,sizeof(actual)));
        for(unsigned i=0;i<T*D;i++) assert(fabs(actual[i]-expected[i])<3e-5);
        if(streamed) assert(ds4_gpu_stream_expert_cache_current_count()<=S);
    }
    /* The shared frequency cache ages once per decoded token, independently
     * of routed layer count. Batched prefill above must not advance it. */
    for(unsigned token=0;token<19;token++) for(unsigned layer=0;layer<2;layer++) {
        assert(ds4_gpu_muse_experts_tensor(mid,part,x,selected,map,SIZE,0,BANK,2*BANK,0,0,1,E,S,D,F,layer,NULL,NULL));
        assert(ds4_gpu_synchronize());
        assert(ds4_gpu_stream_expert_cache_current_count()<=S);
    }
    FILE *report=tmpfile();assert(report);fflush(stderr);
    int saved_stderr=dup(STDERR_FILENO);assert(saved_stderr>=0);
    assert(dup2(fileno(report),STDERR_FILENO)>=0);
    ds4_gpu_print_memory_report("Muse cache aging regression");fflush(stderr);
    assert(dup2(saved_stderr,STDERR_FILENO)>=0);close(saved_stderr);
    rewind(report);char text[16384]={0};assert(fread(text,1,sizeof(text)-1,report)>0);fclose(report);
    if(!strstr(text,"aging tokens=19 first_layer=0 decay_token=17")) fputs(text,stderr);
    assert(strstr(text,"aging tokens=19 first_layer=0 decay_token=17"));
    /* Read GPU-produced routing at its event while independent latent work
     * remains later in the same submission. Exercise hits and forced reuse. */
    const ds4_gpu_stream_expert_table event_table={.model_map=map,.model_size=SIZE,
        .layer=0,.n_total_expert=E,.gate_offset=0,.up_offset=BANK,.down_offset=2*BANK,
        .gate_expert_bytes=D*F*4,.down_expert_bytes=D*F*4};
    ds4_gpu_tensor *routing=ds4_gpu_tensor_alloc(sizeof(ids));assert(routing);
    for(unsigned round=0;round<4;round++) {
        int32_t picks[S]={(int32_t)(round<2?0:2),(int32_t)(round<2?1:3)};
        int32_t invalid[S]={-1,-1};uint64_t event=0;
        assert(ds4_gpu_tensor_write(selected,0,invalid,sizeof(invalid)));
        assert(ds4_gpu_tensor_write(routing,0,picks,sizeof(picks)));
        assert(ds4_gpu_begin_commands()&&ds4_gpu_tensor_copy(selected,0,routing,0,sizeof(picks)));
        assert(ds4_gpu_signal_selected_readback_ready(&event)&&event);
        assert(ds4_gpu_muse_input_norm_tensor(mid,x,map,SIZE,3*BANK,T,D,1e-8f));
        assert(ds4_gpu_muse_prefetch_experts_tensor(&event_table,selected,S,event));
        assert(ds4_gpu_muse_experts_tensor(mid,part,x,selected,map,SIZE,0,BANK,2*BANK,0,0,1,E,S,D,F,0,NULL,NULL));
        assert(ds4_gpu_end_commands()&&ds4_gpu_muse_prefetch_finish());
        float event_output[S*D],control[S*D];
        assert(ds4_gpu_tensor_read(part,0,event_output,sizeof(event_output)));
        assert(ds4_gpu_muse_experts_tensor(mid,part,x,selected,map,SIZE,0,BANK,2*BANK,0,0,1,E,S,D,F,0,NULL,NULL));
        assert(ds4_gpu_synchronize()&&ds4_gpu_tensor_read(part,0,control,sizeof(control)));
        assert(!memcmp(event_output,control,sizeof(control))&&ds4_gpu_stream_expert_cache_current_count()<=S);
    }
    ds4_gpu_tensor_free(routing);
    ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(selected); ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(part); ds4_gpu_tensor_free(w); ds4_gpu_tensor_free(out);
    ds4_gpu_cleanup(); munmap(map,SIZE); fclose(file);
    puts("Muse expert normalization and bounded native SSD cache: PASS");
}
