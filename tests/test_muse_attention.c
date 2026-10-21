#include "ds4_muse_gpu.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool sentinel_last, flash, stress;


static void check_shape(unsigned T, unsigned cap, unsigned window, bool rope, unsigned tail_rows, unsigned H, unsigned KV) {
    enum { D = 64 };
    size_t qbytes=(size_t)T*H*D*sizeof(float),kbytes=(size_t)T*KV*D*sizeof(float);
    float *q=malloc(qbytes),*qr=malloc(qbytes),*k=malloc(kbytes),*v=malloc(kbytes),*kr=malloc(kbytes);
    assert(q && qr && k && v && kr);
    for (unsigned i = 0; i < T*H*D; i++) q[i] = ((int)((i*17)%23)-11)*.0625f;
    for (unsigned i = 0; i < T*KV*D; i++) {
        k[i] = ((int)((i*7)%19)-9)*.0625f;
        v[i] = ((int)((i*13)%29)-14)*.125f;
    }
    if(stress) {
        /* Approximate Muse's large learned query norm and unit key norm. */
        for(unsigned i=0;i<T*H*D;i++) q[i]*=14;
        for(unsigned i=0;i<T*KV*D;i++) k[i]*=3;
    }
    if (tail_rows && !window && !rope) {
        /* An early key must remain visible thousands of tokens later. A
         * mistaken sliding window would return zero, far outside tolerance. */
        for(unsigned i=0;i<T*H*D;i++) q[i]=0;
        for(unsigned i=0;i<T*KV*D;i++) k[i]=v[i]=0;
        for(unsigned t=T-tail_rows;t<T;t++) for(unsigned h=0;h<H;h++) q[(t*H+h)*D]=8;
        for(unsigned h=0;h<KV;h++) {
            unsigned offset=sentinel_last?(T-1)*KV*D:0;
            k[offset+h*D]=8;
            for(unsigned d=0;d<D;d++) v[offset+h*D+d]=.03125f*(d+1)*(h+1);
        }
    }
    for (unsigned t = 0; t < T; t++) for (unsigned h = 0; h < H; h++)
    for (unsigned d = 0; d < D; d += 2) {
        double angle = rope ? t * pow(500000., -(double)d/D) : 0;
        unsigned b = (t*H+h)*D+d;
        qr[b] = q[b]*cos(angle)-q[b+1]*sin(angle);
        qr[b+1] = q[b]*sin(angle)+q[b+1]*cos(angle);
        if (h < KV) {
            b = (t*KV+h)*D+d;
            kr[b] = (float)(_Float16)(k[b]*cos(angle)-k[b+1]*sin(angle));
            kr[b+1] = (float)(_Float16)(k[b]*sin(angle)+k[b+1]*cos(angle));
        }
    }
    ds4_gpu_tensor *gq = ds4_gpu_tensor_alloc(qbytes);
    ds4_gpu_tensor *gk = ds4_gpu_tensor_alloc(kbytes);
    ds4_gpu_tensor *gv = ds4_gpu_tensor_alloc(kbytes);
    ds4_gpu_tensor *kroot=ds4_gpu_tensor_alloc((uint64_t)cap*KV*D*2+32);
    ds4_gpu_tensor *vroot=ds4_gpu_tensor_alloc((uint64_t)cap*KV*D*2+32);assert(kroot&&vroot);
    ds4_gpu_tensor *kc=ds4_gpu_tensor_view(kroot,16,(uint64_t)cap*KV*D*2);
    ds4_gpu_tensor *vc=ds4_gpu_tensor_view(vroot,16,(uint64_t)cap*KV*D*2);
    uint16_t guard[8];for(unsigned i=0;i<8;i++)guard[i]=0x7e00;
    assert(ds4_gpu_tensor_write(kroot,0,guard,16)&&ds4_gpu_tensor_write(kroot,(uint64_t)cap*KV*D*2+16,guard,16));
    assert(ds4_gpu_tensor_write(vroot,0,guard,16)&&ds4_gpu_tensor_write(vroot,(uint64_t)cap*KV*D*2+16,guard,16));
    assert(gq && gk && gv && kc && vc);
    assert(ds4_gpu_tensor_write(gq, 0, q, qbytes));
    assert(ds4_gpu_tensor_write(gk, 0, k, kbytes));
    assert(ds4_gpu_tensor_write(gv, 0, v, kbytes));
    assert(ds4_gpu_begin_commands());
    assert(ds4_gpu_muse_kv_store_tensor(gq, kc, vc, gk, gv, T,H,KV,D,0,cap,500000.f,rope));
    size_t cache_bytes=(size_t)cap*KV*D*2;
    ds4_gpu_tensor *saved_k=ds4_gpu_tensor_alloc(cache_bytes),*saved_v=ds4_gpu_tensor_alloc(cache_bytes);
    assert(saved_k&&saved_v);
    assert(ds4_gpu_tensor_copy(saved_k,0,kc,0,cache_bytes));
    assert(ds4_gpu_tensor_copy(saved_v,0,vc,0,cache_bytes));
    unsigned start = tail_rows ? T-tail_rows : cap < T ? T-2 : 0, rows = T-start;
    ds4_gpu_tensor *query = ds4_gpu_tensor_view(gq,start*H*D*4,rows*H*D*4);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(rows*H*D*4);
    ds4_gpu_tensor *part = ds4_gpu_tensor_alloc(ds4_gpu_muse_attention_scratch_bytes(rows,H,D));
    assert(query && out && part);
    assert((flash ? ds4_gpu_muse_attention_flash_tensor : ds4_gpu_muse_attention_tensor)
        (out,query,kc,vc,part,rows,H,KV,D,start,cap,window));
    assert(ds4_gpu_end_commands());
    float *actual=malloc((size_t)rows*H*D*sizeof(float));assert(actual);
    assert(ds4_gpu_synchronize() && ds4_gpu_tensor_read(out,0,actual,rows*H*D*4));
    if(rows>256) {
        /* A bounded scratch tile must preserve each query's original key
         * partition and reduction, including a nonzero query view offset. */
        float *single=malloc((size_t)rows*H*D*4);assert(single);
        assert(ds4_gpu_begin_commands());
        for(unsigned row=0;row<rows;row++) {
            ds4_gpu_tensor *one_q=ds4_gpu_tensor_view(query,(uint64_t)row*H*D*4,(uint64_t)H*D*4);
            ds4_gpu_tensor *one_o=ds4_gpu_tensor_view(out,(uint64_t)row*H*D*4,(uint64_t)H*D*4);
            assert(one_q&&one_o);
            assert(ds4_gpu_muse_attention_tensor(one_o,one_q,kc,vc,part,1,H,KV,D,start+row,cap,window));
            ds4_gpu_tensor_free(one_q);ds4_gpu_tensor_free(one_o);
        }
        assert(ds4_gpu_end_commands());
        assert(ds4_gpu_tensor_read(out,0,single,(size_t)rows*H*D*4));
        assert(!memcmp(single,actual,(size_t)rows*H*D*4));free(single);
    }
    if(flash && rows==1 && (T<cap?T:cap)<1024) {
        /* Small contexts retain the original arithmetic and launch policy. */
        float *reference=malloc((size_t)rows*H*D*4);assert(reference);
        assert(ds4_gpu_begin_commands());
        assert(ds4_gpu_muse_attention_tensor(out,query,kc,vc,part,rows,H,KV,D,start,cap,window));
        assert(ds4_gpu_end_commands());
        assert(ds4_gpu_tensor_read(out,0,reference,(size_t)rows*H*D*4));
        assert(!memcmp(actual,reference,(size_t)rows*H*D*4));free(reference);
    }
    if(stress) {
        /* Isolate attention at learned-norm magnitudes. The independent FP64
         * dot/softmax uses the actual FP32/F16 RoPE outputs: GPU and CPU angle
         * evaluation can legitimately round differently before quantization.
         * Normal-size cases above still check the end-to-end RoPE reference. */
        assert(ds4_gpu_tensor_read(query,0,qr+(size_t)start*H*D,(size_t)rows*H*D*4));
        _Float16 *stored=malloc(cache_bytes);assert(stored);
        assert(ds4_gpu_tensor_read(saved_k,0,stored,cache_bytes));
        for(unsigned t=T>cap?T-cap:0;t<T;t++) for(unsigned i=0;i<KV*D;i++)
            kr[(size_t)t*KV*D+i]=(float)stored[(size_t)(t%cap)*KV*D+i];
        free(stored);
    }
    double max_error=0;
    for (unsigned t = start; t < T; t++) for (unsigned h = 0; h < H; h++) {
        unsigned lo = window && t+1>window ? t+1-window : 0;
        double score[T], sum=0;
        for (unsigned p=lo;p<=t;p++) {
            score[p]=0;
            for (unsigned d=0;d<D;d++) score[p] += qr[(t*H+h)*D+d]*kr[(p*KV+h/(H/KV))*D+d]/sqrt(D);
            score[p]=exp(score[p]); sum+=score[p];
        }
        for (unsigned d=0;d<D;d++) {
            double expected=0;
            for (unsigned p=lo;p<=t;p++) expected += score[p]/sum*v[(p*KV+h/(H/KV))*D+d];
            double error=fabs(actual[((t-start)*H+h)*D+d]-expected);
            if(error>max_error) max_error=error;
            assert(isfinite(actual[((t-start)*H+h)*D+d]) && error<3e-5);
        }
    }
    if(tail_rows) printf("Muse %u-token %s tail parity: max error %.9g PASS\n",T,window?"sliding-window":"global NoPE",max_error);
    assert(!(flash ? ds4_gpu_muse_attention_flash_tensor : ds4_gpu_muse_attention_tensor)
        (out,query,kc,vc,part,rows,H,KV,D,start,1,window));
    ds4_gpu_tensor_free(part); ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(query);
    /* Compare bytes captured on the GPU before attention, so this checks
     * nonmutation without requiring CPU/GPU RoPE to round identically. */
    void *before=malloc(cache_bytes),*after=malloc(cache_bytes);assert(before&&after);
    assert(ds4_gpu_tensor_read(saved_k,0,before,cache_bytes));
    assert(ds4_gpu_tensor_read(kc,0,after,cache_bytes)&&!memcmp(before,after,cache_bytes));
    assert(ds4_gpu_tensor_read(saved_v,0,before,cache_bytes));
    assert(ds4_gpu_tensor_read(vc,0,after,cache_bytes)&&!memcmp(before,after,cache_bytes));
    free(before);free(after);ds4_gpu_tensor_free(saved_k);ds4_gpu_tensor_free(saved_v);
    uint16_t actual_guard[8];
    assert(ds4_gpu_tensor_read(kroot,0,actual_guard,16)&&!memcmp(actual_guard,guard,16));
    assert(ds4_gpu_tensor_read(kroot,(uint64_t)cap*KV*D*2+16,actual_guard,16)&&!memcmp(actual_guard,guard,16));
    assert(ds4_gpu_tensor_read(vroot,0,actual_guard,16)&&!memcmp(actual_guard,guard,16));
    assert(ds4_gpu_tensor_read(vroot,(uint64_t)cap*KV*D*2+16,actual_guard,16)&&!memcmp(actual_guard,guard,16));
    ds4_gpu_tensor_free(vc); ds4_gpu_tensor_free(kc); ds4_gpu_tensor_free(gv);
    ds4_gpu_tensor_free(gk); ds4_gpu_tensor_free(gq);
    ds4_gpu_tensor_free(kroot);ds4_gpu_tensor_free(vroot);
    free(actual);free(kr);free(qr);free(v);free(k);free(q);
}

static void check_rows(unsigned T,unsigned cap,unsigned window,bool rope,unsigned tail) {
    check_shape(T,cap,window,rope,tail,16,2);
}

static void check(unsigned T,unsigned cap,unsigned window,bool rope) {check_rows(T,cap,window,rope,0);}

int main(int argc,char **argv) {
    assert(argc==1 || (argc==2 && !strcmp(argv[1],"--flash")));
    flash=argc==2;
    assert(ds4_gpu_init());
    check(10,16,0,false); check(10,16,4,true); check(10,7,4,true);
    check(257,512,0,false); check(257,512,127,true);
    check(70,128,0,false); check(70,128,35,true); check(70,39,35,true);
    /* Compare only the newest queries, but include every visible key in the
     * independent FP64 reference. Keep 8K tensors on the heap. */
    check_rows(8193,16384,0,false,1);
    check_rows(8193,2560,2048,true,1);
    check_rows(70,128,0,false,1); check_rows(70,128,35,true,1);
    check_rows(8193,8193,0,false,1);
    sentinel_last=true;check_rows(8193,8193,0,false,1);
    /* Real GQA geometry: all sixteen KV heads must copy the newest tail row. */
    check_shape(1025,1025,0,false,1,128,16);
    sentinel_last=false;
    check_rows(256,256,0,false,1);
    check_rows(257,257,0,false,1);
    check_rows(300,263,256,true,1);
    stress=true;
    check_shape(1025,1025,0,true,1,128,16);
    check_shape(1025,263,256,true,1,128,16);
    ds4_gpu_cleanup();
    puts("Muse causal GQA, RoPE/NoPE, and sliding KV ring: PASS");
}
