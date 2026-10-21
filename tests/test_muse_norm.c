#define _DARWIN_C_SOURCE
#include "ds4_muse_gpu.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

/* Parallel memory access must retain the original per-lane FMA chain and
 * SIMD reduction, including in-place operation and unaligned tensor views. */
static void check_norm(void *map, uint64_t size, unsigned width, unsigned rows,
                       unsigned pattern) {
    size_t count=(size_t)width*rows, bytes=count*sizeof(float);
    float *input=malloc(bytes+32),*gold=malloc(bytes),*actual=malloc(bytes+32);
    assert(input&&gold&&actual);
    float *weight=(float *)((char *)map+16);
    for(unsigned i=0;i<width;i++) weight[i]=cosf(i*.017f)*1.3f;
    for(size_t i=0;i<count;i++) input[i]=pattern==0?sinf(i*.031f)*.17f:
        pattern==1?0.f:pattern==2?(i&1?-0x1p-140f:0x1p-140f):
        sinf(i*.031f)*0x1p40f;
    ds4_gpu_tensor *xp=ds4_gpu_tensor_alloc(bytes+32),
        *yp=ds4_gpu_tensor_alloc(bytes+32),
        *x=ds4_gpu_tensor_view(xp,16,bytes),*y=ds4_gpu_tensor_view(yp,16,bytes);
    assert(xp&&yp&&x&&y&&ds4_gpu_tensor_write(x,0,input,bytes));
    assert(!setenv("DS4_METAL_MUSE_NORM_REFERENCE","1",1));
    assert(ds4_gpu_muse_input_norm_tensor(y,x,map,size,16,rows,width,1e-6f));
    assert(ds4_gpu_tensor_read(y,0,gold,bytes));
    assert(!unsetenv("DS4_METAL_MUSE_NORM_REFERENCE"));
    for(unsigned batched=0;batched<2;batched++) {
        for(size_t i=0;i<count+8;i++) actual[i]=NAN;
        assert(ds4_gpu_tensor_write(yp,0,actual,bytes+32));
        if(batched) assert(ds4_gpu_begin_commands());
        assert(ds4_gpu_muse_input_norm_tensor(y,x,map,size,16,rows,width,1e-6f));
        if(batched) assert(ds4_gpu_end_commands());
        assert(ds4_gpu_tensor_read(yp,0,actual,bytes+32));
        assert(!memcmp(actual+4,gold,bytes));
        for(unsigned i=0;i<4;i++) assert(isnan(actual[i])&&isnan(actual[4+count+i]));
        assert(ds4_gpu_tensor_read(x,0,actual,bytes)&&!memcmp(actual,input,bytes));
    }
    assert(ds4_gpu_muse_input_norm_tensor(x,x,map,size,16,rows,width,1e-6f));
    assert(ds4_gpu_tensor_read(x,0,actual,bytes)&&!memcmp(actual,gold,bytes));
    if(pattern==0||pattern==3) for(unsigned row=0;row<rows;row++) {
        double sum=0;
        for(unsigned i=0;i<width;i++) {double v=(double)input[(size_t)row*width+i]*weight[i];sum+=v*v;}
        double inv=1/sqrt(sum/width+1e-6);
        for(unsigned i=0;i<width;i++) {
            double expected=(double)input[(size_t)row*width+i]*weight[i]*inv;
            assert(fabs(expected-gold[(size_t)row*width+i])<3e-5);
        }
    }
    ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(y);ds4_gpu_tensor_free(xp);ds4_gpu_tensor_free(yp);
    free(input);free(gold);free(actual);
}
int main(void) {
    enum {SIZE=65536};
    void *map=mmap(NULL,SIZE,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
    assert(map!=MAP_FAILED&&ds4_gpu_init()&&ds4_gpu_set_model_map(map,SIZE));
    const unsigned widths[]={1,64,4095,4096,4097,8192,8193};
    for(unsigned i=0;i<sizeof(widths)/sizeof(*widths);i++)
        for(unsigned rows=1;rows<=3;rows+=2)
            for(unsigned pattern=0;pattern<4;pattern++) check_norm(map,SIZE,widths[i],rows,pattern);
    ds4_gpu_cleanup();munmap(map,SIZE);
    puts("Muse normalization exact reference, FP64, zero/subnormal, in-place and offset guards: PASS");
    return 0;
}
