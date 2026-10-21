#define _DARWIN_C_SOURCE
#include "ds4_muse_gpu.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
int main(void) {
    enum {D=256,TMAX=16,SIZE=32768};
    const uint32_t types[4]={14,8,0,8},widths[4]={19,13,7,11};
    uint64_t offsets[4],cursor=0;
    unsigned char *map=mmap(NULL,SIZE,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
    assert(map!=MAP_FAILED);
    uint32_t random=19;
    for(unsigned i=0;i<SIZE;i++){random=random*1664525u+1013904223u;map[i]=random>>24;}
    for(unsigned i=0;i<4;i++) {
        cursor=(cursor+31)/32*32;offsets[i]=cursor;
        unsigned row=types[i]==14?210:types[i]==8?272:D*4;
        for(unsigned r=0;r<widths[i];r++) {
            unsigned char *p=map+cursor+(uint64_t)r*row;
            if(types[i]==0) {
                for(unsigned d=0;d<D;d++){float v=((int)((d*3+r*7)%19)-9)*.0078125f;memcpy(p+d*4,&v,4);}
            } else {
                unsigned block=types[i]==14?210:34;
                for(unsigned b=0;b<row/block;b++){uint16_t scale=0x1800+r%16;memcpy(p+b*block+(types[i]==14?208:0),&scale,2);}
            }
        }
        cursor+=(uint64_t)row*widths[i];assert(cursor<=SIZE);
    }
    assert(ds4_gpu_init()&&ds4_gpu_set_model_map(map,SIZE));
    ds4_gpu_tensor *input_store=ds4_gpu_tensor_alloc(D*TMAX*4+16),
        *x=ds4_gpu_tensor_view(input_store,16,D*TMAX*4),*storage=ds4_gpu_tensor_alloc(4096*TMAX),*out[4];
    assert(input_store&&x&&storage);float input[D*TMAX];
    for(unsigned i=0;i<D*TMAX;i++) input[i]=sinf(i*.031f)*.17f;
    assert(ds4_gpu_tensor_write(x,0,input,sizeof(input)));
    float gold[4][TMAX*19],actual[TMAX*19];cursor=4;
    for(unsigned i=0;i<4;i++){out[i]=ds4_gpu_tensor_view(storage,cursor,TMAX*widths[i]*4);assert(out[i]);cursor+=TMAX*widths[i]*4+16;}
    const unsigned rows[]={1,3,4,5,16};
    for(unsigned reference=0;reference<2;reference++) {
        if(reference) assert(!setenv("DS4_METAL_MUSE_PROJECTIONS_REFERENCE","1",1));
        else assert(!unsetenv("DS4_METAL_MUSE_PROJECTIONS_REFERENCE"));
        for(unsigned shape=0;shape<sizeof(rows)/sizeof(*rows);shape++) {
            unsigned t=rows[shape];
            assert(ds4_gpu_begin_commands());
            for(unsigned i=0;i<4;i++) assert(ds4_gpu_muse_linear_tensor(out[i],map,SIZE,offsets[i],types[i],D,widths[i],x,t));
            assert(ds4_gpu_muse_logits_tensor(out[0],t*widths[0],.5f,20.f)&&ds4_gpu_end_commands());
            for(unsigned i=0;i<4;i++) assert(ds4_gpu_tensor_read(out[i],0,gold[i],t*widths[i]*4));
            /* Without a caller-owned batch, use the existing serial owned-CB
             * path and leave no command scope behind. */
            assert(ds4_gpu_muse_linears_tensor(out,map,SIZE,offsets,types,widths,4,D,x,t));
            assert(!ds4_gpu_commands_active());
            assert(ds4_gpu_muse_logits_tensor(out[0],t*widths[0],.5f,20.f));
            for(unsigned i=0;i<4;i++) assert(ds4_gpu_tensor_read(out[i],0,actual,t*widths[i]*4)&&!memcmp(actual,gold[i],t*widths[i]*4));
            for(unsigned count=1;count<=4;count++) {
                assert(ds4_gpu_begin_commands());
                assert(ds4_gpu_muse_linears_tensor(out,map,SIZE,offsets,types,widths,count,D,x,t));
                assert(ds4_gpu_muse_logits_tensor(out[0],t*widths[0],.5f,20.f)&&ds4_gpu_end_commands());
                for(unsigned i=0;i<count;i++) assert(ds4_gpu_tensor_read(out[i],0,actual,t*widths[i]*4)&&!memcmp(actual,gold[i],t*widths[i]*4));
            }
            for(unsigned repeat=0;repeat<12;repeat++) {
                assert(ds4_gpu_begin_commands());
                assert(!ds4_gpu_muse_linears_tensor(NULL,map,SIZE,offsets,types,widths,4,D,x,t));
                assert(!ds4_gpu_muse_linears_tensor(out,map,SIZE,NULL,types,widths,4,D,x,t));
                assert(!ds4_gpu_muse_linears_tensor(out,map,SIZE,offsets,NULL,widths,4,D,x,t));
                assert(!ds4_gpu_muse_linears_tensor(out,map,SIZE,offsets,types,NULL,4,D,x,t));
                assert(!ds4_gpu_muse_linears_tensor(out,map,SIZE,offsets,types,widths,4,D,x,0));
                assert(!ds4_gpu_muse_linears_tensor(out,map,SIZE,offsets,types,widths,4,0,x,t));
                assert(!ds4_gpu_muse_linears_tensor(out,map,SIZE,offsets,types,widths,0,D,x,t));
                assert(!ds4_gpu_muse_linears_tensor(out,map,SIZE,offsets,types,widths,5,D,x,t));
                ds4_gpu_tensor *overlap[4]={out[0],out[0],out[2],out[3]};
                assert(!ds4_gpu_muse_linears_tensor(overlap,map,SIZE,offsets,types,widths,4,D,x,t));
                overlap[0]=x;overlap[1]=out[1];
                assert(!ds4_gpu_muse_linears_tensor(overlap,map,SIZE,offsets,types,widths,4,D,x,t));
                /* Failure after earlier dispatches must close any concurrent
                 * section and leave the caller's batch usable for recovery. */
                uint32_t bad[4]={types[0],types[1],UINT32_MAX,types[3]};
                assert(!ds4_gpu_muse_linears_tensor(out,map,SIZE,offsets,bad,widths,4,D,x,t));
                assert(ds4_gpu_commands_active());
                assert(ds4_gpu_muse_linears_tensor(out,map,SIZE,offsets,types,widths,4,D,x,t));
                /* A dependent consumer proves the projection scope ended. */
                assert(ds4_gpu_muse_logits_tensor(out[0],t*widths[0],.5f,20.f)&&ds4_gpu_end_commands());
                for(unsigned i=0;i<4;i++) assert(ds4_gpu_tensor_read(out[i],0,actual,t*widths[i]*4)&&!memcmp(actual,gold[i],t*widths[i]*4));
            }
        }
    }
    unsetenv("DS4_METAL_MUSE_PROJECTIONS_REFERENCE");
    for(unsigned i=0;i<4;i++) ds4_gpu_tensor_free(out[i]);
    ds4_gpu_tensor_free(storage);ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(input_store);
    ds4_gpu_cleanup();munmap(map,SIZE);
    puts("Muse independent projections, exact consumers, disjoint views and failure recovery: PASS");
    return 0;
}
