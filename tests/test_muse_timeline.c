#define _DARWIN_C_SOURCE
#include "ds4_muse_gpu.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

/* A routing signal ends one profiled submission and opens another. Both
 * submissions must own their sample records, including latent consumers. */
int main(void) {
    char path[]="/tmp/ds4-muse-timeline-XXXXXX";
    int fd=mkstemp(path);assert(fd>=0);close(fd);
    assert(!setenv("DS4_METAL_ENCODER_TIMELINE",path,1));
    enum { D=64, BYTES=D*4, SIZE=16384 };
    float *map=mmap(NULL,SIZE,PROT_READ|PROT_WRITE,MAP_ANON|MAP_PRIVATE,-1,0);
    assert(map!=MAP_FAILED);
    float input[D];for(unsigned i=0;i<D;i++) {map[i]=1;input[i]=i*.01f;}
    assert(ds4_gpu_init()&&ds4_gpu_set_model_map(map,SIZE));
    ds4_gpu_tensor *x=ds4_gpu_tensor_alloc(BYTES),*y=ds4_gpu_tensor_alloc(BYTES),
        *z=ds4_gpu_tensor_alloc(BYTES);
    assert(x&&y&&z&&ds4_gpu_tensor_write(x,0,input,BYTES));
    uint64_t event=0;
    assert(ds4_gpu_begin_commands());
    assert(ds4_gpu_muse_input_norm_tensor(y,x,map,SIZE,0,1,D,1e-8f));
    assert(ds4_gpu_signal_selected_readback_ready(&event)&&event);
    assert(ds4_gpu_muse_input_norm_tensor(z,y,map,SIZE,0,1,D,1e-8f));
    assert(ds4_gpu_commit_and_wait_selected_readback(event,"timeline regression"));
    assert(ds4_gpu_muse_input_norm_tensor(y,z,map,SIZE,0,1,D,1e-8f));
    assert(ds4_gpu_end_commands());
    ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(y);ds4_gpu_tensor_free(z);
    ds4_gpu_cleanup();munmap(map,SIZE);
    FILE *f=fopen(path,"r");assert(f);char line[2048];
    unsigned batches=0,encoders=0;
    while(fgets(line,sizeof(line),f)) {
        if(line[0]=='B') batches++;
        if(line[0]=='E') encoders++;
    }
    fclose(f);unlink(path);
    printf("Muse routing timeline: batches=%u encoders=%u\n",batches,encoders);
    fflush(stdout);assert(batches==2&&encoders==3);
    puts("Muse routing timeline: PASS");
}
