/* Test-first private Q6 independent FP32 partial sums, with the existing numerical bound.
 * This intentionally measures a different reduction; it is never installed
 * as a production policy. Preserve exact consistency across batch sizes and
 * check the existing 1e-4 per-operator bound against independent FP64 dots. */
#include "/Users/stephenbelanger/.t3/scratch/2026-10-20-seems-like-muse-spark-open-9f0f95e8/dwarfstar/ds4_metal.m"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <sys/mman.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
typedef struct{uint64_t off,bytes,at;uint32_t type,in,out;} Weight;
static double weight_value(const unsigned char *raw,uint32_t type,uint32_t k,uint32_t row,uint32_t col) {
 if(type==8){const unsigned char *b=raw+(uint64_t)row*(k/32*34)+(col/32)*34;return (double)*(const __fp16 *)b*(int8_t)b[2+col%32];}
 const unsigned char *b=raw+(uint64_t)row*(k/256*210)+(col/256)*210;
 unsigned j=col%256,h=j/128,q=j%128/32,l=j%32;
 unsigned low=h*64+(q&1)*32+l,high=128+h*32+l;
 int code=((b[low]>>((q/2)*4))&15)|(((b[high]>>(q*2))&3)<<4);
 return (double)*(const __fp16 *)(b+208)*(int8_t)b[192+j/16]*(code-32);
}
static unsigned partial_chains;
static void run(bool candidate,Weight w,const void *map,uint64_t size,ds4_gpu_tensor *x,ds4_gpu_tensor *y,uint32_t rows) {
 assert(w.type==14);muse_bind b[3];
 assert(muse_weight_bind(&b[0],map,size,0,w.bytes)&&muse_tensor(&b[1],x,(uint64_t)w.in*rows*4)&&muse_tensor(&b[2],y,(uint64_t)w.out*rows*4));
 struct {uint32_t in_dim,out_dim,rows,type,row_bytes;} a={w.in,w.out,rows,w.type,(uint32_t)(w.bytes/w.out)};
 const char *name=!candidate?"kernel_muse_linear_q6_rows2":partial_chains==2?"kernel_probe_q6_acc2":partial_chains==4?"kernel_probe_q6_acc4":partial_chains==6?"kernel_probe_q6_half2":"kernel_probe_q6_block2";
 assert(muse_dispatch(name,&a,sizeof(a),b,3,MTLSizeMake((w.out+1)/2,rows,1)));
}
int main(int argc,char **argv){@autoreleasepool{
 assert(argc==5&&sizeof(Weight)==40);partial_chains=atoi(argv[4]);assert(partial_chains==2||partial_chains==4||partial_chains==6||partial_chains==8);Weight weights[4];FILE *mf=fopen(argv[2],"rb");assert(mf&&fread(weights,sizeof(weights),1,mf)==1);fclose(mf);
 assert(!setenv("DS4_METAL_MUSE_SOURCE",argv[3],1)&&ds4_gpu_init());
 enum{D=8192,N=8192,TMAX=512};uint64_t bytes=(uint64_t)D*TMAX*4;
 float *input=malloc(bytes),*gold=malloc(bytes),*actual=malloc(bytes),*one_actual=malloc(D*4);assert(input&&gold&&actual&&one_actual);
 for(unsigned i=0;i<D*TMAX;i++)input[i]=sinf(i*.031f)*1.41421356237f;
 uint64_t payloads[3]={bytes,bytes,N*4};ds4_gpu_tensor *roots[3],*views[3];uint32_t guards[16];for(unsigned i=0;i<16;i++)guards[i]=0x7fc00000u;
 for(unsigned i=0;i<3;i++){roots[i]=ds4_gpu_tensor_alloc(payloads[i]+128);assert(roots[i]&&ds4_gpu_tensor_write(roots[i],0,guards,64)&&ds4_gpu_tensor_write(roots[i],payloads[i]+64,guards,64));views[i]=ds4_gpu_tensor_view(roots[i],64,payloads[i]);assert(views[i]);}
 ds4_gpu_tensor *x=views[0],*y=views[1],*one=views[2];assert(ds4_gpu_tensor_write(x,0,input,bytes));
 void *maps[2];uint64_t sizes[2];int fd=open(argv[1],O_RDONLY);assert(fd>=0);
 for(unsigned kind=0;kind<1;kind++) {
  Weight w=weights[kind?3:0];assert(w.in==D&&w.out==N);sizes[kind]=(w.bytes+16383)/16384*16384;
  unsigned char *raw=mmap(NULL,sizes[kind],PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);assert(raw!=MAP_FAILED);maps[kind]=raw;
  uint64_t got=0;while(got<w.bytes){ssize_t n=pread(fd,raw+got,w.bytes-got,w.off+got);assert(n>0);got+=n;}
  assert(ds4_gpu_set_model_map(raw,sizes[kind]));
  const unsigned shapes[]={1,3,33,512};
  for(unsigned shape=0;shape<4;shape++) {
   unsigned rows=shapes[shape];run(false,w,raw,sizes[kind],x,y,rows);assert(ds4_gpu_tensor_read(y,0,gold,(uint64_t)rows*N*4));
   run(true,w,raw,sizes[kind],x,y,rows);assert(ds4_gpu_tensor_read(y,0,actual,(uint64_t)rows*N*4));
   double worst=0,old_error=0,mma_error=0;unsigned differing=0;
   for(unsigned i=0;i<rows*N;i++){assert(isfinite(actual[i])&&isfinite(gold[i]));worst=fmax(worst,fabs((double)actual[i]-gold[i]));if(memcmp(gold+i,actual+i,4))differing++;}
   for(unsigned sample=0;sample<256;sample++) {
    unsigned t=(sample*13u)%rows,n=(sample*2654435761u)%N;double expected=0;
    for(unsigned k=0;k<D;k++)expected+=weight_value(raw,w.type,D,n,k)*(double)input[t*D+k];
    old_error=fmax(old_error,fabs(expected-gold[t*N+n]));mma_error=fmax(mma_error,fabs(expected-actual[t*N+n]));
   }
   printf("Q6 FP32 partial sums numerical type=%u rows=%u baseline_differing=%u baseline_max=%g old_FP64_error=%g mma_FP64_error=%g\n",w.type,rows,differing,worst,old_error,mma_error);fflush(stdout);
   assert(worst<1e-4&&old_error<1e-4&&mma_error<1e-4);
   for(unsigned t=0;t<rows;t++) {
    ds4_gpu_tensor *view=ds4_gpu_tensor_view(x,(uint64_t)t*D*4,D*4);assert(view);run(true,w,raw,sizes[kind],view,one,1);
    assert(ds4_gpu_tensor_read(one,0,one_actual,N*4)&&!memcmp(one_actual,actual+t*N,N*4));ds4_gpu_tensor_free(view);
   }
   if(rows!=1)continue;
   double elapsed[2]={0},gpu[2]={0};unsigned rounds=rows==512?6:12,reps=rows==512?1:16;
   for(unsigned r=0;r<rounds;r++)for(unsigned order=0;order<2;order++) {
    unsigned v=(r+order)&1;run(v,w,raw,sizes[kind],x,y,rows);assert(ds4_gpu_synchronize());
    double start=now();assert(ds4_gpu_begin_commands());for(unsigned rep=0;rep<reps;rep++)run(v,w,raw,sizes[kind],x,y,rows);id<MTLCommandBuffer> timed_cb=g_batch_cb;assert(timed_cb);assert(ds4_gpu_end_commands());[timed_cb waitUntilCompleted];elapsed[v]+=now()-start;assert(timed_cb.GPUEndTime>timed_cb.GPUStartTime);gpu[v]+=timed_cb.GPUEndTime-timed_cb.GPUStartTime;
   }
   printf("Q6 split GPU reference_ms=%.6f candidate_ms=%.6f speedup=%.6f\n",gpu[0]*1000/(rounds*reps),gpu[1]*1000/(rounds*reps),gpu[0]/gpu[1]);
   printf("Q6 FP32 partial sums type=%u rows=%u native_ms=%.6f mma_ms=%.6f ratio=%.6f batch_bits_exact=yes\n",w.type,rows,elapsed[0]*1000/(rounds*reps),elapsed[1]*1000/(rounds*reps),elapsed[0]/elapsed[1]);fflush(stdout);
  }
 }
 for(unsigned i=0;i<3;i++){uint32_t got[16];assert(ds4_gpu_tensor_read(roots[i],0,got,64)&&!memcmp(got,guards,64));assert(ds4_gpu_tensor_read(roots[i],payloads[i]+64,got,64)&&!memcmp(got,guards,64));}
 assert(ds4_gpu_tensor_read(x,0,actual,bytes)&&!memcmp(actual,input,bytes));
 puts("Q6 split FP64 bound, offset guards, input nonmutation and batch consistency: PASS");
 close(fd);ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(y);ds4_gpu_tensor_free(one);for(unsigned i=0;i<3;i++)ds4_gpu_tensor_free(roots[i]);ds4_gpu_cleanup();for(unsigned i=0;i<1;i++)munmap(maps[i],sizes[i]);free(input);free(gold);free(actual);free(one_actual);return 0;
}}
