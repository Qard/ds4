#include "ds4_muse_gpu.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

static double bench_seconds(void) {
    struct timespec ts; assert(!clock_gettime(CLOCK_MONOTONIC,&ts));
    return ts.tv_sec+ts.tv_nsec*1e-9;
}

/* Bounded synthetic Q6_K matrices isolate projection work from SSD/cache
 * effects. Alternate order and require every output bit to match the control. */
static int bench_linear(const char *rollback_env,double minimum_speedup) {
    enum {K=8192,N=8192,TMAX=512,ROW=K/256*210};
    size_t bytes=(size_t)N*ROW;
    unsigned char *map=mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
    assert(map!=MAP_FAILED);uint32_t seed=931;
    for(size_t i=0;i<bytes;i++) {seed=seed*1664525u+1013904223u;map[i]=seed>>24;}
    for(size_t i=0;i<bytes/210;i++) {uint16_t d=0x1800+i%16;memcpy(map+i*210+208,&d,2);}
    float *input=malloc((size_t)TMAX*K*4),*reference=malloc((size_t)TMAX*N*4),*actual=malloc((size_t)TMAX*N*4);
    assert(input&&reference&&actual);
    for(size_t i=0;i<(size_t)TMAX*K;i++) input[i]=sinf(i*.031f)*.17f;
    assert(ds4_gpu_init()&&ds4_gpu_set_model_map(map,bytes));
    ds4_gpu_tensor *x=ds4_gpu_tensor_alloc((size_t)TMAX*K*4),*y=ds4_gpu_tensor_alloc((size_t)TMAX*N*4);
    assert(x&&y&&ds4_gpu_tensor_write(x,0,input,(size_t)TMAX*K*4));
    assert(!setenv(rollback_env,"1",1));
    for(unsigned warm=0;warm<3;warm++)
        assert(ds4_gpu_muse_linear_tensor(y,map,bytes,0,14,K,N,x,TMAX)&&ds4_gpu_synchronize());
    const unsigned shapes[]={1,3,4,5,8,16,512};int rc=0;
    for(unsigned shape=0;shape<sizeof(shapes)/sizeof(shapes[0]);shape++) {
        unsigned rows=shapes[shape],reps=rows==512?1:8;double elapsed[2]={0};
        assert(!setenv(rollback_env,"1",1));
        assert(ds4_gpu_muse_linear_tensor(y,map,bytes,0,14,K,N,x,rows));
        assert(ds4_gpu_tensor_read(y,0,reference,(size_t)rows*N*4));
        for(unsigned round=0;round<6;round++) for(unsigned order=0;order<2;order++) {
            unsigned variant=(round+order)&1;
            assert(!(variant?unsetenv(rollback_env):setenv(rollback_env,"1",1)));
            /* Warm pipeline specialization before the timed batch. */
            assert(ds4_gpu_muse_linear_tensor(y,map,bytes,0,14,K,N,x,rows)&&ds4_gpu_synchronize());
            double start=bench_seconds();assert(ds4_gpu_begin_commands());
            for(unsigned rep=0;rep<reps;rep++) assert(ds4_gpu_muse_linear_tensor(y,map,bytes,0,14,K,N,x,rows));
            assert(ds4_gpu_end_commands());elapsed[variant]+=bench_seconds()-start;
            assert(ds4_gpu_tensor_read(y,0,actual,(size_t)rows*N*4));
            if(memcmp(reference,actual,(size_t)rows*N*4)) {fprintf(stderr,"Q6_K linear drift rows=%u variant=%u\n",rows,variant);rc=1;goto done;}
        }
        double speedup=elapsed[0]/elapsed[1];
        printf("Q6_K K=%u N=%u rows=%u rollback=%s control_ms=%.3f candidate_ms=%.3f speedup=%.4f exact=yes\n",K,N,rows,rollback_env,elapsed[0]*1000/(6*reps),elapsed[1]*1000/(6*reps),speedup);fflush(stdout);
        if(speedup<minimum_speedup) rc=1;
    }
done:
    unsetenv(rollback_env);
    ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(y);ds4_gpu_cleanup();
    munmap(map,bytes);free(input);free(reference);free(actual);return rc;
}

/* Match the two gain-before-RMS shapes used by each Muse layer. */
static int bench_input_norm(double minimum) {
    enum {WIDTH=8192,ROWS=512};size_t bytes=(size_t)WIDTH*ROWS*4;
    float *input=malloc(bytes),*reference=malloc(bytes),*actual=malloc(bytes);
    float *weights=mmap(NULL,WIDTH*4,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
    assert(input&&reference&&actual&&weights!=MAP_FAILED);
    for(unsigned i=0;i<WIDTH;i++) weights[i]=.5f+(i%97)*.01f;
    for(size_t i=0;i<bytes/4;i++) input[i]=sinf(i*.031f)*.17f;
    assert(ds4_gpu_init()&&ds4_gpu_set_model_map(weights,WIDTH*4));
    ds4_gpu_tensor *x=ds4_gpu_tensor_alloc(bytes),*y=ds4_gpu_tensor_alloc(bytes);
    assert(x&&y&&ds4_gpu_tensor_write(x,0,input,bytes));int rc=0;
    for(unsigned warm=0;warm<3;warm++)
        assert(ds4_gpu_muse_input_norm_tensor(y,x,weights,WIDTH*4,0,ROWS,WIDTH,1e-5f)&&ds4_gpu_synchronize());
    const unsigned widths[]={8192,4096,8192},rows[]={1,8,512};
    for(unsigned shape=0;shape<3;shape++) {
        unsigned width=widths[shape],count=rows[shape],reps=shape==2?4:128;
        size_t size=(size_t)width*count*4;double elapsed[2]={0};
        assert(!setenv("DS4_METAL_MUSE_NORM_REFERENCE","1",1));
        assert(ds4_gpu_muse_input_norm_tensor(y,x,weights,WIDTH*4,0,count,width,1e-5f));
        assert(ds4_gpu_tensor_read(y,0,reference,size));
        for(unsigned round=0;round<6;round++) for(unsigned order=0;order<2;order++) {
            unsigned variant=(round+order)&1;
            assert(!(variant?unsetenv("DS4_METAL_MUSE_NORM_REFERENCE"):setenv("DS4_METAL_MUSE_NORM_REFERENCE","1",1)));
            assert(ds4_gpu_muse_input_norm_tensor(y,x,weights,WIDTH*4,0,count,width,1e-5f)&&ds4_gpu_synchronize());
            double start=bench_seconds();assert(ds4_gpu_begin_commands());
            for(unsigned rep=0;rep<reps;rep++) assert(ds4_gpu_muse_input_norm_tensor(y,x,weights,WIDTH*4,0,count,width,1e-5f));
            assert(ds4_gpu_end_commands());elapsed[variant]+=bench_seconds()-start;
            assert(ds4_gpu_tensor_read(y,0,actual,size));
            if(memcmp(reference,actual,size)) {fprintf(stderr,"input norm drift width=%u rows=%u variant=%u\n",width,count,variant);rc=1;goto done;}
            /* The actual graph uses in-place normalization. */
            assert(ds4_gpu_tensor_write(y,0,input,size));
            assert(ds4_gpu_muse_input_norm_tensor(y,y,weights,WIDTH*4,0,count,width,1e-5f));
            assert(ds4_gpu_tensor_read(y,0,actual,size)&&!memcmp(reference,actual,size));
        }
        double speedup=elapsed[0]/elapsed[1];
        printf("input_norm width=%u rows=%u control_us=%.3f candidate_us=%.3f speedup=%.4f exact=yes inplace=yes\n",width,count,elapsed[0]*1e6/(6*reps),elapsed[1]*1e6/(6*reps),speedup);fflush(stdout);
        if(shape<2&&speedup<minimum) rc=1;
    }
done:
    unsetenv("DS4_METAL_MUSE_NORM_REFERENCE");ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(y);ds4_gpu_cleanup();
    munmap(weights,WIDTH*4);free(input);free(reference);free(actual);return rc;
}

int main(int argc,char **argv) {
    if(argc>1&&!strcmp(argv[1],"--bench-norm")) {
        char *end=NULL;double minimum=argc==3?strtod(argv[2],&end):0;
        if(argc>3||!isfinite(minimum)||minimum<0||(argc==3&&(!argv[2][0]||*end))) return 2;
        return bench_input_norm(minimum);
    }
    if(argc>1) {

        char *minimum_end=NULL;double minimum=argc==4?strtod(argv[3],&minimum_end):0;
        if(argc<3||argc>4||strcmp(argv[1],"--bench-linear")||!argv[2][0]||strchr(argv[2],'=')||!isfinite(minimum)||minimum<0||
           (argc==4&&(!argv[3][0]||*minimum_end))) {
            fprintf(stderr,"usage: %s [--bench-linear ROLLBACK_ENV [MIN_SPEEDUP]]\n",argv[0]);return 2;
        }
        return bench_linear(argv[2],minimum);
    }
    assert(ds4_gpu_init());
    const uint64_t map_size = 65536;
    unsigned char *matrix = mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    assert(matrix != MAP_FAILED && ds4_gpu_set_model_map(matrix, map_size));
    const unsigned types[] = {0, 8, 30, 14};
    const unsigned K = 256, N = 17, T = 7;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(T * K * sizeof(float));
    ds4_gpu_tensor *y = ds4_gpu_tensor_alloc((T + 1) * N * sizeof(float));
    float inputs[T * K], actual[T * N];
    for (unsigned i = 0; i < T * K; i++) inputs[i] = ((int)(i % 13) - 6) * .0625f;
    assert(x && y && ds4_gpu_tensor_write(x, 0, inputs, sizeof(inputs)));
    float guard[N],guard_actual[N];
    for(unsigned i=0;i<N;i++) guard[i]=9000.f+i*.125f;
    float decoded[N * K];
    for (unsigned w = 0; w < sizeof(types) / sizeof(types[0]); w++) {
        const unsigned type = types[w];
        const uint64_t row_bytes = type == 0 ? K * 4 : type == 30 ? K * 2 : type == 14 ? (K / 256) * 210 : (K / 32) * 34;
        memset(matrix, 0, N * row_bytes);
        for (unsigned row = 0; row < N; row++) {
            for (unsigned i = 0; i < K; i++) {
                float value = ((int)((i + row) % 15) - 7) * .125f;
                decoded[row * K + i] = value;
                unsigned char *p = matrix + row * row_bytes;
                if (type == 0) memcpy(p + i * 4, &value, 4);
                else if (type == 30) {
                    uint32_t word; memcpy(&word, &value, 4);
                    uint16_t h = word >> 16; memcpy(p + i * 2, &h, 2);
                } else if (type == 14) {
                    unsigned j = i % 256, half = j / 128, q = j % 128 / 32, l = j % 32;
                    unsigned low = half * 64 + (q & 1) * 32 + l;
                    unsigned high = 128 + half * 32 + l;
                    unsigned code = (i * 17 + row * 11) % 64;
                    int scale = (int)((i / 16 + row) % 13) - 6;
                    p[low] |= (code & 15) << ((q / 2) * 4);
                    p[high] |= (code >> 4) << (q * 2);
                    ((int8_t *)(p + 192))[j / 16] = scale;
                    uint16_t d = 0x3000; memcpy(p + 208, &d, 2);
                    decoded[row * K + i] = .125f * scale * ((int)code - 32);
                } else {
                    p += (i / 32) * 34;
                    uint16_t d = 0x3000; memcpy(p, &d, 2); /* F16 .125 */
                    ((int8_t *)(p + 2))[i % 32] = (int8_t)((int)((i + row) % 15) - 7);
                }
            }
        }
        assert(ds4_gpu_tensor_write(y,(uint64_t)T*N*4,guard,sizeof(guard)));
        assert(ds4_gpu_muse_linear_tensor(y, matrix, map_size, 0, type, K, N, x, T));
        assert(ds4_gpu_synchronize());
        assert(ds4_gpu_tensor_read(y, 0, actual, sizeof(actual)));
        assert(ds4_gpu_tensor_read(y,(uint64_t)T*N*4,guard_actual,sizeof(guard_actual)));
        assert(!memcmp(guard,guard_actual,sizeof(guard)));
        for (unsigned t = 0; t < T; t++) for (unsigned r = 0; r < N; r++) {
            float expected = 0;
            for (unsigned i = 0; i < K; i++) expected += decoded[r * K + i] * inputs[t * K + i];
            assert(fabsf(expected - actual[t * N + r]) < 1e-4f);
        }
    }
    float norms[K], normed[T * K];
    for (unsigned i = 0; i < K; i++) norms[i] = 1.f + i * .001f;
    memcpy(matrix + 32768, norms, sizeof(norms));
    ds4_gpu_tensor *n = ds4_gpu_tensor_alloc(sizeof(normed));
    assert(n && ds4_gpu_rms_norm_weight_rows_tensor(n, x, matrix, map_size, 32768, K, T, 1e-5f));
    assert(ds4_gpu_synchronize() && ds4_gpu_tensor_read(n, 0, normed, sizeof(normed)));
    for (unsigned t = 0; t < T; t++) {
        float ss = 0;
        for (unsigned i = 0; i < K; i++) ss += inputs[t * K + i] * inputs[t * K + i];
        float inv = 1.f / sqrtf(ss / K + 1e-5f);
        for (unsigned i = 0; i < K; i++) assert(fabsf(normed[t * K + i] - inputs[t * K + i] * inv * norms[i]) < 2e-5f);
    }
    assert(ds4_gpu_muse_input_norm_tensor(n, x, matrix, map_size, 32768, T, K, 1e-8f));
    assert(ds4_gpu_synchronize() && ds4_gpu_tensor_read(n, 0, normed, sizeof(normed)));
    for (unsigned t = 0; t < T; t++) {
        double ss = 0;
        for (unsigned i = 0; i < K; i++) {
            double v = inputs[t * K + i] * norms[i];
            ss += v * v;
        }
        double inv = 1. / sqrt(ss / K + 1e-8);
        for (unsigned i = 0; i < K; i++) assert(fabs(normed[t * K + i] - inputs[t * K + i] * norms[i] * inv) < 2e-5);
    }
    /* Cover unrolled-loop tails and aliasing independently of the large
     * performance fixture, including widths smaller than one SIMD group. */
    const unsigned norm_widths[]={1,31,32,33,127,128,129,255,256};
    for(unsigned shape=0;shape<sizeof(norm_widths)/sizeof(norm_widths[0]);shape++) {
        unsigned width=norm_widths[shape];float expected[T*K],reference[T*K];
        for(unsigned t=0;t<T;t++) {
            double ss=0;
            for(unsigned i=0;i<width;i++) {double v=inputs[t*width+i]*norms[i];ss+=v*v;}
            double inv=1./sqrt(ss/width+1e-8);
            for(unsigned i=0;i<width;i++) expected[t*width+i]=inputs[t*width+i]*norms[i]*inv;
        }
        assert(!setenv("DS4_METAL_MUSE_NORM_REFERENCE","1",1));
        assert(ds4_gpu_muse_input_norm_tensor(n,x,matrix,map_size,32768,T,width,1e-8f));
        assert(ds4_gpu_tensor_read(n,0,reference,T*width*4));
        assert(!unsetenv("DS4_METAL_MUSE_NORM_REFERENCE"));
        assert(ds4_gpu_muse_input_norm_tensor(n,x,matrix,map_size,32768,T,width,1e-8f));
        assert(ds4_gpu_tensor_read(n,0,normed,T*width*4)&&!memcmp(reference,normed,T*width*4));
        for(unsigned i=0;i<T*width;i++) assert(fabsf(normed[i]-expected[i])<2e-5f);
        assert(ds4_gpu_tensor_write(n,0,inputs,T*width*4));
        assert(ds4_gpu_muse_input_norm_tensor(n,n,matrix,map_size,32768,T,width,1e-8f));
        assert(ds4_gpu_tensor_read(n,0,normed,T*width*4)&&!memcmp(reference,normed,T*width*4));
    }
    float beta[K];
    for (unsigned i = 0; i < K; i++) beta[i] = .75f-i*.002f;
    memcpy(matrix + 36864, beta, sizeof(beta));
    assert(ds4_gpu_muse_residual_tensor(n, x, n, matrix, map_size,32768,36864,T,K));
    assert(ds4_gpu_synchronize());
    float residual[T*K];
    assert(ds4_gpu_tensor_read(n,0,residual,sizeof(residual)));
    for (unsigned i = 0; i < T*K; i++)
        assert(fabsf(residual[i] - (inputs[i]*norms[i%K]+normed[i]*beta[i%K]))<2e-5f);
    assert(ds4_gpu_muse_gate_tensor(n,n,x,T*K));
    assert(ds4_gpu_muse_logits_tensor(n,T*K,.1767766953f,20.f));
    assert(ds4_gpu_synchronize() && ds4_gpu_tensor_read(n,0,normed,sizeof(normed)));
    for (unsigned i = 0; i < T*K; i++) {
        double z = residual[i]/(1.+exp(-inputs[i]));
        double expected = 20.*tanh(z*.1767766953/20.);
        assert(fabs(normed[i]-expected)<2e-5);
    }
    assert(!ds4_gpu_muse_logits_tensor(n,T*K+1,1.f,20.f));
    ds4_gpu_tensor_free(n);
    ds4_gpu_tensor_free(y);
    ds4_gpu_tensor_free(x);
    ds4_gpu_cleanup();
    munmap(matrix, map_size);
    puts("Muse native Metal kernels: PASS");
    return 0;
}
