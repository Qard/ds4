/* Packed IQ2_XXS parity at the complete expert boundary, including post-expert
 * gain-before-RMS and cache eviction. All weights are synthetic (< 1 MiB). */
#include "../ds4.c"
#include <assert.h>

typedef struct {uint16_t d;int8_t qs[32];} muse_test_q8_0;

static void test_q8_linears(void) {
    enum {KMAX=8192,NMAX=64,TMAX=32,PAD=16};
    size_t bytes=(size_t)NMAX*(KMAX/32)*34;
    unsigned char *map=mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
    assert(map!=MAP_FAILED);uint32_t seed=817;
    for(size_t i=0;i<bytes/34;i++) {
        muse_test_q8_0 *b=(muse_test_q8_0 *)(map+i*34);b->d=(uint16_t)(0x2155+i%16);
        for(unsigned j=0;j<32;j++) {seed=seed*1664525u+1013904223u;b->qs[j]=(int8_t)(seed>>24);}
    }
    float *input=xmalloc((size_t)TMAX*KMAX*4),*actual=xmalloc((TMAX*NMAX+2*PAD)*4);
    for(unsigned i=0;i<TMAX*KMAX;i++) input[i]=sinf(i*.17f)*.135f+cosf(i*.07f)*.031f;
    assert(ds4_gpu_init() && ds4_gpu_set_model_map(map,bytes));
    ds4_gpu_tensor *xb=ds4_gpu_tensor_alloc((TMAX*KMAX+2*PAD)*4),*yb=ds4_gpu_tensor_alloc((TMAX*NMAX+2*PAD)*4);
    ds4_gpu_tensor *x=ds4_gpu_tensor_view(xb,PAD*4,TMAX*KMAX*4),*y=ds4_gpu_tensor_view(yb,PAD*4,TMAX*NMAX*4);
    assert(xb&&yb&&x&&y && ds4_gpu_tensor_write(x,0,input,TMAX*KMAX*4));
    const unsigned rows[]={1,2,3,5,8,9,17,32};double worst=0;
    for(unsigned shape=0;shape<2;shape++) for(unsigned ti=0;ti<sizeof(rows)/sizeof(*rows);ti++) {
        unsigned k=shape?8192:128,n=shape?64:17,t=rows[ti],stride=k/32*34;
        for(unsigned i=0;i<TMAX*NMAX+2*PAD;i++) actual[i]=123456.f;
        assert(ds4_gpu_tensor_write(yb,0,actual,(TMAX*NMAX+2*PAD)*4));
        assert(ds4_gpu_muse_linear_tensor(y,map,bytes,0,8,k,n,x,t) && ds4_gpu_synchronize());
        assert(ds4_gpu_tensor_read(yb,0,actual,(TMAX*NMAX+2*PAD)*4));
        for(unsigned i=0;i<PAD;i++) assert(actual[i]==123456.f);
        for(unsigned i=PAD+t*n;i<TMAX*NMAX+2*PAD;i++) assert(actual[i]==123456.f);
        for(unsigned row=0;row<t;row++) for(unsigned out=0;out<n;out++) {
            double expected=0;
            for(unsigned col=0;col<k;col++) {
                const muse_test_q8_0 *b=(muse_test_q8_0 *)(map+out*stride+(col/32)*34);
                expected+=(double)f16_to_f32(b->d)*b->qs[col%32]*input[row*k+col];
            }
            double error=fabs(actual[PAD+row*n+out]-expected);worst=MAX(worst,error);
            assert(error<6e-5+2e-6*fabs(expected));
        }
    }
    printf("Muse Q8_0 FP32 linears, irregular rows and offset views: max error %.8g PASS\n",worst);
    ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(y);ds4_gpu_tensor_free(xb);ds4_gpu_tensor_free(yb);
    ds4_gpu_cleanup();munmap(map,bytes);free(input);free(actual);
}

int main(void) {
    enum {D=256,F=256,E=4,S=2,T=5,ROW=66,BANK=E*F*ROW,SIZE=3*BANK+D*4};
    FILE *file=tmpfile();assert(file && !ftruncate(fileno(file),SIZE));
    unsigned char *map=mmap(NULL,SIZE,PROT_READ|PROT_WRITE,MAP_SHARED,fileno(file),0);
    assert(map!=MAP_FAILED);uint32_t seed=1773;
    for(unsigned i=0;i<3*E*F;i++) {
        block_iq2_xxs *block=(block_iq2_xxs *)(map+i*ROW);block->d=0x1c00;
        for(unsigned j=0;j<32;j++) {seed=seed*1664525u+1013904223u;block->qs[j]=(uint16_t)(seed>>16);}
    }
    float *gain=(float *)(map+3*BANK);
    for(unsigned i=0;i<D;i++) gain[i]=.7f+i/512.f;
    assert(!msync(map,SIZE,MS_SYNC));
    float input[T*D],prob[T*S]={.2f,.8f,.3f,.7f,.4f,.6f,.25f,.75f,.9f,.1f};
    int32_t ids[T*S]={0,1,1,2,0,3,3,0,2,1};
    for(unsigned i=0;i<T*D;i++) input[i]=((int)(i%23)-11)*.03125f;
    double expected[T*D];memset(expected,0,sizeof(expected));
    for(unsigned t=0;t<T;t++) for(unsigned s=0;s<S;s++) {
        unsigned expert=ids[t*S+s];float hidden[F],raw[D];double ss=0;
        for(unsigned r=0;r<F;r++) {
            float g=ds4_vec_dot_iq2_xxs_f32(D,(block_iq2_xxs *)(map+(expert*F+r)*ROW),input+t*D);
            float u=ds4_vec_dot_iq2_xxs_f32(D,(block_iq2_xxs *)(map+BANK+(expert*F+r)*ROW),input+t*D);
            hidden[r]=g/(1.f+expf(-g))*u;
        }
        for(unsigned d=0;d<D;d++) {
            raw[d]=ds4_vec_dot_iq2_xxs_f32(F,(block_iq2_xxs *)(map+2*BANK+(expert*D+d)*ROW),hidden)*gain[d];
            ss+=(double)raw[d]*raw[d];
        }
        double inv=1./sqrt(ss/D+1e-8);
        for(unsigned d=0;d<D;d++) expected[t*D+d]+=prob[t*S+s]*raw[d]*inv;
    }
    assert(ds4_gpu_init() && ds4_gpu_set_model_map(map,SIZE) && ds4_gpu_set_model_fd(fileno(file)));
    ds4_gpu_tensor *x=ds4_gpu_tensor_alloc(sizeof(input)),*selected=ds4_gpu_tensor_alloc(sizeof(ids));
    ds4_gpu_tensor *mid=ds4_gpu_tensor_alloc(T*S*F*4),*part=ds4_gpu_tensor_alloc(T*S*D*4);
    ds4_gpu_tensor *weights=ds4_gpu_tensor_alloc(sizeof(prob)),*out=ds4_gpu_tensor_alloc(T*D*4);
    assert(x&&selected&&mid&&part&&weights&&out);
    assert(ds4_gpu_tensor_write(x,0,input,sizeof(input)) && ds4_gpu_tensor_write(selected,0,ids,sizeof(ids)));
    assert(ds4_gpu_tensor_write(weights,0,prob,sizeof(prob)));
    for(unsigned streamed=0;streamed<2;streamed++) {
        ds4_gpu_set_ssd_streaming(streamed);ds4_gpu_set_streaming_expert_cache_budget(S);
        ds4_gpu_set_streaming_expert_cache_expert_bytes(3*F*ROW);
        assert(ds4_gpu_begin_commands());
        assert(ds4_gpu_muse_experts_tensor(mid,part,x,selected,map,SIZE,0,BANK,2*BANK,16,16,T,E,S,D,F,0));
        assert(ds4_gpu_muse_input_norm_tensor(part,part,map,SIZE,3*BANK,T*S,D,1e-8f));
        assert(ds4_gpu_qwen4_moe_reduce_tensor(out,part,weights,NULL,NULL,NULL,NULL,T,S,S,D,0));
        assert(ds4_gpu_end_commands());float actual[T*D];
        assert(ds4_gpu_tensor_read(out,0,actual,sizeof(actual)));double worst=0;
        for(unsigned i=0;i<T*D;i++) {double error=fabs(actual[i]-expected[i]);worst=MAX(worst,error);assert(error<3e-5);}
        if(streamed) assert(ds4_gpu_stream_expert_cache_current_count()<=S);
        printf("Muse IQ2_XXS %s expert parity: max error %.8g PASS\n",streamed?"SSD":"resident",worst);
    }
    ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(selected);ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(part);ds4_gpu_tensor_free(weights);ds4_gpu_tensor_free(out);
    ds4_gpu_cleanup();munmap(map,SIZE);fclose(file);test_q8_linears();return 0;
}
