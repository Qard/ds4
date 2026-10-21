/* Packed IQ2_XXS parity at the complete expert boundary, including post-expert
 * gain-before-RMS and cache eviction. Normal fixtures use <1 MiB of synthetic
 * weights; the optional expert benchmark uses eight 4096-wide experts. */
#include "../ds4.c"
#include <assert.h>

typedef struct {uint16_t d;int8_t qs[32];} muse_test_q8_0;

/* The logical cache permits a whole 512-token batch, but only eight synthetic
 * expert triplets are instantiated. Warm both variants; compare every raw
 * intermediate bit and independently check selected rows with the CPU decoder. */
static int bench_experts(double minimum_speedup,const char *rollback,bool decode_only) {
    enum {D=4096,F=4096,E=8,S=8,TMAX=512,ROW=D/256*66,BANK=E*F*ROW,SIZE=3*BANK};
    FILE *file=tmpfile();assert(file&&!ftruncate(fileno(file),SIZE));
    unsigned char *map=mmap(NULL,SIZE,PROT_READ|PROT_WRITE,MAP_SHARED,fileno(file),0);
    assert(map!=MAP_FAILED);uint32_t seed=981;
    for(unsigned i=0;i<SIZE;i++) {seed=seed*1664525u+1013904223u;map[i]=seed>>24;}
    for(unsigned i=0;i<SIZE/66;i++) {uint16_t d=0x1800+i%16;memcpy(map+i*66,&d,2);}
    assert(!msync(map,SIZE,MS_SYNC));
    size_t max_values=(size_t)TMAX*S*D;
    float *input=xmalloc((size_t)TMAX*D*4),*reference_mid=xmalloc(max_values*4),*reference_part=xmalloc(max_values*4),*actual=xmalloc(max_values*4);
    int32_t *ids=xmalloc((size_t)TMAX*S*4);
    for(unsigned i=0;i<TMAX*D;i++) input[i]=sinf(i*.031f)*.17f;
    for(unsigned t=0;t<TMAX;t++) for(unsigned s=0;s<S;s++) ids[t*S+s]=(t*3+s)%E;
    assert(ds4_gpu_init());ds4_gpu_set_ssd_streaming(true);
    assert(ds4_gpu_set_model_map(map,SIZE)&&ds4_gpu_set_model_fd(fileno(file)));
    ds4_gpu_set_streaming_expert_cache_budget(TMAX*S);
    ds4_gpu_set_streaming_expert_cache_expert_bytes(3ull*F*ROW);
    ds4_gpu_tensor *x=ds4_gpu_tensor_alloc((size_t)TMAX*D*4),*selected=ds4_gpu_tensor_alloc((size_t)TMAX*S*4);
    ds4_gpu_tensor *mid=ds4_gpu_tensor_alloc(max_values*4),*part=ds4_gpu_tensor_alloc(max_values*4);
    ds4_gpu_tensor *lists=ds4_gpu_tensor_alloc((uint64_t)E*TMAX*S*4),*counts=ds4_gpu_tensor_alloc(E*4);
    assert(lists&&counts);
    assert(x&&selected&&mid&&part&&ds4_gpu_tensor_write(x,0,input,(size_t)TMAX*D*4)&&ds4_gpu_tensor_write(selected,0,ids,(size_t)TMAX*S*4));
    assert(!setenv(rollback,"1",1));
    for(unsigned warm=0;warm<3;warm++) {
        assert(ds4_gpu_begin_commands());
        assert(ds4_gpu_muse_experts_tensor(mid,part,x,selected,map,SIZE,0,BANK,2*BANK,16,16,TMAX,E,S,D,F,0,lists,counts,0, false));
        assert(ds4_gpu_end_commands());
    }
    const unsigned shapes[]={1,2,3,5,16,512};int rc=0;
    for(unsigned shape=0;shape<sizeof(shapes)/sizeof(shapes[0]);shape++) {
        unsigned rows=shapes[shape];if(decode_only&&rows!=1) continue;
        unsigned reps=rows==512?1:8;size_t bytes=(size_t)rows*S*D*4;double elapsed[2]={0};
        assert(!setenv(rollback,"1",1)&&ds4_gpu_begin_commands());
        assert(ds4_gpu_muse_experts_tensor(mid,part,x,selected,map,SIZE,0,BANK,2*BANK,16,16,rows,E,S,D,F,0,lists,counts,0, false));
        assert(ds4_gpu_end_commands()&&ds4_gpu_tensor_read(mid,0,reference_mid,bytes)&&ds4_gpu_tensor_read(part,0,reference_part,bytes));
        for(unsigned s=0;s<S;s++) for(unsigned edge=0;edge<2;edge++) {
            unsigned row=edge?D-1:0,expert=ids[s];
            const block_iq2_xxs *gate=(const block_iq2_xxs *)(map+(expert*F+row)*ROW);
            const block_iq2_xxs *up=(const block_iq2_xxs *)(map+BANK+(expert*F+row)*ROW);
            const block_iq2_xxs *down=(const block_iq2_xxs *)(map+2*BANK+(expert*D+row)*ROW);
            float g=ds4_vec_dot_iq2_xxs_f32(D,gate,input),u=ds4_vec_dot_iq2_xxs_f32(D,up,input);
            float expected=g/(1.f+expf(-g))*u;
            assert(fabsf(reference_mid[s*F+row]-expected)<1e-4f+fabsf(expected)*1e-5f);
            expected=ds4_vec_dot_iq2_xxs_f32(F,down,reference_mid+s*F);
            assert(fabsf(reference_part[s*D+row]-expected)<1e-4f+fabsf(expected)*1e-5f);
        }
        for(unsigned round=0;round<6;round++) for(unsigned order=0;order<2;order++) {
            unsigned variant=(round+order)&1;
            assert(!(variant?unsetenv(rollback):setenv(rollback,"1",1)));
            assert(ds4_gpu_begin_commands()&&ds4_gpu_muse_experts_tensor(mid,part,x,selected,map,SIZE,0,BANK,2*BANK,16,16,rows,E,S,D,F,0,lists,counts,0, false)&&ds4_gpu_end_commands());
            double start=now_sec();assert(ds4_gpu_begin_commands());
            for(unsigned rep=0;rep<reps;rep++) assert(ds4_gpu_muse_experts_tensor(mid,part,x,selected,map,SIZE,0,BANK,2*BANK,16,16,rows,E,S,D,F,0,lists,counts,0, false));
            assert(ds4_gpu_end_commands());elapsed[variant]+=now_sec()-start;
            assert(ds4_gpu_tensor_read(mid,0,actual,bytes));
            if(memcmp(reference_mid,actual,bytes)) {fprintf(stderr,"IQ2 mid drift rows=%u variant=%u\n",rows,variant);rc=1;goto done;}
            assert(ds4_gpu_tensor_read(part,0,actual,bytes));
            if(memcmp(reference_part,actual,bytes)) {fprintf(stderr,"IQ2 down drift rows=%u variant=%u\n",rows,variant);rc=1;goto done;}
        }
        double speedup=elapsed[0]/elapsed[1];
        printf("IQ2_XXS D=%u F=%u experts=%u rows=%u control_ms=%.3f candidate_ms=%.3f speedup=%.4f exact=yes\n",D,F,E,rows,elapsed[0]*1000/(6*reps),elapsed[1]*1000/(6*reps),speedup);fflush(stdout);
        if(((rows>1||decode_only)&&speedup<minimum_speedup)||(rows==1&&speedup<.97)) rc=1;
    }
done:
    unsetenv(rollback);ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(selected);ds4_gpu_tensor_free(mid);ds4_gpu_tensor_free(part);ds4_gpu_tensor_free(lists);ds4_gpu_tensor_free(counts);
    ds4_gpu_cleanup();munmap(map,SIZE);fclose(file);free(input);free(ids);free(reference_mid);free(reference_part);free(actual);return rc;
}

/* Test the existing Q8 RHS specializations on fixed-projection shapes. */
static int bench_q8_decode(double minimum) {
    enum {KMAX=8192,NMAX=8192};size_t bytes=(size_t)KMAX*NMAX/32*34;
    unsigned char *map=mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
    assert(map!=MAP_FAILED);uint32_t seed=817;
    for(size_t i=0;i<bytes/34;i++) {
        muse_test_q8_0 *b=(muse_test_q8_0 *)(map+i*34);b->d=(uint16_t)(0x2155+i%16);
        for(unsigned j=0;j<32;j++){seed=seed*1664525u+1013904223u;b->qs[j]=(int8_t)(seed>>24);}
    }
    float input[KMAX],reference[NMAX],actual[NMAX];
    for(unsigned i=0;i<KMAX;i++) input[i]=sinf(i*.17f)*.135f+cosf(i*.07f)*.031f;
    assert(ds4_gpu_init()&&ds4_gpu_set_model_map(map,bytes));
    ds4_gpu_tensor *x=ds4_gpu_tensor_alloc((uint64_t)KMAX*512*4),*y=ds4_gpu_tensor_alloc((uint64_t)NMAX*512*4);
    assert(x&&y&&ds4_gpu_tensor_write(x,0,input,sizeof(input)));int rc=0;
    const unsigned shapes[][2]={{4096,8192},{8192,4096},{8192,1024},{8192,8192}};
    const char *rollback="DS4_METAL_MUSE_Q8_RHS_REFERENCE";
    for(unsigned shape=0;shape<4;shape++) {
        unsigned k=shapes[shape][0],n=shapes[shape][1];double elapsed[2]={0};
        assert(!setenv(rollback,"1",1));
        assert(ds4_gpu_muse_linear_tensor(y,map,bytes,0,8,k,n,x,1, false)&&ds4_gpu_tensor_read(y,0,reference,n*4));
        enum {ROUNDS=24,REPS=32};
        for(unsigned round=0;round<ROUNDS;round++) for(unsigned order=0;order<2;order++) {
            unsigned v=(round+order)&1;assert(!(v?unsetenv(rollback):setenv(rollback,"1",1)));
            assert(ds4_gpu_muse_linear_tensor(y,map,bytes,0,8,k,n,x,1, false)&&ds4_gpu_synchronize());
            double start=now_sec();assert(ds4_gpu_begin_commands());
            for(unsigned rep=0;rep<REPS;rep++) assert(ds4_gpu_muse_linear_tensor(y,map,bytes,0,8,k,n,x,1, false));
            assert(ds4_gpu_end_commands());elapsed[v]+=now_sec()-start;
            assert(ds4_gpu_tensor_read(y,0,actual,n*4)&&!memcmp(reference,actual,n*4));
        }
        double ratio=elapsed[0]/elapsed[1];
        printf("Q8 decode K=%u N=%u control_ms=%.6f candidate_ms=%.6f ratio=%.6f exact=yes\n",k,n,elapsed[0]*1000/(ROUNDS*REPS),elapsed[1]*1000/(ROUNDS*REPS),ratio);fflush(stdout);
        if(ratio<minimum)rc=1;
    }
    unsetenv(rollback);ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(y);ds4_gpu_cleanup();munmap(map,bytes);return rc;
}

static void test_q8_linears(void) {
    enum {KMAX=8192,NMAX=64,TMAX=65,PAD=16};
    size_t bytes=(size_t)NMAX*(KMAX/32)*34;
    unsigned char *map=mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANON,-1,0);
    assert(map!=MAP_FAILED);uint32_t seed=817;
    for(size_t i=0;i<bytes/34;i++) {
        muse_test_q8_0 *b=(muse_test_q8_0 *)(map+i*34);b->d=(uint16_t)(0x2155+i%16);
        for(unsigned j=0;j<32;j++) {seed=seed*1664525u+1013904223u;b->qs[j]=(int8_t)(seed>>24);}
    }
    float *input=xmalloc((size_t)TMAX*KMAX*4),*actual=xmalloc((TMAX*NMAX+2*PAD)*4);
    float *reference=xmalloc(TMAX*NMAX*4);
    for(unsigned i=0;i<TMAX*KMAX;i++) input[i]=sinf(i*.17f)*.135f+cosf(i*.07f)*.031f;
    assert(ds4_gpu_init() && ds4_gpu_set_model_map(map,bytes));
    ds4_gpu_tensor *xb=ds4_gpu_tensor_alloc((TMAX*KMAX+2*PAD)*4),*yb=ds4_gpu_tensor_alloc((TMAX*NMAX+2*PAD)*4);
    ds4_gpu_tensor *x=ds4_gpu_tensor_view(xb,PAD*4,TMAX*KMAX*4),*y=ds4_gpu_tensor_view(yb,PAD*4,TMAX*NMAX*4);
    assert(xb&&yb&&x&&y && ds4_gpu_tensor_write(x,0,input,TMAX*KMAX*4));
    const unsigned rows[]={1,2,3,5,8,9,17,32,33,65};double worst=0;
    for(unsigned matrix_policy=0;matrix_policy<2;matrix_policy++) for(unsigned shape=0;shape<2;shape++) for(unsigned ti=0;ti<sizeof(rows)/sizeof(*rows);ti++) {
        unsigned k=shape?8192:128,n=shape?64:17,t=rows[ti],stride=k/32*34;
        for(unsigned i=0;i<TMAX*NMAX+2*PAD;i++) actual[i]=123456.f;
        assert(ds4_gpu_tensor_write(yb,0,actual,(TMAX*NMAX+2*PAD)*4));
        assert(!setenv("DS4_METAL_MUSE_Q8_RHS_REFERENCE","1",1));
        assert(ds4_gpu_muse_linear_tensor(y,map,bytes,0,8,k,n,x,t, matrix_policy!=0) && ds4_gpu_synchronize());
        assert(ds4_gpu_tensor_read(yb,0,actual,(TMAX*NMAX+2*PAD)*4));
        memcpy(reference,actual+PAD,t*n*4);
        assert(!unsetenv("DS4_METAL_MUSE_Q8_RHS_REFERENCE"));
        assert(ds4_gpu_muse_linear_tensor(y,map,bytes,0,8,k,n,x,t, matrix_policy!=0)&&ds4_gpu_synchronize());
        assert(ds4_gpu_tensor_read(yb,0,actual,(TMAX*NMAX+2*PAD)*4)&&!memcmp(reference,actual+PAD,t*n*4));
        for(unsigned i=0;i<PAD;i++) assert(actual[i]==123456.f);
        for(unsigned i=PAD+t*n;i<TMAX*NMAX+2*PAD;i++) assert(actual[i]==123456.f);
        for(unsigned row=0;!matrix_policy && row<t;row++) {
            float single[NMAX];
            ds4_gpu_tensor *rx=ds4_gpu_tensor_view(x,(uint64_t)row*k*4,(uint64_t)k*4);
            ds4_gpu_tensor *ry=ds4_gpu_tensor_alloc(n*4);
            assert(rx&&ry && ds4_gpu_muse_linear_tensor(ry,map,bytes,0,8,k,n,rx,1, false));
            assert(ds4_gpu_tensor_read(ry,0,single,n*4));
            assert(!memcmp(single,actual+PAD+row*n,n*4));
            ds4_gpu_tensor_free(rx);ds4_gpu_tensor_free(ry);
        }
        for(unsigned row=0;row<t;row++) for(unsigned out=0;out<n;out++) {
            double expected=0;
            for(unsigned col=0;col<k;col++) {
                const muse_test_q8_0 *b=(muse_test_q8_0 *)(map+out*stride+(col/32)*34);
                expected+=(double)f16_to_f32(b->d)*b->qs[col%32]*input[row*k+col];
            }
            double error=fabs(actual[PAD+row*n+out]-expected);worst=MAX(worst,error);
            if(!(error<6e-5+2e-6*fabs(expected)))fprintf(stderr,"Q8 mismatch matrix=%u K=%u N=%u T=%u row=%u out=%u actual=%.9g expected=%.12g error=%.12g bound=%.12g\n",matrix_policy,k,n,t,row,out,actual[PAD+row*n+out],expected,error,6e-5+2e-6*fabs(expected));
            assert(error<6e-5+2e-6*fabs(expected));
        }
    }
    printf("Muse Q8_0 FP32 linears, irregular rows and offset views: max error %.8g PASS\n",worst);
    ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(y);ds4_gpu_tensor_free(xb);ds4_gpu_tensor_free(yb);
    ds4_gpu_cleanup();munmap(map,bytes);free(input);free(actual);free(reference);
}

static void test_iq2_experts(unsigned T,bool matrix_policy,unsigned route_pattern) {
    enum {D=256,F=256,E=4,S=2,ROW=66,BANK=E*F*ROW,SIZE=3*BANK+D*4};
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
    float input[T*D],prob[T*S];int32_t ids[T*S];
    const float base_prob[]={.2f,.8f,.3f,.7f,.4f,.6f,.25f,.75f,.9f,.1f};
    const int32_t base_ids[]={0,1,1,1,0,3,3,0,2,1};
    for(unsigned i=0;i<T*S;i++) {
        prob[i]=base_prob[i%10];
        ids[i]=route_pattern==2 ? 0 : !matrix_policy || route_pattern==1 ?
            base_ids[i%10] : (int32_t)((i/S+i%S)%E);
    }
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
    ds4_gpu_tensor *lists=ds4_gpu_tensor_alloc(E*T*S*4),*counts=ds4_gpu_tensor_alloc(E*4);
    assert(lists&&counts);
    ds4_gpu_tensor *weights=ds4_gpu_tensor_alloc(sizeof(prob)),*out=ds4_gpu_tensor_alloc(T*D*4);
    assert(x&&selected&&mid&&part&&weights&&out);
    assert(ds4_gpu_tensor_write(x,0,input,sizeof(input)) && ds4_gpu_tensor_write(selected,0,ids,sizeof(ids)));
    assert(ds4_gpu_tensor_write(weights,0,prob,sizeof(prob)));
    for(unsigned streamed=0;streamed<3;streamed++) {
        uint32_t budget=streamed==2?E:S;
        ds4_gpu_set_ssd_streaming(streamed!=0);ds4_gpu_set_streaming_expert_cache_budget(budget);
        ds4_gpu_set_streaming_expert_cache_expert_bytes(3*F*ROW);
        assert(ds4_gpu_begin_commands());
        if(streamed) {
            const ds4_gpu_stream_expert_table table={.model_map=map,.model_size=SIZE,
                .layer=0,.n_total_expert=E,.gate_offset=0,.up_offset=BANK,.down_offset=2*BANK,
                .gate_expert_bytes=F*ROW,.down_expert_bytes=D*ROW};
            assert(ds4_gpu_muse_prefetch_experts_tensor(&table,selected,S,0));
        }
        assert(ds4_gpu_muse_experts_tensor(mid,part,x,selected,map,SIZE,0,BANK,2*BANK,16,16,T,E,S,D,F,0,lists,counts,0, matrix_policy));
        assert(ds4_gpu_muse_input_norm_tensor(part,part,map,SIZE,3*BANK,T*S,D,1e-8f));
        assert(ds4_gpu_qwen4_moe_reduce_tensor(out,part,weights,NULL,NULL,NULL,NULL,T,S,S,D,0));
        assert(ds4_gpu_end_commands());float actual[T*D];
        assert(ds4_gpu_tensor_read(out,0,actual,sizeof(actual)));double worst=0;
        for(unsigned i=0;i<T*D;i++) {double error=fabs(actual[i]-expected[i]);worst=MAX(worst,error);if(!(error<3e-5))fprintf(stderr,"IQ2 mismatch T=%u matrix=%d route_pattern=%u streamed=%u i=%u actual=%.9g expected=%.12g error=%.12g\n",T,matrix_policy,route_pattern,streamed,i,actual[i],expected[i],error);assert(error<3e-5);}
        if(streamed) assert(ds4_gpu_stream_expert_cache_current_count()<=budget);
        printf("Muse IQ2_XXS %s expert parity: max error %.8g PASS\n",streamed?"SSD":"resident",worst);
    }
    ds4_gpu_tensor_free(x);ds4_gpu_tensor_free(selected);ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(part);ds4_gpu_tensor_free(lists);ds4_gpu_tensor_free(counts);ds4_gpu_tensor_free(weights);ds4_gpu_tensor_free(out);
    ds4_gpu_cleanup();munmap(map,SIZE);fclose(file);
}

int main(int argc,char **argv) {
    if(argc>1&&!strcmp(argv[1],"--bench-q8-decode")) {
        char *end=NULL;double minimum=argc==3?strtod(argv[2],&end):0;
        if(argc>3||!isfinite(minimum)||minimum<0||(argc==3&&(!argv[2][0]||*end)))return 2;
        return bench_q8_decode(minimum);
    }
    if(argc>1) {
        char *end=NULL;double minimum=argc>=3?strtod(argv[2],&end):0;
        bool decode_only=!strcmp(argv[1],"--bench-experts-decode");
        if(argc>4||(!decode_only&&strcmp(argv[1],"--bench-experts"))||!isfinite(minimum)||minimum<0||
           (argc>=3&&(!argv[2][0]||*end)) ||
           (argc==4&&(!argv[3][0]||strchr(argv[3],'=')))) {
            fprintf(stderr,"usage: %s [--bench-experts|--bench-experts-decode [MIN_SPEEDUP [REFERENCE_ENV]]]\n",argv[0]);return 2;
        }
        return bench_experts(minimum,argc==4?argv[3]:"DS4_METAL_MUSE_EXPERT_BATCH_REFERENCE",decode_only);
    }
    test_iq2_experts(5,false,false);
    test_iq2_experts(32,true,false);
    test_iq2_experts(33,true,false);
    test_iq2_experts(33,true,1);
    test_iq2_experts(33,true,2);
    test_q8_linears();return 0;
}
