/* Explicitly invoked full-checkpoint validation; never part of make test. */
#define _POSIX_C_SOURCE 200809L
#include "ds4.h"
#include "ds4_gpu.h"
#include "muse_native_check.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/time.h>
#include <unistd.h>

static FILE *artifact(const char *directory,const char *name,const char *suffix) {
    char path[4096];
    int n=snprintf(path,sizeof(path),"%s/%s.%s",directory,name,suffix);
    if(n<0 || (size_t)n>=sizeof(path)) return NULL;
    int fd=open(path,O_RDWR|O_CREAT|O_EXCL,0600);
    if(fd<0) {perror(path);return NULL;}
    FILE *fp=fdopen(fd,"w+b"); if(!fp) close(fd); return fp;
}
static int integer(const char *value,int minimum,int maximum) {
    char *end;errno=0;long n=strtol(value,&end,10);
    return errno || !value[0] || *end || n<minimum || n>maximum ? -1 : (int)n;
}
static double seconds(void) {
    struct timeval tv;gettimeofday(&tv,NULL);return tv.tv_sec+tv.tv_usec*1e-6;
}
/* Controlled prompt experiments use the same native session and exact cached
 * prefix. This is an explicitly invoked test mode, not an inference frontend. */
static unsigned char *probe_read(const char *path,size_t limit,size_t *length) {
    FILE *fp=fopen(path,"rb");if(!fp) return NULL;
    unsigned char *data=NULL;
    if(fseek(fp,0,SEEK_END)) goto done;
    long n=ftell(fp);if(n<=0 || (uint64_t)n>limit || fseek(fp,0,SEEK_SET)) goto done;
    data=malloc((size_t)n+1);if(!data) goto done;
    if(fread(data,1,(size_t)n,fp)!=(size_t)n) {free(data);data=NULL;goto done;}
    data[n]=0;*length=(size_t)n;
done:
    fclose(fp);return data;
}
static void probe_progress(void *ud,const char *event,int current,int total) {
    (void)ud;
    if(!strcmp(event,"prefill_chunk")) {printf("prefill %d/%d\n",current,total);fflush(stdout);}
}
static int prompt_probes(int argc,char **argv) {
    if(argc<8) {fprintf(stderr,"Usage: --prompt-probes MODEL OUTPUT_DIR CTX STEPS PREFIX.i32 SUFFIX.txt...\n");return 2;}
    int ctx=integer(argv[4],32,1048576),steps=integer(argv[5],1,2048),rc=1;
    if(ctx<0 || steps<0) return 2;
    size_t bytes=0;unsigned char *input=probe_read(argv[6],(size_t)ctx*4,&bytes);
    if(!input || bytes%4) {free(input);fprintf(stderr,"Invalid prefix IDs\n");return 2;}
    ds4_tokens prefix={0};
    for(size_t i=0;i<bytes/4;i++) {int32_t id;memcpy(&id,input+i*4,4);ds4_tokens_push(&prefix,id);}
    free(input);
    ds4_engine_options opt={.model_path=argv[2],.backend=DS4_BACKEND_METAL,
        .context_size=ctx,.prefill_chunk=512,.power_percent=100,.ssd_streaming=true,
        .placement_ctx_hint=ctx,.placement_session_count_hint=2};
    ds4_engine *engine=NULL;ds4_session *session=NULL;ds4_session_snapshot snapshot={0};
    FILE *report=artifact(argv[3],"prompt-probes","summary.json");
    float *initial=NULL,*restored=NULL;char err[512]={0};int completed=0;
    if(!report) goto done;
    fprintf(report,"{\"prefix_tokens\":%d,\"cases\":[",prefix.len);
    if(ds4_engine_open(&engine,&opt) || !ds4_engine_is_muse(engine) ||
       ds4_session_create(&session,engine,ctx)) goto done;
    int vocab=ds4_engine_vocab_size(engine);
    for(int i=0;i<prefix.len;i++) if(prefix.v[i]<0 || prefix.v[i]>=vocab) goto done;
    initial=malloc((size_t)vocab*sizeof(float));restored=malloc((size_t)vocab*sizeof(float));
    if(!initial || !restored) goto done;
    ds4_session_set_progress(session,probe_progress,NULL);
    if(ds4_session_sync(session,&prefix,err,sizeof(err)) ||
       ds4_session_copy_logits(session,initial,vocab)!=vocab ||
       ds4_session_save_snapshot(session,&snapshot,err,sizeof(err))) goto done;
    puts("shared prefix snapshot ready");fflush(stdout);
    for(int i=7;i<argc;i++) {
        size_t n=0;unsigned char *text=probe_read(argv[i],262144,&n);
        ds4_tokens prompt={0};ds4_chat_decode_state decode;
        FILE *raw=NULL,*public=NULL;char name[40];int generated=0;bool stopped=false;float error=0;
        if(!text || memchr(text,0,n)) {free(text);goto done;}
        for(int j=0;j<prefix.len;j++) ds4_tokens_push(&prompt,prefix.v[j]);
        ds4_tokenize_rendered_chat(engine,(char *)text,&prompt);free(text);
        if(prompt.len>ctx-steps || ds4_chat_decode_prefix(engine,&decode,&prompt)) {
            ds4_tokens_free(&prompt);goto done;
        }
        if(ds4_session_load_snapshot(session,&snapshot,err,sizeof(err)) ||
           ds4_session_copy_logits(session,restored,vocab)!=vocab ||
           muse_native_difference(initial,restored,vocab,0,&error)) {ds4_tokens_free(&prompt);goto done;}
        snprintf(name,sizeof(name),"probe-%d",i-7);
        raw=artifact(argv[3],name,"raw.txt");public=artifact(argv[3],name,"public.txt");
        if(!raw || !public || ds4_session_sync(session,&prompt,err,sizeof(err))) goto case_done;
        for(int step=0;step<steps;step++) {
            int token=ds4_session_argmax(session);size_t length=0;
            if(token<0 || token>=vocab) goto case_done;
            char *piece=ds4_token_text(engine,token,&length);
            if(!piece || fwrite(piece,1,length,raw)!=length) {free(piece);goto case_done;}
            free(piece);piece=ds4_chat_decode_token(engine,&decode,token,&length);
            if(!piece) goto case_done;
            if(decode.final_answer && fwrite(piece,1,length,public)!=length) {free(piece);goto case_done;}
            free(piece);
            if(ds4_token_is_stop(engine,token)) {stopped=true;break;}
            if(ds4_session_eval(session,token,err,sizeof(err))) goto case_done;
            generated++;
        }
        fprintf(report,"%s{\"index\":%d,\"prompt_tokens\":%d,\"generated\":%d,\"stopped\":%s,\"prefix_restore_max_error\":%.9g}",
            completed?",":"",i-7,prompt.len,generated,stopped?"true":"false",error);fflush(report);
        completed++;printf("probe %d complete: %d tokens stopped=%d\n",i-7,generated,stopped);fflush(stdout);
case_done:
        if(raw) fclose(raw);if(public) fclose(public);ds4_tokens_free(&prompt);
        if(completed!=i-6) goto done;
    }
    rc=0;
done:
    if(err[0]) fprintf(stderr,"%s\n",err);
    if(report) {fprintf(report,"],\"status\":\"%s\"}\n",rc?"ERROR":"COMPLETE");fclose(report);}
    free(initial);free(restored);ds4_session_snapshot_free(&snapshot);
    ds4_session_free(session);ds4_engine_close(engine);ds4_tokens_free(&prefix);return rc;
}
int main(int argc,char **argv) {
    if(argc>1 && !strcmp(argv[1],"--prompt-probes")) return prompt_probes(argc,argv);
    if(argc==2 && !strcmp(argv[1],"--help")) {
        puts("Usage: test_muse_native_live MODEL OUTPUT_DIR PREFILL CACHE_EXPERTS CTX STEPS [TEACHER_DIR]\n"
             "       test_muse_native_live --prompt-probes MODEL OUTPUT_DIR CTX STEPS PREFIX.i32 SUFFIX.txt...\n"
             "Runs three native cases, checks restored KV and rewind, and writes full-vocabulary FP32 logits.\n"
             "Prompt probes restore one exact prefix snapshot before each rendered suffix and greedily generate bounded text.\n"
             "OUTPUT_DIR must exist with unused artifact names. All runs use Metal SSD streaming.\n"
             "This opens the full checkpoint. Do not run while another large model is resident.");return 0;
    }
    if(argc!=7 && argc!=8) {fprintf(stderr,"Use --help for explicit full-checkpoint invocation.\n");return 2;}
    int chunk=integer(argv[3],1,8192),cache=integer(argv[4],8,15872);
    int ctx=integer(argv[5],32,1048576),steps=integer(argv[6],1,16);
    if(chunk<0 || cache<0 || ctx<0 || steps<0) {fprintf(stderr,"Invalid dimensions.\n");return 2;}
    ds4_engine_options opt={.model_path=argv[1],.backend=DS4_BACKEND_METAL,
        .context_size=ctx,.prefill_chunk=(uint32_t)chunk,.power_percent=100,
        .ssd_streaming=true,.ssd_streaming_cache_experts=(uint32_t)cache,
        .placement_ctx_hint=ctx,.placement_session_count_hint=2};
    ds4_engine *e=NULL;
    if(ds4_engine_open(&e,&opt)) return 1;
    if(!ds4_engine_is_muse(e)) {fprintf(stderr,"A Muse checkpoint is required.\n");ds4_engine_close(e);return 2;}
    const char *names[]={"raw","chat-none","chat-high"};
    const char *texts[]={"Once upon a time","What is 2+2? Answer briefly.",
                         "What is 17 times 19? Check your arithmetic."};
    int rc=1;
    FILE *report=artifact(argv[2],"summary","json");
    if(!report) goto done;
    fprintf(report,"{\"prefill\":%d,\"cache_experts_requested\":%d,\"context\":%d,\"cases\":[",chunk,cache,ctx);
    for(int c=0;c<3;c++) {
        char err[512]={0};ds4_tokens prompt={0};
        if(!c) ds4_tokenize_text(e,texts[c],&prompt);
        else {
            /* Fix the date so comparison runs reproduce exactly the same prefix. */
            ds4_think_mode mode=c==1?DS4_THINK_NONE:DS4_THINK_HIGH;
            char *system=ds4_muse_system_prompt(NULL,NULL,NULL,mode,"2026-10-20");
            ds4_chat_begin(e,&prompt);ds4_chat_append_message(e,&prompt,"system",system);free(system);
            ds4_chat_append_message(e,&prompt,"user",texts[c]);
            ds4_chat_append_assistant_prefix(e,&prompt,mode);
        }
        int teacher[16];const int *forced=NULL;
        if(argc==8) {
            char path[4096];int n=snprintf(path,sizeof(path),"%s/%s.ids.i32",argv[7],names[c]);
            FILE *fp=n>0 && (size_t)n<sizeof(path)?fopen(path,"rb"):NULL;
            if(!fp || fread(teacher,sizeof(int32_t),(size_t)steps,fp)!=(size_t)steps || fgetc(fp)!=EOF) {
                fprintf(stderr,"Invalid teacher IDs for %s\n",names[c]);if(fp)fclose(fp);
                ds4_tokens_free(&prompt);goto close_report;
            }
            fclose(fp);forced=teacher;
        }
        FILE *vectors=artifact(argv[2],names[c],"logits.f32");
        FILE *ids=artifact(argv[2],names[c],"ids.i32");
        FILE *prefix=artifact(argv[2],names[c],"prompt.i32");
        FILE *text=artifact(argv[2],names[c],"raw.txt");
        int ok=vectors && ids && prefix && text;
        muse_native_result result={0};double start=seconds();
        if(ok) ok=fwrite(prompt.v,sizeof(int32_t),(size_t)prompt.len,prefix)==(size_t)prompt.len;
        if(ok) ok=muse_native_check(e,ctx,&prompt,forced,steps,vectors,ids,&result,err,sizeof(err))==0;
        double elapsed=seconds()-start;
        if(ok) {
            rewind(ids);
            for(int s=0;s<steps;s++) {
                int32_t id;size_t len=0;
                if(fread(&id,sizeof(id),1,ids)!=1) {ok=0;break;}
                char *piece=ds4_token_text(e,id,&len);
                if(!piece || fwrite(piece,1,len,text)!=len) ok=0;
                free(piece);if(!ok)break;
            }
        }
        if(vectors && fclose(vectors))ok=0;
        if(ids && fclose(ids))ok=0;
        if(prefix && fclose(prefix))ok=0;
        if(text && fclose(text))ok=0;
        fprintf(stderr,"Muse %s paired validation: %.3f seconds, %s %s\n",names[c],elapsed,ok?"PASS":"FAIL",err);
        if(!ok) {ds4_tokens_free(&prompt);goto close_report;}
        fprintf(report,"%s{\"name\":\"%s\",\"vocab\":%d,\"prompt_tokens\":%d,\"steps\":%d,"
                "\"restore_max_error\":%.9g,\"decode_max_error\":%.9g,\"rewind_max_error\":%.9g,"
                "\"paired_validation_seconds\":%.6f}",c?",":"",names[c],result.vocab,prompt.len,result.steps,
                result.restore_max_error,result.decode_max_error,result.rewind_max_error,elapsed);
        ds4_tokens_free(&prompt);ds4_gpu_print_memory_report("Muse live validation");
    }
    fprintf(report,"]}\n");rc=0;
close_report:
    if(ferror(report))rc=1;
    if(fclose(report))rc=1;
done:
    ds4_engine_close(e);return rc;
}
