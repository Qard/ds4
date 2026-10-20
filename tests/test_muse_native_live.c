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
int main(int argc,char **argv) {
    if(argc==2 && !strcmp(argv[1],"--help")) {
        puts("Usage: test_muse_native_live MODEL OUTPUT_DIR PREFILL CACHE_EXPERTS CTX STEPS [TEACHER_DIR]\n"
             "Runs three native cases, checks restored KV and rewind, and writes full-vocabulary FP32 logits.\n"
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
