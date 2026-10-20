/* Test-only public-session checks shared by the tiny fixture and live driver. */
#ifndef MUSE_NATIVE_CHECK_H
#define MUSE_NATIVE_CHECK_H
#include "ds4.h"
#include "ds4_gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int vocab, steps;
    float restore_max_error, decode_max_error, rewind_max_error;
} muse_native_result;

static int muse_native_difference(const float *a, const float *b, int n,
                                  float tolerance, float *maximum) {
    for (int i=0;i<n;i++) {
        if (!isfinite(a[i]) || !isfinite(b[i])) return -1;
        float d=fabsf(a[i]-b[i]);
        if (d>*maximum) *maximum=d;
    }
    return *maximum<=tolerance ? 0 : -1;
}

static int muse_native_check(ds4_engine *e, int ctx, const ds4_tokens *prompt,
                             const int *teacher, int steps, FILE *vectors, FILE *ids,
                             muse_native_result *result, char *err, size_t errlen) {
    ds4_session *a=NULL,*b=NULL;
    ds4_session_snapshot snap={0};
    int rc=-1, vocab=ds4_engine_vocab_size(e);
    float *initial=NULL,*current=NULL,*restored=NULL;
    const char *failure="native session validation failed";
    if (err && errlen) err[0]=0;
    memset(result,0,sizeof(*result)); result->vocab=vocab;
    if (vocab<=0 || prompt->len<=0 || steps<1 || steps>16 || ctx<prompt->len ||
        steps>ctx-prompt->len) { failure="invalid validation dimensions"; goto done; }
    initial=malloc((size_t)vocab*sizeof(float));
    current=malloc((size_t)vocab*sizeof(float));
    restored=malloc((size_t)vocab*sizeof(float));
    if (!initial || !current || !restored) { failure="validation allocation failed"; goto done; }
    if (ds4_session_create(&a,e,ctx) || ds4_session_create(&b,e,ctx)) goto done;
    if (ds4_session_sync(a,prompt,err,errlen)) goto done;
    if (ds4_session_copy_logits(a,initial,vocab)!=vocab) goto done;
    if (muse_native_difference(initial,initial,vocab,0,&result->restore_max_error)) goto done;
    if (ds4_session_save_snapshot(a,&snap,err,errlen) ||
        ds4_session_load_snapshot(b,&snap,err,errlen)) goto done;
    if (ds4_session_copy_logits(b,restored,vocab)!=vocab ||
        muse_native_difference(initial,restored,vocab,0,&result->restore_max_error)) {
        failure="snapshot logits differ"; goto done;
    }
    if (vectors && fwrite(initial,sizeof(float),(size_t)vocab,vectors)!=(size_t)vocab) goto io_error;
    for (int step=0;step<steps;step++) {
        int token=teacher ? teacher[step] : ds4_session_argmax(a);
        if (token<0 || token>=vocab) { failure="invalid teacher or greedy token"; goto done; }
        if (ds4_session_eval(a,token,err,errlen) || ds4_session_eval(b,token,err,errlen)) goto done;
        if (ds4_session_copy_logits(a,current,vocab)!=vocab ||
            ds4_session_copy_logits(b,restored,vocab)!=vocab ||
            muse_native_difference(current,restored,vocab,3e-5f,&result->decode_max_error)) {
            failure="restored-session decode logits differ"; goto done;
        }
        int32_t id=(int32_t)token;
        if ((vectors && fwrite(current,sizeof(float),(size_t)vocab,vectors)!=(size_t)vocab) ||
            (ids && fwrite(&id,sizeof(id),1,ids)!=1)) goto io_error;
        result->steps++;
    }
    /* This may rebuild once generated tokens have displaced the SWA prefix. */
    if (ds4_session_sync(a,prompt,err,errlen) ||
        ds4_session_copy_logits(a,current,vocab)!=vocab) goto done;
    if (muse_native_difference(initial,current,vocab,2e-3f,&result->rewind_max_error)) {
        failure="rewound-prefix logits differ"; goto done;
    }
    if (vectors) ds4_gpu_print_memory_report("Muse paired sessions resident");
    rc=0; goto done;
io_error:
    failure="cannot write validation artifacts";
done:
    if (rc && err && errlen && !err[0]) snprintf(err,errlen,"%s",failure);
    ds4_session_snapshot_free(&snap); ds4_session_free(a); ds4_session_free(b);
    free(initial); free(current); free(restored);
    return rc;
}
#endif
