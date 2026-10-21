/* Muse graph: reused FP32 transients, layer-private F16 KV, and the existing
 * bounded expert cache. No model tensor is expanded into an unquantized bank. */
#define MUSE_DEFAULT_PREFILL_CHUNK 512u
#define MUSE_GRAPH_SCRATCH(X) \
    X(tokens,T) X(r,T*DS4_N_EMBD) X(xn,T*DS4_N_EMBD) X(branch,T*DS4_N_EMBD) \
    X(q,T*DS4_N_HEAD*DS4_N_HEAD_DIM) X(k,T*DS4_N_HEAD_KV*DS4_N_HEAD_DIM) \
    X(v,T*DS4_N_HEAD_KV*DS4_N_VALUE_DIM) X(gate,T*DS4_N_HEAD*DS4_N_VALUE_DIM) \
    X(attn,T*DS4_N_HEAD*DS4_N_VALUE_DIM) \
    X(attn_part,ds4_gpu_muse_attention_scratch_bytes(T,DS4_N_HEAD,DS4_N_HEAD_DIM)/4) \
    X(latent,T*DS4_N_MOE_LATENT) X(router,T*DS4_N_EXPERT) X(probs,T*DS4_N_EXPERT) \
    X(selected,T*DS4_N_EXPERT_USED) X(weights,T*DS4_N_EXPERT_USED) \
    X(moe_lists,T*DS4_N_EXPERT_USED*DS4_N_EXPERT) X(moe_counts,DS4_N_EXPERT) \
    X(mid,T*DS4_N_EXPERT_USED*DS4_N_FF_EXP) X(part,T*DS4_N_EXPERT_USED*DS4_N_MOE_LATENT) \
    X(moe,T*DS4_N_MOE_LATENT) X(logits,L*DS4_N_VOCAB)

typedef struct {
    uint32_t pos, kv_highwater, swa_history_start, ctx_cap, cap_tokens, n_logit_rows;
    bool valid, cancelled;
    /* Immutable math profile for this session and its persisted KV. */
    bool flash_attention, matmul_prefill;
    ds4_session_cancel_fn cancel;
    void *cancel_ud;
#define MUSE_FIELD(field,count) ds4_gpu_tensor *field;
    MUSE_GRAPH_SCRATCH(MUSE_FIELD)
#undef MUSE_FIELD
    ds4_gpu_tensor *kc[DS4_MAX_LAYER], *vc[DS4_MAX_LAYER];
    uint32_t cache_cap[DS4_MAX_LAYER];
} ds4_muse_gpu_graph;

static uint32_t muse_graph_cache_cap(unsigned layer,uint32_t context,uint32_t cap) {
    return g_muse_swa[layer] ? MIN((uint64_t)context,(uint64_t)DS4_N_SWA+cap) : context;
}

/* Session admission counts one final logit row, matching the other native
 * graphs. Diagnostics explicitly reserve additional rows when allocating. */
static DS4_MAYBE_UNUSED uint64_t muse_graph_bytes(uint32_t context,uint32_t cap) {
    uint64_t bytes=0,T=cap,L=1;
#define MUSE_COUNT(field,count) bytes+=(uint64_t)(count)*4;
    MUSE_GRAPH_SCRATCH(MUSE_COUNT)
#undef MUSE_COUNT
    for(unsigned il=0;il<DS4_N_LAYER;il++)
        bytes+=(uint64_t)muse_graph_cache_cap(il,context,cap)*DS4_N_HEAD_KV*DS4_N_HEAD_DIM*4;
    return bytes;
}

static ds4_context_memory muse_graph_memory(uint32_t ctx, uint32_t prefill) {
    ds4_context_memory m = {0};
    uint32_t cap = MIN(ctx, prefill ? prefill : MUSE_DEFAULT_PREFILL_CHUNK);
    m.prefill_cap = cap;
    m.raw_cap = ctx;
    for (unsigned il = 0; il < DS4_N_LAYER; il++)
        m.raw_bytes += (uint64_t)muse_graph_cache_cap(il, ctx, cap) * DS4_N_HEAD_KV * DS4_N_HEAD_DIM * 4;
    m.total_bytes = muse_graph_bytes(ctx, cap);
    m.scratch_bytes = m.total_bytes - m.raw_bytes;
    return m;
}

static void muse_graph_free(ds4_muse_gpu_graph *g) {
#define MUSE_FREE(field,count) ds4_gpu_tensor_free(g->field);
    MUSE_GRAPH_SCRATCH(MUSE_FREE)
#undef MUSE_FREE
    for(unsigned il=0;il<DS4_MAX_LAYER;il++) {
        ds4_gpu_tensor_free(g->kc[il]); ds4_gpu_tensor_free(g->vc[il]);
    }
    memset(g,0,sizeof(*g));
}

static DS4_MAYBE_UNUSED bool muse_graph_alloc(ds4_muse_gpu_graph *g,uint32_t context,uint32_t cap,
        uint32_t logit_rows) {
    memset(g,0,sizeof(*g));
    if (!context || context>1048576 || !cap || cap>context || cap>8192 ||
        !logit_rows || logit_rows>cap) return false;
    uint64_t T=cap,L=logit_rows;
    g->ctx_cap=context; g->cap_tokens=cap; g->n_logit_rows=logit_rows;
#define MUSE_ALLOC(field,count) if (!(g->field=ds4_gpu_tensor_alloc((uint64_t)(count)*4))) goto fail;
    MUSE_GRAPH_SCRATCH(MUSE_ALLOC)
#undef MUSE_ALLOC
    for(unsigned il=0;il<DS4_N_LAYER;il++) {
        uint32_t n=muse_graph_cache_cap(il,context,cap);
        uint64_t bytes=(uint64_t)n*DS4_N_HEAD_KV*DS4_N_HEAD_DIM*2;
        g->cache_cap[il]=n;
        g->kc[il]=ds4_gpu_tensor_alloc(bytes); g->vc[il]=ds4_gpu_tensor_alloc(bytes);
        if (!g->kc[il] || !g->vc[il]) goto fail;
    }
    g->valid=true;
    return true;
fail:
    muse_graph_free(g);
    return false;
}

static bool muse_gemv(ds4_gpu_tensor *out,const ds4_model *m,const ds4_tensor *w,
        const ds4_gpu_tensor *x,uint32_t tokens, bool matmul_prefill) {
    return ds4_gpu_muse_linear_tensor(out,m->map,m->size,w->abs_offset,w->type,
                                      w->dim[0],w->dim[1],x,tokens, matmul_prefill);
}

static bool muse_norm(ds4_gpu_tensor *out,const ds4_gpu_tensor *x,const ds4_model *m,
        const ds4_tensor *w,uint32_t rows,uint32_t width) {
    return ds4_gpu_rms_norm_weight_rows_tensor(out,x,m->map,m->size,w->abs_offset,
                                               width,rows,DS4_RMS_EPS);
}

static bool muse_attention_projections(ds4_muse_gpu_graph *g,const ds4_model *m,
        const ds4_layer_weights *l,uint32_t tokens) {
    ds4_gpu_tensor *outputs[4]={g->q,g->k,g->v,g->gate};
    const ds4_tensor *weights[4]={l->attn_q,l->attn_k,l->attn_v,l->attn_gate};
    uint64_t offsets[4];uint32_t types[4],widths[4];
    for(unsigned i=0;i<4;i++) {
        offsets[i]=weights[i]->abs_offset;types[i]=weights[i]->type;
        widths[i]=weights[i]->dim[1];
    }
    return ds4_gpu_muse_linears_tensor(outputs,m->map,m->size,offsets,types,
        widths,4,DS4_N_EMBD,g->xn,tokens, g->matmul_prefill);
}

static bool muse_latent_with_prefetch(ds4_muse_gpu_graph *g,const ds4_model *m,
        const ds4_layer_weights *l,uint32_t layer,uint32_t tokens,uint64_t *routing_event) {
    uint64_t event=0;
    *routing_event=0;
    bool overlap=tokens==1&&!getenv("DS4_METAL_MUSE_LATENT_REFERENCE")&&
        !getenv("DS4_METAL_DISABLE_STREAMING_EXPERT_EARLY_LOAD");
    ds4_gpu_stream_expert_table table=graph_stream_expert_table_make(m,l,layer,
        l->ffn_gate_exps->bytes/DS4_N_EXPERT,l->ffn_down_exps->bytes/DS4_N_EXPERT);
    if(overlap) {
        if(!ds4_gpu_signal_selected_readback_ready(&event)) return false;
    } else if(tokens==1&&!ds4_gpu_muse_prefetch_experts_tensor(&table,g->selected,DS4_N_EXPERT_USED,0)) return false;
    if(!muse_gemv(g->latent,m,l->ffn_latent_down,g->xn,tokens, g->matmul_prefill)||
       !muse_norm(g->latent,g->latent,m,l->ffn_pre_exp_norm,tokens,DS4_N_MOE_LATENT)) return false;
    if(overlap&&!ds4_gpu_muse_prefetch_experts_tensor(&table,g->selected,DS4_N_EXPERT_USED,event)) return false;
    *routing_event=event;
    return true;
}

static DS4_MAYBE_UNUSED bool muse_graph_forward(ds4_muse_gpu_graph *g,const ds4_model *m,const ds4_weights *w,
        const int *tokens,uint32_t T,float *out,bool all_rows) {
    if (g) g->cancelled=false;
    if (!g || !g->valid || !T || T>g->cap_tokens ||
        (all_rows && T>g->n_logit_rows) || g->pos>g->ctx_cap || T>g->ctx_cap-g->pos) return false;
    for(unsigned t=0;t<T;t++) if(tokens[t]<0 || (unsigned)tokens[t]>=DS4_N_VOCAB) return false;
    if (!ds4_gpu_tensor_write(g->tokens,0,tokens,(uint64_t)T*4) || !ds4_gpu_begin_commands()) return false;
    bool ok=ds4_gpu_embed_tokens_quant_tensor(g->r,g->tokens,m->map,m->size,
        w->token_embd->abs_offset,w->token_embd->type,DS4_N_VOCAB,T,DS4_N_EMBD) &&
        ds4_gpu_rms_norm_plain_rows_tensor(g->r,g->r,DS4_N_EMBD,T,DS4_RMS_EPS);
    for(unsigned il=0;ok && il<DS4_N_LAYER;il++) {
        if (g->cancel && g->cancel(g->cancel_ud)) {
            g->cancelled=true;
            ok=false;
            break;
        }
        const ds4_layer_weights *l=&w->layer[il];
        uint64_t routing_event=0;
        ok=muse_norm(g->xn,g->r,m,l->attn_norm,T,DS4_N_EMBD) &&
            muse_attention_projections(g,m,l,T) &&
            muse_norm(g->q,g->q,m,l->attn_q_norm,T*DS4_N_HEAD,DS4_N_HEAD_DIM) &&
            muse_norm(g->k,g->k,m,l->attn_k_norm,T*DS4_N_HEAD_KV,DS4_N_HEAD_DIM) &&
            ds4_gpu_muse_kv_store_tensor(g->q,g->kc[il],g->vc[il],g->k,g->v,T,
                DS4_N_HEAD,DS4_N_HEAD_KV,DS4_N_HEAD_DIM,g->pos,g->cache_cap[il],DS4_ROPE_FREQ_BASE,g_muse_swa[il]!=0) &&
            (g->flash_attention ? ds4_gpu_muse_attention_flash_tensor :
                                  ds4_gpu_muse_attention_tensor)(g->attn,g->q,g->kc[il],g->vc[il],g->attn_part,T,
                DS4_N_HEAD,DS4_N_HEAD_KV,DS4_N_HEAD_DIM,g->pos,g->cache_cap[il],g_muse_swa[il]?DS4_N_SWA:0) &&
            ds4_gpu_rms_norm_plain_rows_tensor(g->attn,g->attn,DS4_N_VALUE_DIM,T*DS4_N_HEAD,DS4_RMS_EPS) &&
            ds4_gpu_muse_gate_tensor(g->attn,g->attn,g->gate,T*DS4_N_HEAD*DS4_N_VALUE_DIM) &&
            muse_gemv(g->branch,m,l->attn_output,g->attn,T, g->matmul_prefill) &&
            ds4_gpu_rms_norm_plain_rows_tensor(g->branch,g->branch,DS4_N_EMBD,T,g_ds4_shape.post_norm_eps) &&
            ds4_gpu_muse_residual_tensor(g->r,g->r,g->branch,m->map,m->size,
                l->attn_resid_alpha->abs_offset,l->attn_resid_beta->abs_offset,T,DS4_N_EMBD) &&
            muse_norm(g->xn,g->r,m,l->ffn_norm,T,DS4_N_EMBD) &&
            muse_gemv(g->router,m,l->ffn_gate_inp,g->xn,T, g->matmul_prefill) &&
            ds4_gpu_glm_router_select_batch_tensor(g->selected,g->weights,g->probs,m->map,m->size,
                l->ffn_exp_probs_b->abs_offset,g->router,DS4_N_EXPERT,DS4_N_EXPERT_USED,DS4_EXPERT_WEIGHT_SCALE,T) &&
            muse_latent_with_prefetch(g,m,l,il,T,&routing_event) &&
            ds4_gpu_muse_experts_tensor(g->mid,g->part,g->latent,g->selected,m->map,m->size,
                l->ffn_gate_exps->abs_offset,l->ffn_up_exps->abs_offset,l->ffn_down_exps->abs_offset,
                l->ffn_gate_exps->type,l->ffn_down_exps->type,T,DS4_N_EXPERT,DS4_N_EXPERT_USED,
                DS4_N_MOE_LATENT,DS4_N_FF_EXP,il,g->moe_lists,g->moe_counts,routing_event, g->matmul_prefill) &&
            ds4_gpu_muse_input_norm_tensor(g->part,g->part,m->map,m->size,l->ffn_post_exp_norm->abs_offset,
                T*DS4_N_EXPERT_USED,DS4_N_MOE_LATENT,g_ds4_shape.post_norm_eps) &&
            ds4_gpu_qwen4_moe_reduce_tensor(g->moe,g->part,g->weights,NULL,NULL,NULL,NULL,
                T,DS4_N_EXPERT_USED,DS4_N_EXPERT_USED,DS4_N_MOE_LATENT,0) &&
            muse_gemv(g->branch,m,l->ffn_latent_up,g->moe,T, g->matmul_prefill) &&
            ds4_gpu_muse_input_norm_tensor(g->branch,g->branch,m->map,m->size,l->ffn_post_norm->abs_offset,
                T,DS4_N_EMBD,g_ds4_shape.post_norm_eps) &&
            ds4_gpu_muse_residual_tensor(g->r,g->r,g->branch,m->map,m->size,
                l->ffn_resid_alpha->abs_offset,l->ffn_resid_beta->abs_offset,T,DS4_N_EMBD);
#ifdef DS4_MUSE_LAYER_OBSERVER
        /* Test builds can observe layer boundaries without a runtime hook in
         * production. Observers must leave a command buffer active on success. */
        if(ok) ok=DS4_MUSE_LAYER_OBSERVER(g,il,T);
#endif
    }
    uint32_t rows=all_rows?T:1;
    ds4_gpu_tensor *last=NULL;
    if (ok) ok=muse_norm(g->xn,g->r,m,w->output_norm,T,DS4_N_EMBD);
    if (ok && !all_rows) {
        last=ds4_gpu_tensor_view(g->xn,(uint64_t)(T-1)*DS4_N_EMBD*4,(uint64_t)DS4_N_EMBD*4);
        ok=last!=NULL;
    }
    if (ok) ok=muse_gemv(g->logits,m,w->output,last?last:g->xn,rows, g->matmul_prefill) &&
        ds4_gpu_muse_logits_tensor(g->logits,rows*DS4_N_VOCAB,g_ds4_shape.logit_scale,g_ds4_shape.logit_softcap);
    ds4_gpu_tensor_free(last);
    if (ds4_gpu_commands_active()) ok=ds4_gpu_end_commands() && ok;
    if (ok && out) ok=ds4_gpu_tensor_read(g->logits,0,out,(uint64_t)rows*DS4_N_VOCAB*4);
    if (!ok) { (void)ds4_gpu_muse_prefetch_finish();g->valid=false;return false; }
    g->pos+=T;
    if (g->pos>g->kv_highwater) g->kv_highwater=g->pos;
    return true;
}
