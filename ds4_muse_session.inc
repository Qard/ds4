/* Session policy for the native Muse graph. KV remains private to each session;
 * the existing engine-global expert cache is shared by serialized forwards. */
static bool ds4_session_cancelled(ds4_session *s);

static uint32_t muse_session_prefill_cap(const ds4_engine *e,uint32_t context) {
    return MIN(context,e->prefill_chunk?e->prefill_chunk:MUSE_DEFAULT_PREFILL_CHUNK);
}

static uint64_t muse_session_memory_bytes(uint32_t context,uint32_t cap) {
    return muse_graph_bytes(context,cap)+2ull*DS4_N_VOCAB*sizeof(float)+sizeof(ds4_session);
}

static uint64_t muse_memory_available_budget(const ds4_engine *e,uint64_t available) {
    /* Diagnostic memory is already locked when host availability is sampled.
     * Restore it to this baseline because fixed admission charges it below. */
    return ds4_add_sat_u64(available,e->simulated_memory.bytes)/8*7;
}

static bool muse_memory_admit_for_host(ds4_engine *e,uint64_t graph_bytes,bool fit_cache,
        uint64_t host,uint64_t recommended) {
    if(!host || !recommended) return false;
    uint64_t budget=MIN(host/8*7,recommended),resident=e->model.size;
    if(e->muse_memory_budget) budget=MIN(budget,e->muse_memory_budget);
    if(e->ssd_streaming && !weights_streaming_non_routed_bytes(&e->weights,&resident)) return false;
    graph_bytes=MAX(graph_bytes,e->muse_reserved_graph_bytes);
    uint64_t fixed=ds4_add_sat_u64(resident,ds4_add_sat_u64(graph_bytes,
        ds4_add_sat_u64(2ull<<30,e->simulated_memory.bytes)));
    if(fixed>=budget) {
        fprintf(stderr,"ds4: Muse weights and runtime need %.2f GiB; safe budget %.2f GiB\n",
                ds4_bytes_to_gib(fixed),ds4_bytes_to_gib(budget));
        return false;
    }
    if(!e->ssd_streaming) return true;
    uint64_t expert=0;
    if(!ds4_streaming_routed_expert_bytes(&e->weights,&expert)) return false;
    uint64_t max_count=MIN((budget-fixed)/expert,(uint64_t)DS4_N_LAYER*DS4_N_EXPERT);
    uint32_t count=e->ssd_streaming_cache_experts;
    if(e->ssd_streaming_cache_bytes) count=ds4_ssd_cache_experts_for_byte_budget(e->ssd_streaming_cache_bytes,expert);
    if(!count && fit_cache) count=(uint32_t)max_count;
    if(count>max_count) {
        if(!fit_cache) {
            fprintf(stderr,"ds4: no safe memory budget for another Muse session; reduce context or expert cache\n");
            return false;
        }
        count=(uint32_t)max_count;
    }
    if(count<DS4_N_EXPERT_USED) {
        fprintf(stderr,"ds4: Muse needs room for at least %u selected experts\n",DS4_N_EXPERT_USED);
        return false;
    }
    if(fit_cache) {
        e->ssd_streaming_cache_experts=count;
        e->ssd_streaming_cache_bytes=(uint64_t)count*expert;
    }
    return true;
}

static bool muse_memory_admit(ds4_engine *e,uint64_t bytes,bool fit_cache) {
    if(!e->muse_memory_budget) {
        /* Take the available-memory baseline before installing model views or
         * pinning weights. Counting only physical capacity would overcommit
         * a Mac that already has another GPU model loaded. Speculative pages
         * are included in free_count and must not be counted twice. */
        vm_statistics64_data_t vm={0};mach_msg_type_number_t count=HOST_VM_INFO64_COUNT;
        mach_port_t host=mach_host_self();
        kern_return_t rc=host_statistics64(host,HOST_VM_INFO64,(host_info64_t)&vm,&count);
        mach_port_deallocate(mach_task_self(),host);
        if(rc!=KERN_SUCCESS) {fprintf(stderr,"ds4: cannot measure available memory for Muse\n");return false;}
        uint64_t available=((uint64_t)vm.free_count+vm.inactive_count)*(uint64_t)getpagesize();
        e->muse_memory_budget=muse_memory_available_budget(e,available);
        if(!e->muse_memory_budget) return false;
    }
    return muse_memory_admit_for_host(e,bytes,fit_cache,glm_graph_host_memory_bytes(),
                                      ds4_gpu_recommended_working_set_size());
}

static bool muse_session_admit(ds4_engine *e,uint64_t bytes) {
    /* Some frontends choose context after tokenizing their inputs. Refit the
     * cache before the first graph exists; live sessions keep their admitted
     * cache and must never trigger a resize underneath an in-flight forward. */
    bool first=e->muse_session_bytes==0;
    uint64_t count=e->placement_session_count_hint>0?e->placement_session_count_hint:1;
    uint64_t reserve=MAX(e->muse_reserved_graph_bytes,ds4_mul_sat_u64(bytes,count));
    uint32_t old_cache=e->ssd_streaming_cache_experts;
    if(!muse_memory_admit(e,MAX(reserve,ds4_add_sat_u64(e->muse_session_bytes,bytes)),first)) return false;
    e->muse_reserved_graph_bytes=reserve;
    if(e->ssd_streaming && e->ssd_streaming_cache_experts!=old_cache) {
        uint64_t expert=0;
        if(!ds4_streaming_routed_expert_bytes(&e->weights,&expert)) return false;
        ds4_gpu_set_streaming_expert_cache_budget(e->ssd_streaming_cache_experts);
        ds4_gpu_set_streaming_expert_cache_expert_bytes(expert);
        fprintf(stderr,"ds4: Muse first-session expert cache resized from %u to %u experts\n",
            old_cache,e->ssd_streaming_cache_experts);
    }
    return true;
}

static void muse_session_reset_graph(ds4_session *s) {
    s->muse_graph.pos=s->muse_graph.kv_highwater=s->muse_graph.swa_history_start=0;
    s->muse_graph.valid=true;s->muse_graph.cancelled=false;
    s->muse_rewound=false;
}

/* Re-evaluate the final retained token to regenerate its logits. The extra
 * ring history reserved by the graph makes short canonical rewrites cheap. */
static bool muse_graph_seek(ds4_muse_gpu_graph *g,uint32_t position) {
    if(!g->valid || !position || position>g->pos) return false;
    for(unsigned il=0;il<DS4_N_LAYER;il++) if(g_muse_swa[il]) {
        uint32_t first=g->kv_highwater>g->cache_cap[il]?g->kv_highwater-g->cache_cap[il]:0;
        first=MAX(first,g->swa_history_start);
        uint32_t needed=position>DS4_N_SWA?position-DS4_N_SWA:0;
        if(needed<first) return false;
    }
    g->pos=position-1;
    return true;
}

static int muse_session_sync(ds4_session *s,const ds4_tokens *prompt,char *err,size_t errlen) {
    ds4_muse_gpu_graph *g=&s->muse_graph;
    if(!s->muse_graph_ready) {snprintf(err,errlen,"Muse graph is not initialized");return 1;}
    for(int i=0;i<prompt->len;i++) if(prompt->v[i]<0 || (unsigned)prompt->v[i]>=DS4_N_VOCAB) {
        snprintf(err,errlen,"token id %d at position %d is outside the vocabulary",prompt->v[i],i);
        return 1;
    }
    g->cancel=s->cancel;g->cancel_ud=s->cancel_ud;
    uint32_t start=0;
    if(s->checkpoint_valid && g->valid) {
        uint32_t common=0,n=MIN(s->checkpoint.len,prompt->len);
        while(common<n && s->checkpoint.v[common]==prompt->v[common]) common++;
        if(common==(unsigned)s->checkpoint.len && g->pos<=(unsigned)s->checkpoint.len)
            start=g->pos;
        else if(common && muse_graph_seek(g,common)) start=g->pos;
        else muse_session_reset_graph(s);
    } else muse_session_reset_graph(s);
    s->checkpoint.len=start;
    for(uint32_t i=start;i<(uint32_t)prompt->len;) {
        if(ds4_session_cancelled(s)) {
            snprintf(err,errlen,"interrupted");return DS4_SESSION_SYNC_INTERRUPTED;
        }
        uint32_t n=MIN((uint32_t)prompt->len-i,g->cap_tokens);
        s->checkpoint_valid=false;
        if(!muse_graph_forward(g,&s->engine->model,&s->engine->weights,prompt->v+i,n,s->logits,false)) {
            if(g->cancelled) {snprintf(err,errlen,"interrupted");return DS4_SESSION_SYNC_INTERRUPTED;}
            snprintf(err,errlen,"Muse prefill failed at token %u",i);return 1;
        }
        for(unsigned j=0;j<n;j++) token_vec_push(&s->checkpoint,prompt->v[i+j]);
        i+=n;s->checkpoint_valid=true;s->muse_rewound=false;
        if(s->progress) s->progress(s->progress_ud,"prefill_chunk",i,prompt->len);
    }
    s->checkpoint_valid=true;s->muse_rewound=false;s->mtp_draft_valid=false;
    return 0;
}

static int muse_session_eval(ds4_session *s,int token,char *err,size_t errlen) {
    ds4_muse_gpu_graph *g=&s->muse_graph;
    if(!s->muse_graph_ready || token<0 || (unsigned)token>=DS4_N_VOCAB) {
        snprintf(err,errlen,"invalid Muse session or token");return 1;
    }
    if((unsigned)s->checkpoint.len>=g->ctx_cap) {snprintf(err,errlen,"context is full");return 1;}
    if(!s->checkpoint_valid && s->checkpoint.len) {
        snprintf(err,errlen,"Muse session needs sync after an invalidated forward");return 1;
    }
    g->cancel=s->cancel;g->cancel_ud=s->cancel_ud;
    if(!s->checkpoint.len && (!g->valid || g->pos)) muse_session_reset_graph(s);
    /* A rewind preserves the transcript, while cached KV may retain just its
     * prefix or require a full replay. Sampling stays disabled until caught up. */
    while(g->pos<(unsigned)s->checkpoint.len) {
        uint32_t n=MIN((uint32_t)s->checkpoint.len-g->pos,g->cap_tokens);
        if(ds4_session_cancelled(s) || !muse_graph_forward(g,&s->engine->model,&s->engine->weights,
                s->checkpoint.v+g->pos,n,s->logits,false)) goto fail;
    }
    if(ds4_session_cancelled(s) || !muse_graph_forward(g,&s->engine->model,&s->engine->weights,
                                                      &token,1,s->logits,false)) goto fail;
    token_vec_push(&s->checkpoint,token);s->checkpoint_valid=true;s->muse_rewound=false;
    s->mtp_draft_valid=false;
    return 0;
fail:
    s->checkpoint_valid=false;
    snprintf(err,errlen,g->cancelled || ds4_session_cancelled(s)?"interrupted":"Muse decode failed");return 1;
}
