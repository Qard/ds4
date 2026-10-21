/* Admission checks read headers only, including when using the real checkpoint. */
#include "../ds4.c"
#include <assert.h>
int main(int argc,char **argv) {
    assert(argc==2);
    ds4_engine e={0};e.ssd_streaming=true;
    model_open(&e.model,argv[1],true,false);config_validate_model(&e.model);
    weights_bind(&e.weights,&e.model,false,0,UINT32_MAX,true,false);
    const uint64_t gib=UINT64_C(1073741824);
    ds4_context_memory estimate=ds4_context_memory_estimate_with_prefill_mode(DS4_BACKEND_METAL,32768,32,true);
    assert(estimate.total_bytes==muse_graph_bytes(32768,32));
    assert(estimate.raw_bytes+estimate.scratch_bytes==estimate.total_bytes && estimate.prefill_cap==32);
    ds4_context_memory defaults=ds4_context_memory_estimate_with_prefill_mode(DS4_BACKEND_METAL,8192,0,true);
    assert(defaults.prefill_cap==512 && defaults.total_bytes==muse_graph_bytes(8192,512));
    assert(muse_session_prefill_cap(&e,8192)==512);
    assert(muse_session_prefill_cap(&e,100)==100);
    assert(muse_session_memory_bytes(8192,muse_session_prefill_cap(&e,8192))==muse_session_memory_bytes(8192,512));
    e.prefill_chunk=17;assert(muse_session_prefill_cap(&e,8192)==17);e.prefill_chunk=0;
    uint64_t graph=muse_session_memory_bytes(32768,32),resident=0,expert=0;
    assert(weights_streaming_non_routed_bytes(&e.weights,&resident));
    assert(ds4_streaming_routed_expert_bytes(&e.weights,&expert));
    assert(muse_memory_admit_for_host(&e,graph,true,128*gib,96*gib));
    uint32_t count=e.ssd_streaming_cache_experts;
    assert(count>=DS4_N_EXPERT_USED && count<=DS4_N_LAYER*DS4_N_EXPERT);
    assert(resident+graph+2*gib+e.ssd_streaming_cache_bytes<=96*gib);
    assert(!muse_memory_admit_for_host(&e,80*gib,false,128*gib,96*gib));
    assert(e.ssd_streaming_cache_experts==count);
    assert(!muse_memory_admit_for_host(&e,graph,true,16*gib,12*gib));
    e.ssd_streaming_cache_experts=0;e.ssd_streaming_cache_bytes=0;
    e.simulated_memory.bytes=16*gib;
    assert(muse_memory_available_budget(&e,64*gib)==70*gib);
    assert(muse_memory_admit_for_host(&e,graph,true,128*gib,96*gib));
    assert(e.ssd_streaming_cache_experts<count);
    e.simulated_memory.bytes=0;e.ssd_streaming_cache_experts=0;e.ssd_streaming_cache_bytes=0;
    e.muse_reserved_graph_bytes=6*graph;
    assert(muse_memory_admit_for_host(&e,graph,true,128*gib,96*gib));
    assert(muse_memory_admit_for_host(&e,6*graph,false,128*gib,96*gib));
    assert(!muse_memory_admit_for_host(&e,7*graph,false,128*gib,96*gib));
    e.muse_memory_budget=8*gib;
    assert(!muse_memory_admit_for_host(&e,graph,true,128*gib,96*gib));
    e.muse_memory_budget=0;
    e.muse_reserved_graph_bytes=0;e.ssd_streaming=false;
    assert(!muse_memory_admit_for_host(&e,graph,false,128*gib,96*gib));
    printf("Muse 128 GiB budget: %.2f GiB resident, %.2f GiB per 32K session, %u initial cached experts: PASS\n",
           ds4_bytes_to_gib(resident),ds4_bytes_to_gib(graph),count);
    model_close(&e.model);
}
