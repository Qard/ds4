/* Sparse headers exercise the native loader without touching model weights. */
#include "../ds4.c"
#include <assert.h>

int main(int argc, char **argv) {
    assert(argc == 2);
    ds4_model m;
    model_open(&m, argv[1], false, false);
    config_validate_model(&m);
    assert(DS4_N_LAYER == 62 && DS4_N_EMBD == 8192 && DS4_N_HEAD_KV == 16);
    ds4_weights w;
    weights_bind(&w, &m, false, 0, UINT32_MAX, true, false);
    assert(weights_layers_bound(&w, 0, UINT32_MAX) && weights_have_output_head(&w));
    ds4_model_map_span_vec resident = {0};
    model_map_span_vec_include_layer_decode_static(&resident, &w.layer[0]);
    assert(resident.len == 20);
    for (unsigned t = 0; t < 3; t++) {
        const ds4_tensor *expert = t == 0 ? w.layer[0].ffn_gate_exps : t == 1 ? w.layer[0].ffn_up_exps : w.layer[0].ffn_down_exps;
        for (unsigned i = 0; i < resident.len; i++)
            assert(resident.v[i].end <= expert->abs_offset || resident.v[i].off >= expert->abs_offset + expert->bytes);
    }
    free(resident.v);
    model_close(&m);
    puts("Muse native GGUF layout: PASS");
    return 0;
}
