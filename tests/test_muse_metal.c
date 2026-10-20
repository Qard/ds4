#include "ds4_muse_gpu.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

int main(void) {
    assert(ds4_gpu_init());
    const uint64_t map_size = 65536;
    unsigned char *matrix = mmap(NULL, map_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    assert(matrix != MAP_FAILED && ds4_gpu_set_model_map(matrix, map_size));
    const unsigned types[] = {0, 8, 30, 14};
    const unsigned K = 256, N = 16, T = 2;
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(T * K * sizeof(float));
    ds4_gpu_tensor *y = ds4_gpu_tensor_alloc(T * N * sizeof(float));
    float inputs[T * K], actual[T * N];
    for (unsigned i = 0; i < T * K; i++) inputs[i] = ((int)(i % 13) - 6) * .0625f;
    assert(x && y && ds4_gpu_tensor_write(x, 0, inputs, sizeof(inputs)));
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
        assert(ds4_gpu_muse_linear_tensor(y, matrix, map_size, 0, type, K, N, x, T));
        assert(ds4_gpu_synchronize());
        assert(ds4_gpu_tensor_read(y, 0, actual, sizeof(actual)));
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
