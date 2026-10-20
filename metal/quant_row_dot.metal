/* Shared native GGUF row dot: Q8_0, Q4_0, Q4_K, Q6_K, Q2_K, IQ2_XXS, MXFP4, BF16, F16, F32. */
static inline float ds4_quant_row_dot(device const char *row, device const float *x,
                                  uint weight_type, uint in_dim, ushort tiisg) {
    float acc = 0.0f;
    if (weight_type == 8) {
        const short ix = tiisg / 8, it = tiisg % 8;
        const uint nb = in_dim / 32;
        for (uint ib = (uint)ix; ib < nb; ib += 4) {
            device const char *b = row + (uint64_t)ib * 34;
            device const float *y = x + ib * 32 + (uint)it * 2;
            const float d = (float)(*(device const half *)b);
            device const char *q = b + 2 + it * 2;
            acc += d * (y[0] * (float)q[0] + y[1] * (float)q[1] +
                        y[16] * (float)q[16] + y[17] * (float)q[17]);
        }
    } else if (weight_type == 39) {
        const short ix = tiisg / 8, it = tiisg % 8;
        const uint nb = in_dim / 32;
        for (uint ib = (uint)ix; ib < nb; ib += 4) {
            device const uchar *b = (device const uchar *)(row + (uint64_t)ib * 17);
            device const float *y = x + ib * 32 + (uint)it * 2;
            const float d = ds4_metal_e8m0_to_f32(b[0]);
            const uint q0 = b[1 + it * 2], q1 = b[2 + it * 2];
            acc += d * (y[0] * ds4_metal_mxfp4_values[q0 & 0xfu] +
                        y[1] * ds4_metal_mxfp4_values[q1 & 0xfu] +
                        y[16] * ds4_metal_mxfp4_values[q0 >> 4] +
                        y[17] * ds4_metal_mxfp4_values[q1 >> 4]);
        }
    } else if (weight_type == 2) {
        const short ix = tiisg / 8, it = tiisg % 8;
        const uint nb = in_dim / 32;
        for (uint ib = (uint)ix; ib < nb; ib += 4) {
            device const uchar *b = (device const uchar *)(row + (uint64_t)ib * 18);
            device const float *y = x + ib * 32 + (uint)it * 2;
            const float d = (float)(*(device const half *)b);
            const uint q0 = b[2 + it * 2], q1 = b[3 + it * 2];
            acc += d * (y[0] * ((float)(q0 & 0xfu) - 8.0f) + y[1] * ((float)(q1 & 0xfu) - 8.0f) +
                        y[16] * ((float)(q0 >> 4) - 8.0f) + y[17] * ((float)(q1 >> 4) - 8.0f));
        }
    } else if (weight_type == 12) {
        /* q4_K: 256-element super-blocks (d, dmin, 12 packed 6-bit scale/min pairs, 128 nibble bytes);
         * lane owns 8 consecutive elements of every block: group = lane/4, l = (lane%4)*8 */
        const uint nb = in_dim / 256;
        const uint group = tiisg / 4, l = (tiisg % 4) * 8;
        for (uint ib = 0; ib < nb; ib++) {
            device const uchar *blk = (device const uchar *)(row + (uint64_t)ib * 144);
            const float d = (float)(*(device const half *)blk);
            const float dmin = (float)(*(device const half *)(blk + 2));
            device const uchar *sc = blk + 4;
            uint s, mn;
            if (group < 4) { s = sc[group] & 63u; mn = sc[group + 4] & 63u; }
            else { s = (sc[group + 4] & 0xFu) | ((sc[group - 4] & 0xC0u) >> 2); mn = (sc[group + 4] >> 4) | ((sc[group] & 0xC0u) >> 2); }
            const float ds = d * (float)s, dm = dmin * (float)mn;
            device const uchar *qs = blk + 16 + (group >> 1) * 32 + l;
            const uint shift = (group & 1u) * 4u;
            device const float *y = x + ib * 256 + group * 32 + l;
            for (uint i = 0; i < 8; i++) acc += (ds * (float)((qs[i] >> shift) & 0xFu) - dm) * y[i];
        }
    } else if (weight_type == 10) {
        /* q2_K: 84-byte super-blocks of 256 (16 scale/min nibble pairs, 64 packed 2-bit bytes, d, dmin);
         * lane owns 8 consecutive elements: group = lane/2 (16 per block), l = (lane%2)*8 */
        const uint nb = (in_dim + 255u) / 256u;
        const uint group = tiisg / 2, l = (tiisg % 2) * 8;
        const uint q_base = 32u * (group / 8u) + 16u * (group & 1u), shift = ((group / 2u) & 3u) * 2u;
        for (uint ib = 0; ib < nb; ib++) {
            /* Padded Q2_K down weights have no corresponding activation tail. */
            if (ib * 256u + group * 16u + l >= in_dim) continue;
            device const uchar *blk = (device const uchar *)(row + (uint64_t)ib * 84);
            const float d = (float)(*(device const half *)(blk + 80));
            const float dmin = (float)(*(device const half *)(blk + 82));
            const uint sc = blk[group];
            const float ds = d * (float)(sc & 0xFu), dm = dmin * (float)(sc >> 4);
            device const uchar *qs = blk + 16 + q_base + l;
            device const float *y = x + ib * 256 + group * 16 + l;
            for (uint i = 0; i < 8; i++) acc += (ds * (float)((qs[i] >> shift) & 3u) - dm) * y[i];
        }
    } else if (weight_type == 16) {
        /* iq2_xxs: 66-byte super-blocks of 256 (d, 8 x 4 u16: 4 grid bytes + 28 sign bits + 4-bit
         * scale per 32); lane owns one 8-value grid entry: sub-block = lane/4, entry = lane%4 */
        const uint nb = in_dim / 256;
        const uint ib32 = tiisg / 4, j = tiisg % 4;
        for (uint ib = 0; ib < nb; ib++) {
            device const uchar *blk = (device const uchar *)(row + (uint64_t)ib * 66);
            const float d = (float)(*(device const half *)blk);
            device const ushort *q2 = (device const ushort *)(blk + 2) + 4 * ib32;
            const uint aux_g = (uint)q2[0] | ((uint)q2[1] << 16);
            const uint aux_s = (uint)q2[2] | ((uint)q2[3] << 16);
            const float dl = d * (0.5f + (float)(aux_s >> 28)) * 0.25f;
            constant const uchar *grid = (constant const uchar *)(ds4_metal_iq2xxs_grid + ((aux_g >> (8 * j)) & 0xFFu));
            const uint signs = ds4_metal_ksigns_iq2xs[(aux_s >> (7 * j)) & 127u];
            device const float *y = x + ib * 256 + ib32 * 32 + j * 8;
            float part = 0.0f;
            for (uint i = 0; i < 8; i++) part += (float)grid[i] * ((signs >> i) & 1u ? -y[i] : y[i]);
            acc += dl * part;
        }
    } else if (weight_type == 14) {
        for (uint i = tiisg; i < in_dim; i += 32)
            acc += ds4_glm_q6_K_value((device const block_q6_K *)row, i) * x[i];
    } else if (weight_type == 30) {
        device const ushort *w = (device const ushort *)row;
        for (uint i = tiisg * 4; i < in_dim; i += 128) {
            acc += as_type<float>((uint)w[i] << 16) * x[i] + as_type<float>((uint)w[i + 1] << 16) * x[i + 1] +
                   as_type<float>((uint)w[i + 2] << 16) * x[i + 2] + as_type<float>((uint)w[i + 3] << 16) * x[i + 3];
        }
    } else if (weight_type == 1) {
        device const half *w = (device const half *)row;
        for (uint i = tiisg * 4; i < in_dim; i += 128) {
            acc += (float)w[i] * x[i] + (float)w[i + 1] * x[i + 1] + (float)w[i + 2] * x[i + 2] + (float)w[i + 3] * x[i + 3];
        }
    } else {
        device const float *w = (device const float *)row;
        for (uint i = tiisg * 4; i < in_dim; i += 128) {
            acc += w[i] * x[i] + w[i + 1] * x[i + 1] + w[i + 2] * x[i + 2] + w[i + 3] * x[i + 3];
        }
    }
    return simd_sum(acc);
}

