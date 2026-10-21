/* Native Muse KV uses the shared session payload envelope and bounded IO.
 * Store ring rows in token order, independent of the saving prefill capacity. */
#define DS4_MUSE_PAYLOAD_TAG 0x4d555301u
#define DS4_MUSE_FLASH_PAYLOAD_TAG 0x4d555302u

static uint32_t muse_payload_tag(const ds4_muse_gpu_graph *g) {
    return g->flash_attention ? DS4_MUSE_FLASH_PAYLOAD_TAG : DS4_MUSE_PAYLOAD_TAG;
}

static uint64_t muse_payload_size(uint32_t rows, uint32_t swa_live) {
    uint64_t bytes = DS4_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t) +
                     (uint64_t)rows * sizeof(uint32_t) +
                     (uint64_t)DS4_N_VOCAB * sizeof(float);
    for (unsigned il = 0; il < DS4_N_LAYER; il++)
        bytes += (uint64_t)(g_muse_swa[il] ? swa_live : rows) *
                 DS4_N_HEAD_KV * DS4_N_HEAD_DIM * 4;
    return bytes;
}

static uint32_t muse_payload_swa_live(const ds4_muse_gpu_graph *g) {
    uint32_t live = g->pos;
    for (unsigned il = 0; il < DS4_N_LAYER; il++) if (g_muse_swa[il]) {
        uint32_t first = g->kv_highwater > g->cache_cap[il] ?
                         g->kv_highwater - g->cache_cap[il] : 0;
        first = MAX(first, g->swa_history_start);
        if (first > g->pos) return 0;
        live = MIN(live, g->pos - first);
    }
    return live;
}

static uint64_t muse_session_payload_bytes(ds4_session *s) {
    if (!s->muse_graph_ready || !s->checkpoint_valid || s->muse_rewound ||
        !s->muse_graph.valid || !s->checkpoint.len ||
        s->muse_graph.pos != (unsigned)s->checkpoint.len) return 0;
    uint32_t live = muse_payload_swa_live(&s->muse_graph);
    if (live < MIN(s->muse_graph.pos, DS4_N_SWA)) return 0;
    return muse_payload_size(s->muse_graph.pos, live);
}

static int muse_payload_write_ring(FILE *fp, const ds4_gpu_tensor *tensor,
        uint32_t cap, uint32_t start, uint32_t rows, uint8_t *buf,
        char *err, size_t errlen) {
    const uint64_t stride = (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM * 2;
    const uint32_t slot = start % cap, first = MIN(rows, cap - slot);
    int rc = payload_write_tensor_span(fp, tensor, slot * stride, first * stride,
                                      buf, DS4_SESSION_IO_CHUNK, err, errlen);
    if (!rc && first < rows)
        rc = payload_write_tensor_span(fp, tensor, 0, (rows - first) * stride,
                                       buf, DS4_SESSION_IO_CHUNK, err, errlen);
    return rc;
}

static int muse_payload_read_ring(FILE *fp, ds4_gpu_tensor *tensor,
        uint32_t cap, uint32_t end, uint32_t saved_rows, uint8_t *buf,
        uint64_t *remaining, char *err, size_t errlen) {
    const uint64_t stride = (uint64_t)DS4_N_HEAD_KV * DS4_N_HEAD_DIM * 2;
    const uint32_t rows = MIN(saved_rows, cap), slot = (end - rows) % cap;
    const uint32_t first = MIN(rows, cap - slot);
    int rc = payload_skip_bytes(fp, (saved_rows - rows) * stride,
                                buf, DS4_SESSION_IO_CHUNK, remaining, err, errlen);
    if (!rc) rc = payload_read_tensor_span(fp, tensor, slot * stride, first * stride,
                        buf, DS4_SESSION_IO_CHUNK, remaining, err, errlen);
    if (!rc && first < rows)
        rc = payload_read_tensor_span(fp, tensor, 0, (rows - first) * stride,
                        buf, DS4_SESSION_IO_CHUNK, remaining, err, errlen);
    return rc;
}

static int muse_session_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen) {
    if (!muse_session_payload_bytes(s) || !ds4_gpu_synchronize()) {
        payload_set_err(err, errlen, "Muse session has no synchronized checkpoint to save");
        return 1;
    }
    const ds4_muse_gpu_graph *g = &s->muse_graph;
    const uint32_t rows = g->pos, live = muse_payload_swa_live(g);
    const uint32_t h[DS4_SESSION_PAYLOAD_U32_FIELDS] = {
        DS4_SESSION_PAYLOAD_MAGIC, DS4_SESSION_PAYLOAD_VERSION, g->ctx_cap,
        DS4_N_SWA, live, DS4_N_HEAD, DS4_N_MOE_LATENT, rows,
        DS4_N_LAYER, DS4_N_HEAD_KV, DS4_N_HEAD_DIM, DS4_N_VOCAB, muse_payload_tag(g)
    };
    for (unsigned i = 0; i < DS4_SESSION_PAYLOAD_U32_FIELDS; i++)
        if (payload_write_u32(fp, h[i], err, errlen)) return 1;
    for (unsigned i = 0; i < rows; i++)
        if (payload_write_u32(fp, s->checkpoint.v[i], err, errlen)) return 1;
    if (payload_write_bytes(fp, s->logits, (uint64_t)DS4_N_VOCAB * 4, err, errlen)) return 1;
    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = 0;
    for (unsigned il = 0; il < DS4_N_LAYER && !rc; il++) {
        uint32_t n = g_muse_swa[il] ? live : rows;
        rc = muse_payload_write_ring(fp, g->kc[il], g->cache_cap[il], rows - n, n, buf, err, errlen);
        if (!rc) rc = muse_payload_write_ring(fp, g->vc[il], g->cache_cap[il], rows - n, n, buf, err, errlen);
    }
    free(buf);
    return rc;
}

static int muse_session_load_payload(ds4_session *s, FILE *fp, const uint32_t *h,
        uint64_t remaining, char *err, size_t errlen) {
    ds4_muse_gpu_graph *g = &s->muse_graph;
    const uint32_t rows = h[7], live = h[4];
    if (!s->muse_graph_ready || h[12] != muse_payload_tag(g) ||
        !rows || rows > h[2] || h[2] > 1048576 || rows > g->ctx_cap ||
        h[3] != DS4_N_SWA || h[5] != DS4_N_HEAD || h[6] != DS4_N_MOE_LATENT ||
        h[8] != DS4_N_LAYER || h[9] != DS4_N_HEAD_KV ||
        h[10] != DS4_N_HEAD_DIM || h[11] != DS4_N_VOCAB ||
        live < MIN(rows, DS4_N_SWA) || live > rows ||
        live > MIN(rows, DS4_N_SWA + 8192u) ||
        remaining != muse_payload_size(rows, live) - DS4_SESSION_PAYLOAD_U32_FIELDS * 4) {
        payload_set_err(err, errlen, "incompatible or truncated Muse session payload");
        return 1;
    }
    token_vec tokens = {0};
    float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(float));
    uint8_t *buf = xmalloc(DS4_SESSION_IO_CHUNK);
    int rc = 1;
    for (unsigned i = 0; i < rows; i++) {
        uint32_t token;
        if (payload_read_u32(fp, &token, &remaining, err, errlen)) goto done;
        if (token >= DS4_N_VOCAB) {
            payload_set_err(err, errlen, "invalid token in Muse session payload");
            goto done;
        }
        token_vec_push(&tokens, token);
    }
    if (payload_read_bytes(fp, logits, (uint64_t)DS4_N_VOCAB * 4, &remaining, err, errlen)) goto done;
    for (unsigned i = 0; i < DS4_N_VOCAB; i++) {
        /* The release build uses fast-math: inspect bits rather than letting
         * the compiler assume that payload values are finite. */
        uint32_t bits;
        memcpy(&bits, &logits[i], sizeof(bits));
        if ((bits & 0x7f800000u) == 0x7f800000u) {
            payload_set_err(err, errlen, "non-finite Muse checkpoint logits");
            goto done;
        }
    }
    /* Validation above leaves an existing checkpoint untouched. Once KV writes
     * start, any failure requires replay because that checkpoint was overwritten. */
    s->checkpoint_valid = false;
    g->valid = false;
    g->pos = g->kv_highwater = g->swa_history_start = 0;
    if (!ds4_gpu_synchronize()) {
        payload_set_err(err, errlen, "failed to synchronize Muse checkpoint restore");
        goto done;
    }
    uint32_t kept = live;
    for (unsigned il = 0; il < DS4_N_LAYER; il++) {
        uint32_t n = g_muse_swa[il] ? live : rows;
        if (g_muse_swa[il]) kept = MIN(kept, g->cache_cap[il]);
        if (muse_payload_read_ring(fp, g->kc[il], g->cache_cap[il], rows, n, buf, &remaining, err, errlen) ||
            muse_payload_read_ring(fp, g->vc[il], g->cache_cap[il], rows, n, buf, &remaining, err, errlen)) goto done;
    }
    if (!ds4_gpu_tensor_write(g->logits, 0, logits, (size_t)DS4_N_VOCAB * 4)) {
        payload_set_err(err, errlen, "failed to restore Muse checkpoint logits");
        goto done;
    }
    token_vec_free(&s->checkpoint);
    s->checkpoint = tokens;
    memset(&tokens, 0, sizeof(tokens));
    memcpy(s->logits, logits, (size_t)DS4_N_VOCAB * 4);
    g->pos = g->kv_highwater = rows;
    g->swa_history_start = rows - kept;
    g->valid = true;
    s->checkpoint_valid = true;
    s->muse_rewound = s->mtp_draft_valid = false;
    rc = 0;
done:
    token_vec_free(&tokens);
    free(logits);
    free(buf);
    return rc;
}
