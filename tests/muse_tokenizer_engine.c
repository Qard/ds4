/* Frontend tests need the real tokenizer without allocating model weights or
 * a session. Keep this construction entirely outside the production API. */
#include "../ds4.c"
ds4_engine *ds4_test_muse_tokenizer_engine(const char *path) {
    ds4_engine *e = xcalloc(1, sizeof(*e));
    model_open(&e->model, path, false, false);
    config_validate_model(&e->model);
    vocab_load(&e->vocab, &e->model);
    return e;
}

ds4_session *ds4_test_muse_empty_session(void) { return xcalloc(1,sizeof(ds4_session)); }
