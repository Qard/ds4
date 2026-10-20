/* Recipient control is decoded from token IDs. Reasoning travels separately
 * from the content/tool buffer, so literal <think> text cannot change channels.
 * Reuse each API's existing SSE emitters and item/block lifetimes. */
static bool muse_sse_reasoning_update(int fd,const request *r,const char *id,
        openai_stream *openai,anthropic_stream *anthropic,responses_stream *responses,
        const buf *reasoning,size_t *emitted,bool inside) {
    if(!ds4_think_mode_enabled(r->think_mode)) {*emitted=reasoning->len;return true;}
    size_t limit=utf8_stream_safe_len(reasoning->ptr?reasoning->ptr:"",*emitted,reasoning->len,!inside);
    if(limit>*emitted) {
        const char *p=reasoning->ptr+*emitted;size_t n=limit-*emitted;
        if(r->api==API_OPENAI) {
            if(!sse_chat_delta_n(fd,r,id,"reasoning_content",p,n)) return false;
            openai->sent_reasoning=true;
        } else if(r->api==API_ANTHROPIC) {
            if(anthropic->open_block!=ANTH_BLOCK_THINKING &&
                !anthropic_sse_close_block_live(fd,id,anthropic)) return false;
            if(!anthropic_sse_open_block(fd,anthropic,ANTH_BLOCK_THINKING) ||
               !anthropic_sse_delta_live(fd,anthropic,ANTH_BLOCK_THINKING,p,n)) return false;
            anthropic->sent_thinking=true;
        } else if(r->api==API_RESPONSES && r->reasoning_summary_emit) {
            if(!responses->reasoning_item_opened) {
                responses->reasoning_index=responses->next_output_index++;
                if(!responses_sse_reasoning_added(fd,responses)) return false;
                responses->reasoning_item_opened=true;
            }
            if(!responses->reasoning_summary_started) {
                if(!responses_sse_reasoning_summary_part_added(fd,responses)) return false;
                responses->reasoning_summary_started=true;
            }
            if(!responses_sse_reasoning_delta(fd,responses,p,n)) return false;
            buf_append(&responses->reasoning_text,p,n);responses->reasoning_emitted_any=true;
        }
        *emitted=limit;
    }
    responses->reasoning_closed_naturally=!inside;
    if(!inside && r->api==API_ANTHROPIC && anthropic->open_block==ANTH_BLOCK_THINKING)
        return anthropic_sse_close_block_live(fd,id,anthropic);
    return true;
}
