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

/* Recipient semantics decide which bytes are executable. Public examples may
 * contain ATEM verbatim; the shared text stream must not rediscover calls in
 * those bytes. Completed calls use the existing API tool emitters at finish. */
static request muse_public_stream_request(const request *r) {
    /* Borrowed view: only the original request owns its pointers. */
    request visible=*r;visible.has_tools=false;return visible;
}
static void muse_server_output_append(const ds4_chat_decode_state *state,
        buf *visible,buf *reasoning,buf *tools,const char *piece,size_t len) {
    buf_append(state->reasoning?reasoning:state->tool_call?tools:visible,piece,len);
}
static bool muse_server_parse_output(const request *r,const buf *visible,const buf *tools,
        bool saw_start,const char **finish,char *err,size_t errlen,
        char **content,char **reasoning,tool_calls *calls,bool *recovered) {
    if(tools->len && !r->has_tools) {
        *content=xstrdup(visible->ptr?visible->ptr:"");*reasoning=NULL;
        *recovered=false;*finish="error";
        snprintf(err,errlen,"Muse generated a tool recipient without enabled tools");return false;
    }
    bool ok=parse_generated_message_for_response_for_syntax(SERVER_MODEL_SYNTAX_MUSE,
        tools->ptr?tools->ptr:"",r->has_tools,saw_start,false,finish,err,errlen,
        content,reasoning,calls,recovered,&r->tool_orders);
    buf text={0};buf_puts(&text,visible->ptr?visible->ptr:"");
    if(!ok) buf_puts(&text,*content?*content:"");
    free(*content);*content=buf_take(&text);return ok;
}
