/* ATEM values are delimiter-bounded text, deliberately not XML. Preserve
 * string whitespace and let schemas distinguish a numeric-looking string from
 * a JSON scalar. Attribute entities apply only to names. */
static char *muse_tool_attr(const char *tag,const char *name) {
    const char *p=tag,*value=NULL;size_t len=0;
    while(*p && !isspace((unsigned char)*p) && *p!='>') p++;
    if(!ds4_tool_attr_span(p,name,&value,&len)) return NULL;
    char *raw=xstrndup(value,len),*decoded=dsml_unescape_text(raw);free(raw);return decoded;
}

/* Parse one complete block and return its exact boundary. Rendering reuses
 * this grammar so closing-looking text inside a parameter cannot split a call. */
static bool muse_parse_tool_block(const char **cursor,tool_calls *calls,
                                  const tool_schema_orders *orders) {
    const char *p=skip_ascii_ws(*cursor);
    if(strncmp(p,"<atem:function_calls>",strlen("<atem:function_calls>"))) return false;
    p+=strlen("<atem:function_calls>");bool any=false;
    for(;;) {
        p=skip_ascii_ws(p);
        if(!strncmp(p,"</atem:function_calls>",strlen("</atem:function_calls>"))) {
            p+=strlen("</atem:function_calls>");if(!any) return false;*cursor=p;return true;
        }
        if(strncmp(p,"<atem:invoke ",strlen("<atem:invoke "))) return false;
        const char *tag_end=strchr(p,'>');if(!tag_end) return false;
        char *tag=xstrndup(p,tag_end-p+1),*name=muse_tool_attr(tag,"name");free(tag);
        if(!name || !name[0]) {free(name);return false;}
        p=tag_end+1;buf args={0};json_args seen={0};bool ok=true;
        for(;;) {
            const char *q=skip_ascii_ws(p);
            if(!strncmp(q,"</atem:invoke>",strlen("</atem:invoke>"))) {p=q+strlen("</atem:invoke>");break;}
            if(strncmp(q,"<atem:parameter ",strlen("<atem:parameter "))) {ok=false;break;}
            tag_end=strchr(q,'>');if(!tag_end) {ok=false;break;}
            tag=xstrndup(q,tag_end-q+1);char *key=muse_tool_attr(tag,"name");free(tag);
            if(!key || !key[0] || json_args_find_unused(&seen,key)>=0) {free(key);ok=false;break;}
            json_args_push(&seen,(json_arg){.key=xstrdup(key),.value=NULL});
            const char *end=strstr(tag_end+1,"</atem:parameter>");
            if(!end) {free(key);ok=false;break;}
            char *value=xstrndup(tag_end+1,end-(tag_end+1));
            bool string=qwen_param_declared_string(orders,name,key) || !qwen_param_value_is_json(value);
            tool_call_json_args_add(&args,key,value,string?"true":"false");
            free(key);free(value);p=end+strlen("</atem:parameter>");
        }
        json_args_free(&seen);
        if(!ok) {free(name);buf_free(&args);return false;}
        buf wrapped={0};buf_putc(&wrapped,'{');buf_puts(&wrapped,args.ptr?args.ptr:"");buf_putc(&wrapped,'}');
        tool_calls_push(calls,(tool_call){.name=name,.arguments=buf_take(&wrapped)});buf_free(&args);any=true;
    }
}

static bool parse_muse_generated_message(const char *text,char **content_out,
        char **reasoning_out,tool_calls *calls,const tool_schema_orders *orders) {
    const char *start=strstr(text,"<atem:function_calls>");
    *reasoning_out=NULL;
    if(!start) {*content_out=xstrdup(text);return true;}
    const char *p=start;
    while(*(p=skip_ascii_ws(p))) if(!muse_parse_tool_block(&p,calls,orders)) return false;
    if(!calls->len) return false;
    *content_out=xstrndup(text,start-text);
    calls->raw_tool_text=xstrndup(start,p-start);
    return true;
}
