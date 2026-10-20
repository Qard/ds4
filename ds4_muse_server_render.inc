/* Text-only Muse adapter. The server already normalizes schemas and splits
 * Anthropic result blocks; reuse those representations rather than parsing a
 * second request format here. */
static void append_muse_text(buf *out,const char *text) {
    char *escaped=ds4_muse_escape_text(text);buf_puts(out,escaped);free(escaped);
}
static void append_muse_header_text(buf *out,const char *text) {
    char *escaped=ds4_muse_escape_text(text);
    for(char *p=escaped;*p;p++) if(*p=='\r' || *p=='\n') *p=' ';
    buf_puts(out,escaped);free(escaped);
}
static void append_muse_attr(buf *out,const char *text) {
    buf header={0};append_muse_header_text(&header,text);
    for(const char *p=header.ptr?header.ptr:"";*p;p++) {
        if(*p=='\'') buf_puts(out,"&#39;");
        else {char c[2]={*p,0};append_dsml_attr_escaped(out,c);}
    }
    buf_free(&header);
}
static const char *muse_json_field(const json_args *args,const char *key) {
    for(int i=0;i<args->len;i++) if(!strcmp(args->v[i].key,key)) return args->v[i].value;
    return NULL;
}
static char *muse_tools_prompt(const char *schemas,buf *recipients) {
    const char *p=schemas?schemas:"";buf functions={0};bool first=true;
    json_ws(&p);
    while(*p) {
        char *raw=NULL;if(!json_raw_value(&p,&raw)) break;
        json_args fields={0};
        if(json_args_parse(raw,&fields)) {
            const char *name=muse_json_field(&fields,"name");
            const char *desc=muse_json_field(&fields,"description");
            const char *params=muse_json_field(&fields,"parameters");
            const char *deferred=muse_json_field(&fields,"defer_loading");
            if(name && name[0]) {
                if(first) {buf_puts(recipients,"\"[func_name] (e.g., ");append_muse_header_text(recipients,name);buf_puts(recipients,")\", ");first=false;}
                buf_puts(&functions,"\n{\"name\": ");json_escape(&functions,name);
                buf_puts(&functions,", \"description\": ");
                if(deferred && !strcmp(deferred,"true")) {
                    buf d={0};buf_puts(&d,desc?desc:"");buf_puts(&d," [deferred: use tool_search to load full schema]");
                    json_escape(&functions,d.ptr);buf_free(&d);buf_puts(&functions,", \"defer_loading\": true}");
                } else {
                    json_escape(&functions,desc?desc:"");buf_puts(&functions,", \"parameters\": ");
                    json_args properties={0};
                    if(params && json_args_parse(params,&properties) && properties.len) {
                        size_t n=strlen(params);while(n && isspace((unsigned char)params[n-1])) n--;
                        if(muse_json_field(&properties,"additionalProperties")) buf_append(&functions,params,n);
                        else {buf_append(&functions,params,n-1);buf_puts(&functions,", \"additionalProperties\": false}");}
                    } else buf_puts(&functions,"{\"type\": \"object\", \"properties\": {}, \"additionalProperties\": false}");
                    json_args_free(&properties);buf_putc(&functions,'}');
                }
            }
        }
        json_args_free(&fields);free(raw);json_ws(&p);
    }
    char *escaped=ds4_muse_escape_text(functions.ptr?functions.ptr:"");
    char *out=ds4_muse_tool_prompt("\n{\"name\": \"\", \"description\": \"\"}",escaped);
    free(escaped);buf_free(&functions);return out;
}
static void append_muse_tool_call(buf *out,const tool_call *tc) {
    buf_puts(out,"<atem:function_calls>\n<atem:invoke name=\"");append_muse_attr(out,tc->name);buf_puts(out,"\">\n");
    json_args args={0};
    if(json_args_parse(tc->arguments,&args)) for(int i=0;i<args.len;i++) {
        buf_puts(out,"<atem:parameter name=\"");append_muse_attr(out,args.v[i].key);buf_puts(out,"\">");
        append_muse_text(out,args.v[i].value);buf_puts(out,"</atem:parameter>\n");
    }
    json_args_free(&args);buf_puts(out,"</atem:invoke>\n</atem:function_calls>");
}
static void append_muse_tool_calls(buf *out,const tool_calls *calls) {
    if(!calls) return;
    if(calls->raw_tool_text) {buf_puts(out,calls->raw_tool_text);return;}
    for(int i=0;i<calls->len;i++) append_muse_tool_call(out,&calls->v[i]);
}
static void muse_message_open(buf *out,const char *header) {
    buf_puts(out,"<|start|>");buf_puts(out,header);buf_puts(out,"<|message|>");
}
static const char *muse_tool_name(const chat_msgs *msgs,const char *id) {
    for(int i=msgs?msgs->len-1:-1;i>=0;i--) for(int j=0;j<msgs->v[i].calls.len;j++) {
        const tool_call *tc=&msgs->v[i].calls.v[j];
        if(id && tc->id && !strcmp(tc->id,id)) return tc->name;
    }
    return id && id[0]?id:"tool";
}
static void append_muse_assistant(buf *out,const chat_msg *m,bool same_role) {
    const char *content=m->content?m->content:"",*reasoning=m->reasoning?m->reasoning:"";
    if(reasoning[0]) {muse_message_open(out,"assistant to=self");append_muse_text(out,reasoning);buf_puts(out,"<|eom|>");}
    if(m->calls.len) {
        if(content[0]) {muse_message_open(out,"assistant to=commentary");append_muse_text(out,content);buf_puts(out,"<|eom|>");}
        const char *raw=m->calls.raw_tool_text;
        for(int j=0;j<m->calls.len;j++) {
            buf_puts(out,"<|start|>assistant to=");append_muse_header_text(out,m->calls.v[j].name);buf_puts(out,"<|message|>");
            const char *end=raw?strstr(raw,"</atem:function_calls>"):NULL;
            if(end) {end+=strlen("</atem:function_calls>");buf_append(out,raw,end-raw);raw=end;}
            else append_muse_tool_call(out,&m->calls.v[j]);
            buf_puts(out,j+1<m->calls.len || same_role?"<|eom|>":"<|eot|>");
        }
    } else if(content[0] || !reasoning[0]) {
        muse_message_open(out,"assistant to=user");append_muse_text(out,content);buf_puts(out,"<|eot|>");
    }
}
static char *render_muse_chat(const chat_msgs *source,int start,const char *schemas,ds4_think_mode mode,bool live_tail) {
    buf out={0};chat_msgs fragments={0};chat_msgs views=ds41_message_views(source,&fragments,&start);
    const chat_msgs *msgs=&views;
    if(live_tail) buf_puts(&out,"<|eot|>");
    else {
        const char *system=NULL;
        if(msgs->len && role_is_system(msgs->v[0].role)) {system=msgs->v[0].content;if(start<1) start=1;}
        buf recipients={0};char *tools=schemas && schemas[0]?muse_tools_prompt(schemas,&recipients):NULL;
        char *text=ds4_muse_system_prompt(system,tools,recipients.ptr,mode,NULL);
        buf_puts(&out,"<|begin_of_text|>");muse_message_open(&out,"system");buf_puts(&out,text);buf_puts(&out,"<|eot|>");
        free(text);free(tools);buf_free(&recipients);
    }
    for(int i=start;i<msgs->len;i++) {
        const chat_msg *m=&msgs->v[i];const char *role=m->role?m->role:"user";
        if(!strcmp(role,"assistant")) {append_muse_assistant(&out,m,i+1<msgs->len && !strcmp(msgs->v[i+1].role,"assistant"));continue;}
        if(!strcmp(role,"tool") || !strcmp(role,"function")) {
            const char *name=muse_tool_name(source,m->tool_call_id);
            buf_puts(&out,"<|start|>tool ");append_muse_header_text(&out,name);buf_puts(&out,"<|message|><tool_output name=\"");
            append_muse_attr(&out,name);buf_puts(&out,"\">\n");append_muse_text(&out,m->content);buf_puts(&out,"\n</tool_output><|eot|>");
        } else {muse_message_open(&out,role_is_system(role)?role:"user");append_muse_text(&out,m->content);buf_puts(&out,"<|eot|>");}
    }
    buf_puts(&out,ds4_think_mode_enabled(mode)?"<|start|>assistant":"<|start|>assistant to=user<|message|>");
    free(views.v);chat_msgs_free(&fragments);return buf_take(&out);
}

static char *build_muse_assistant_suffix(const request *r,const char *content,
        const char *reasoning,bool with_reasoning,const tool_calls *calls) {
    chat_msg m={.content=(char *)content,.reasoning=with_reasoning?(char *)reasoning:NULL};
    if(calls) m.calls=*calls;
    buf full={0},suffix={0};append_muse_assistant(&full,&m,false);
    const char *prefix=r && ds4_think_mode_enabled(r->think_mode)?"<|start|>assistant":"<|start|>assistant to=user<|message|>";
    size_t n=strlen(prefix);
    if(full.len>=n && !memcmp(full.ptr,prefix,n)) buf_append(&suffix,full.ptr+n,full.len-n);
    else {buf_puts(&suffix,"<|eom|>");buf_puts(&suffix,full.ptr?full.ptr:"");}
    buf_free(&full);return buf_take(&suffix);
}
