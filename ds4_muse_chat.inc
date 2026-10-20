/* Muse's template uses recipient headers rather than <think> tokens. Keep
 * content escaping and the system prelude shared by CLI and server renderers. */
char *ds4_muse_escape_text(const char *text) {
    if (!text) text="";
    size_t n=strlen(text),used=0;
    if(n>(SIZE_MAX-1)/2) ds4_die("Muse text is too large");
    char *out=xmalloc(n*2+1);
    for(size_t i=0;i<n;) {
        const char *replacement=NULL;size_t skip=0;
        if(!strncmp(text+i,"<|",2)) {replacement="< |";skip=2;}
        else if(!strncmp(text+i,"<image>",7)) {replacement="< image>";skip=7;}
        else if(!strncmp(text+i,"<video>",7)) {replacement="< video>";skip=7;}
        else if(!strncmp(text+i,"<audio>",7)) {replacement="< audio>";skip=7;}
        if(replacement) {size_t k=strlen(replacement);memcpy(out+used,replacement,k);used+=k;i+=skip;}
        else out[used++]=text[i++];
    }
    out[used]=0;return out;
}

static char *muse_replace(const char *text,const char *from,const char *to) {
    size_t n=strlen(text),a=strlen(from),b=strlen(to),count=0;
    for(const char *p=text;(p=strstr(p,from));p+=a) count++;
    size_t cap=n+count*(b>a?b-a:0)+1;
    char *out=xmalloc(cap),*w=out;const char *p=text,*q;
    while((q=strstr(p,from))) {size_t k=q-p;memcpy(w,p,k);w+=k;memcpy(w,to,b);w+=b;p=q+a;}
    strcpy(w,p);return out;
}

char *ds4_muse_system_prompt(const char *system,const char *tools,const char *recipients,
        ds4_think_mode mode,const char *date) {
    char today[32]={0};
    if(!date) {time_t now=time(NULL);struct tm tm;if(localtime_r(&now,&tm)) strftime(today,sizeof(today),"%Y-%m-%d",&tm);date=today;}
    char *instructions=ds4_muse_escape_text(system);
    const char *from[]={"Reasoning effort","Reasoning Effort","reasoning effort","REASONING EFFORT"};
    const char *to[]={"Reasoning strength","Reasoning Strength","reasoning strength","REASONING STRENGTH"};
    for(unsigned i=0;i<4;i++) {char *next=muse_replace(instructions,from[i],to[i]);free(instructions);instructions=next;}
    bool have_strength=false;
    for(const char *p=instructions;*p;p++) if(!strncasecmp(p,"reasoning strength:",19)) {have_strength=true;break;}
    int strength=ds4_think_mode_enabled(mode)?(mode==DS4_THINK_LOW?8:mode==DS4_THINK_MEDIUM?32:mode==DS4_THINK_MAX?512:256):8;
    if(ds4_think_mode_level(mode)>0) {int level=ds4_think_mode_level(mode);strength=level<=25?8:level<=50?32:level<=75?256:512;}
    size_t cap=strlen(instructions)+(tools?strlen(tools):0)+(recipients?strlen(recipients):0)+strlen(date)+2048;
    char *out=xmalloc(cap);size_t n=snprintf(out,cap,"You are a helpful AI assistant.\n\nKnowledge cutoff: 2026-01-04.");
#define MUSE_TEXT(...) n+=(size_t)snprintf(out+n,cap-n,__VA_ARGS__)
    if(date[0]) MUSE_TEXT("\n\nCurrent date: %s.",date);
    if(instructions[0]) MUSE_TEXT("\n\n%s",instructions);
    if(tools && tools[0]) MUSE_TEXT("\n\n%s",tools);
    if(!have_strength) MUSE_TEXT("\n\nReasoning strength: %d.",strength);
    MUSE_TEXT("\n\nUse the appropriate recipient for each message:\n\n"
              "- \"self\": private reasoning and tool planning.\n"
              "- \"commentary\": user-visible intermediate updates while the assistant will continue working, including messages sent to the user before or between tool calls.\n");
    if(tools && tools[0]) MUSE_TEXT("- \"tool\" or \"[tool name]\": the name of the tool to call.\n");
    MUSE_TEXT("- \"user\": messages that end the assistant turn, such as a completed response or clarification question that waits for the user's reply; do not use for pre-tool updates or partial responses.\n\n"
              "# Valid recipients: \"self\", \"commentary\", %s\"user\".",recipients?recipients:"");
#undef MUSE_TEXT
    free(instructions);return out;
}

static void muse_chat_open(const ds4_vocab *vocab,const char *header,token_vec *out) {
    token_vec_push(out,vocab->muse_start_id);bpe_tokenize_text(vocab,header,out);
    token_vec_push(out,vocab->muse_message_id);
}
static void muse_chat_content(const ds4_vocab *vocab,const char *text,token_vec *out) {
    char *escaped=ds4_muse_escape_text(text);bpe_tokenize_text(vocab,escaped,out);free(escaped);
}
void ds4_chat_append_tool_result(ds4_engine *e,ds4_tokens *tokens,const char *name,const char *content) {
    if(!ds4_engine_is_muse(e)) {ds4_chat_append_message(e,tokens,"tool",content);return;}
    char *header=ds4_muse_escape_text(name && name[0]?name:"tool");
    for(char *p=header;*p;p++) if(*p=='\r' || *p=='\n') *p=' ';
    size_t n=strlen(header);
    if(n>(SIZE_MAX-1)/6) ds4_die("Muse tool name is too large");
    char *attr=xmalloc(6*n+1),*w=attr;
    for(const char *p=header;*p;p++) {
        const char *entity=*p=='&'?"&amp;":*p=='<'?"&lt;":*p=='>'?"&gt;":*p=='\"'?"&quot;":*p=='\''?"&#39;":NULL;
        if(entity) {size_t k=strlen(entity);memcpy(w,entity,k);w+=k;} else *w++=*p;
    }
    *w=0;
    token_vec_push(tokens,e->vocab.muse_start_id);bpe_tokenize_text(&e->vocab,"tool ",tokens);
    bpe_tokenize_text(&e->vocab,header,tokens);token_vec_push(tokens,e->vocab.muse_message_id);
    bpe_tokenize_text(&e->vocab,"<tool_output name=\"",tokens);bpe_tokenize_text(&e->vocab,attr,tokens);
    bpe_tokenize_text(&e->vocab,"\">\n",tokens);muse_chat_content(&e->vocab,content,tokens);
    bpe_tokenize_text(&e->vocab,"\n</tool_output>",tokens);token_vec_push(tokens,e->vocab.eos_id);
    free(attr);free(header);
}
static void muse_chat_system(const ds4_vocab *vocab,const char *system,ds4_think_mode mode,token_vec *out) {
    char *text=ds4_muse_system_prompt(system,NULL,NULL,mode,NULL);
    muse_chat_open(vocab,"system",out);bpe_tokenize_text(vocab,text,out);free(text);
    token_vec_push(out,vocab->eos_id);
}
static void muse_chat_assistant_prefix(const ds4_vocab *vocab,token_vec *out) {
    token_vec_push(out,vocab->muse_start_id);
    bpe_tokenize_text(vocab,"assistant",out);
}

/* Protocol instructions from the published chat template. Schema adapters
 * supply only the metadata and function lines between these fixed sections. */
char *ds4_muse_tool_prompt(const char *metadata,const char *functions) {
    const char *prefix="In this environment you have access to a set of tools you can use to answer the user's question.\n\nYou can invoke a function by writing a \"<atem:function_calls>\" block like the following as part of your reply to the user:\n<atem:function_calls>\n<atem:invoke name=\"$FUNCTION_NAME\">\n<atem:parameter name=\"$PARAMETER_NAME\">$PARAMETER_VALUE</atem:parameter>\n...\n</atem:invoke>\n</atem:function_calls>\n\nString and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.\nHere are the functions available in JSONSchema format:\n";
    const char *suffix="\n\nHere are examples of how to call functions in the tool set. To make parallel tool calls, emit each call as its own message (one tool call per message) in consecutive messages, as shown below:\n(If the tool namespace is not specified, invoke the function directly as `example_function_name_1` rather than `example_tool_name_1.example_function_name_1`)\n\nto=example_tool_name_1.example_function_name_1\n\n<atem:function_calls>\n<atem:invoke name=\"example_tool_name_1.example_function_name_1\">\n<atem:parameter name=\"example_parameter_1\">value_1</atem:parameter>\n</atem:invoke>\n</atem:function_calls>\n\nto=example_tool_name_2.example_function_name_2\n\n<atem:function_calls>\n<atem:invoke name=\"example_tool_name_2.example_function_name_2\">\n<atem:parameter name=\"example_parameter_2\">This is the value for the second parameter\nthat can span\n\"multiple\" lines\n</atem:parameter>\n<atem:parameter name=\"example_parameter_3\">value_3</atem:parameter>\n</atem:invoke>\n</atem:function_calls>";
    if(!metadata) metadata="";if(!functions) functions="";
    size_t n=strlen(prefix)+strlen(suffix)+strlen(metadata)+strlen(functions)+64;
    char *out=xmalloc(n);
    snprintf(out,n,"%s// Tool metadata%s\n// Function schemas%s%s",prefix,metadata,functions,suffix);
    return out;
}
