#define main ds4_cli_program_main
#include "../ds4_cli.c"
#undef main
#include <assert.h>
extern ds4_engine *ds4_test_muse_tokenizer_engine(const char *path);
int main(int argc,char **argv) {
    assert(argc==2);ds4_engine *e=ds4_test_muse_tokenizer_engine(argv[1]);
    cli_generation_options defaults={.min_p=.05f};cli_apply_model_sampling_defaults(e,&defaults);
    assert(defaults.temperature==1 && defaults.top_p==1 && defaults.min_p==0);
    cli_generation_options explicit={.temperature=.2f,.top_p=.4f,.min_p=.7f,
        .temperature_set=true,.top_p_set=true,.min_p_set=true};
    cli_apply_model_sampling_defaults(e,&explicit);
    assert(explicit.temperature==.2f && explicit.top_p==.4f && explicit.min_p==.7f);
    const char *sample=" to=self<|message|>PRIVATE<|eom|><|start|>assistant to=user<|message|>VISIBLE<|eot|>";
    ds4_tokens tokens={0};ds4_tokenize_rendered_chat(e,sample,&tokens);
    for(int mode=0;mode<3;mode++) {
        FILE *fp=tmpfile();assert(fp);
        token_printer printer={.engine=e,.fp=fp,.format_thinking=mode==1,.raw_output=mode==2};
        for(int i=0;i<tokens.len;i++)print_generated_token(&printer,tokens.v[i]);
        token_printer_finish(&printer);assert(!printer.decode_failed);
        rewind(fp);char text[4096]={0};assert(fread(text,1,sizeof(text)-1,fp)>0);fclose(fp);
        if(mode==0)assert(!strcmp(text,"VISIBLE"));
        else if(mode==1)assert(strstr(text,"PRIVATE") && strstr(text,"VISIBLE") && !strstr(text,"<|message|>"));
        else assert(!strcmp(text,sample));
    }
    ds4_tokens prefix={0},tail={0};
    ds4_tokenize_rendered_chat(e,"<|start|>assistant to=user<|message|>prefilled",&prefix);
    ds4_tokenize_text(e,"TAIL",&tail);FILE *fp=tmpfile();assert(fp);
    token_printer printer={.engine=e,.fp=fp};assert(!token_printer_set_prefix(&printer,&prefix));
    for(int i=0;i<tail.len;i++)print_generated_token(&printer,tail.v[i]);
    rewind(fp);char text[16]={0};assert(fread(text,1,sizeof(text)-1,fp)>0 && !strcmp(text,"TAIL"));fclose(fp);
    ds4_tokens_free(&prefix);ds4_tokens_free(&tail);
    ds4_tokens_free(&tokens);ds4_engine_close(e);puts("Muse CLI native recipients and hidden reasoning: PASS");return 0;
}
