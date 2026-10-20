#include "../ds4.c"
#include <assert.h>
int main(int argc,char **argv) {
    assert(argc==5);
    return ds4_dump_chat_tokenization(argv[1],argv[3],argv[4],atoi(argv[2]),32768,stdout);
}
