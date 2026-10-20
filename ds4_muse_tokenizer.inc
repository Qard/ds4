/* GPT-4o's ordered regex differs from the Qwen/GLM split in case boundaries,
 * contractions, three-digit number groups, and punctuation after newlines.
 * Unicode lookup and byte BPE remain the shared native implementation. */
#include "ds4_muse_unicode.inc"

#define MUSE_CLASS(name,cp) \
    qwen4_urange_has(ds4_muse_##name##_ranges, \
        sizeof(ds4_muse_##name##_ranges)/sizeof(ds4_muse_##name##_ranges[0]),cp)

static qwen4_char_info muse_char_at(const char *s,uint64_t len,uint64_t pos) {
    qwen4_char_info c={0};
    if(pos>=len) return c;
    c.valid=true;
    c.cp=utf8_peek_one(s,len,pos,&c.next);
    c.letter=MUSE_CLASS(letter,c.cp);
    c.mark=MUSE_CLASS(mark,c.cp);
    c.number=MUSE_CLASS(number,c.cp);
    c.space=MUSE_CLASS(space,c.cp);
    return c;
}

static uint64_t muse_contraction_end(const char *s,uint64_t len,uint64_t pos) {
    qwen4_char_info c=muse_char_at(s,len,pos);
    if(!c.valid || c.cp!='\'') return pos;
    qwen4_char_info a=muse_char_at(s,len,c.next), b=muse_char_at(s,len,a.next);
    uint32_t x=ascii_tolower_cp(a.cp),y=ascii_tolower_cp(b.cp);
    if(x==0x17f) x='s'; /* Unicode case-insensitive matching includes long s. */
    if(x=='s' || x=='t' || x=='m' || x=='d') return a.next;
    if((x=='r' && y=='e') || (x=='v' && y=='e') || (x=='l' && y=='l')) return b.next;
    return pos;
}

static uint64_t muse_word_end(const char *s,uint64_t len,uint64_t pos,bool lower_required) {
    uint64_t p=pos,last_lower=UINT64_MAX;
    while(p<len) {
        qwen4_char_info c=muse_char_at(s,len,p);
        if(!MUSE_CLASS(upper,c.cp)) break;
        if(MUSE_CLASS(lower,c.cp)) last_lower=p;
        p=c.next;
    }
    qwen4_char_info c=muse_char_at(s,len,p);
    if(lower_required) {
        /* A* B+ may need to give back a shared Lo/Lm/Mark character when
         * no lowercase-only character follows the greedy A run. */
        if(!c.valid || !MUSE_CLASS(lower,c.cp)) {
            if(last_lower==UINT64_MAX) return 0;
            p=last_lower;
        }
    } else if(p==pos) return 0;
    while(p<len) {
        c=muse_char_at(s,len,p);
        if(!MUSE_CLASS(lower,c.cp)) break;
        p=c.next;
    }
    return muse_contraction_end(s,len,p);
}

static uint64_t muse_next_piece(const char *s,uint64_t len,uint64_t pos) {
    qwen4_char_info c=muse_char_at(s,len,pos);
    if(!c.valid) return pos;
    for(unsigned alternative=0;alternative<2;alternative++) {
        uint64_t end=0;
        if(c.cp!='\r' && c.cp!='\n' && !c.letter && !c.number)
            end=muse_word_end(s,len,c.next,alternative==0);
        if(!end) end=muse_word_end(s,len,pos,alternative==0);
        if(end) return end;
    }
    if(c.number) {
        uint64_t p=pos;
        for(unsigned n=0;n<3 && p<len;n++) {
            qwen4_char_info v=muse_char_at(s,len,p);
            if(!v.number) break;
            p=v.next;
        }
        return p;
    }
    uint64_t p=c.cp==' '?c.next:pos;
    qwen4_char_info v=muse_char_at(s,len,p);
    if(v.valid && !v.space && !v.letter && !v.number) {
        while(p<len) {
            v=muse_char_at(s,len,p);
            if(v.space || v.letter || v.number) break;
            p=v.next;
        }
        while(p<len) {
            v=muse_char_at(s,len,p);
            if(v.cp!='\r' && v.cp!='\n' && v.cp!='/') break;
            p=v.next;
        }
        return p;
    }
    if(c.space) {
        uint64_t last_newline=0,last_space=pos;
        p=pos;
        while(p<len) {
            v=muse_char_at(s,len,p);
            if(!v.space) break;
            last_space=p;
            p=v.next;
            if(v.cp=='\r' || v.cp=='\n') last_newline=p;
        }
        if(last_newline) return last_newline;
        if(p<len && last_space>pos) return last_space;
        return p;
    }
    return c.next;
}

static void bpe_tokenize_text_muse(const ds4_vocab *vocab,const char *text,token_vec *out) {
    uint64_t len=strlen(text),pos=0;
    while(pos<len) {
        uint64_t end=muse_next_piece(text,len,pos);
        bpe_emit_piece(vocab,(ds4_str){text+pos,end-pos},out);
        pos=end;
    }
}
