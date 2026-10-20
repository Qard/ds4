#ifndef DS4_TOOL_TEXT_H
#define DS4_TOOL_TEXT_H

#include <stdbool.h>
#include <ctype.h>
#include <stddef.h>
#include <string.h>

/* Start after the tag name. Return a borrowed, unescaped attribute span only
 * after validating the whole attribute list. A key inside another key or a
 * quoted value must not become a tool name; duplicate requested keys fail. */
static inline bool ds4_tool_attr_span(const char *p,const char *name,
                                      const char **value,size_t *value_len) {
    const char *found=NULL;size_t length=0,wanted=strlen(name);
    while(*p) {
        while(isspace((unsigned char)*p)) p++;
        if(*p=='>') {
            if(p[1] || !found) return false;
            *value=found;*value_len=length;return true;
        }
        const char *key=p;
        while(*p && !isspace((unsigned char)*p) && *p!='=' && *p!='>') p++;
        size_t key_len=p-key;
        while(isspace((unsigned char)*p)) p++;
        if(!key_len || *p++!='=') return false;
        while(isspace((unsigned char)*p)) p++;
        char quote=*p++;
        if(quote!='\"' && quote!='\'') return false;
        const char *end=strchr(p,quote);
        if(!end) return false;
        if(key_len==wanted && !memcmp(key,name,wanted)) {
            if(found) return false;
            found=p;length=end-p;
        }
        p=end+1;
        if(*p!='>' && !isspace((unsigned char)*p)) return false;
    }
    return false;
}

/* Tool bodies are not HTML. Escape only their own closing delimiter and
 * spellings that would otherwise collide with that escape. */
static inline bool ds4_tool_text_escaped_close(const char *s, const char *end) {
    if (*s != '&') return false;
    s++;
    while (!strncmp(s, "amp;", 4)) s += 4;
    return !strncmp(s, "lt;", 3) && !strncmp(s + 3, end + 1, strlen(end + 1));
}

static inline bool ds4_tool_text_needs_escape(const char *s, const char *end) {
    return !strncmp(s, end, strlen(end)) || ds4_tool_text_escaped_close(s, end);
}

static inline void ds4_tool_text_unescape(char *s, const char *end) {
    char *out = s;
    while (*s) {
        if (ds4_tool_text_escaped_close(s, end)) {
            if (!strncmp(s, "&amp;", 5)) {
                *out++ = '&';
                s += 5;
            } else {
                *out++ = '<';
                s += 4;
            }
        } else {
            *out++ = *s++;
        }
    }
    *out = '\0';
}

/* A streaming fragment must not split an escape that could become a closing
 * delimiter. Ordinary entities pass through unchanged. */
static inline bool ds4_tool_text_partial_escape(const char *s, size_t n,
                                                const char *end) {
    if (!n || *s != '&') return false;
    s++;
    n--;
    while (n >= 4 && !memcmp(s, "amp;", 4)) {
        s += 4;
        n -= 4;
    }
    if (n < 4 && !memcmp(s, "amp;", n)) return true;
    if (n < 3) return !memcmp(s, "lt;", n);
    if (memcmp(s, "lt;", 3)) return false;
    s += 3;
    n -= 3;
    size_t tail = strlen(end + 1);
    return n < tail && !memcmp(s, end + 1, n);
}

#endif
