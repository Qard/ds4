# /// script
# requires-python = ">=3.11"
# dependencies = ["tokenizers==0.23.3"]
# ///
"""Differential GPT-4o pre-tokenization against Hugging Face's regex engine."""
import random
import struct
import subprocess
import sys
import tokenizers
from tokenizers.pre_tokenizers import Split

PATTERN=(r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
         r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"
         r"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+")
reference=Split(tokenizers.Regex(PATTERN),'isolated')
cases=["Hello World", "XMLHttpRequest", "I'm HERE's", "ÉLÈVE's été", "ABC中DEF",
       "中A中A中abC", "\u0301A", "\u0301a\u0301B", "A中B", "!\n/\r\n/ ok",
       "a  b   ", "a \n \r\n  b", "1234567 ١٢٣٤٥", "ΣΑΣ Ελληνικά", "Русский ТЕКСТ",
       "中文日本語 カタカナ ひらがな", "emoji 👨‍👩‍👧‍👦 👍🏽", "'ſ's 'RE 'Ve 'LL 'M 'D 'T",
       "\u00a0\u2003\n\t  ", "!A's  A'ſ word\r\nnext", "", " ", "\n", "\r\n/"]
cases += ['<|begin_of_text|><|start|>user<|message|>Hello<|eom|><|start|>assistant to=self<|message|>',
          '<|start|>assistant to=user<|message|>Done.<|eot|><|end_of_text|>',
          '<｜begin▁of▁sentence｜><|im_start|>user<|im_end|><think>']
rng=random.Random(817)
pool=list("aAzZ0_ !\r\n/\t'strevml")+list("éÉǅ中字\u0301\u02b0ᾀΣσЖж١Ⅳ👍🏽\u2003")
for _ in range(1500):
    cases.append(''.join(rng.choice(pool) for _ in range(rng.randrange(1,90))))
for _ in range(500):
    text=[]
    for _ in range(50):
        cp=rng.randrange(1,0x110000)
        if not 0xd800<=cp<=0xdfff: text.append(chr(cp))
    cases.append(''.join(text))
request=b''.join(struct.pack('<I',len(s.encode()))+s.encode() for s in cases)
bpe=len(sys.argv)==4
result=subprocess.run([sys.argv[1]]+([sys.argv[2]] if bpe else []),input=request,stdout=subprocess.PIPE,check=True).stdout
full=tokenizers.Tokenizer.from_file(sys.argv[3]) if bpe else None
pos=0
for s in cases:
    n,=struct.unpack_from('<I',result,pos); pos+=4
    actual=list(struct.unpack_from('<'+'I'*n,result,pos)); pos+=4*n
    expected=full.encode(s,add_special_tokens=False).ids if full else [len(piece.encode()) for piece,_ in reference.pre_tokenize_str(s)]
    assert actual==expected,(repr(s),actual,expected)
assert pos==len(result)
print(f'Muse GPT-4o {"BPE" if bpe else "pre-tokenization"}: {len(cases)} differential cases PASS')
