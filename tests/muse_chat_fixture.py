"""Production metadata with an unmerged byte vocabulary, no weight payload."""
import struct
from pathlib import Path
from muse_gguf_fixture import fixture
from qwen4_pack_to_qwen4exp import Reader, Arr, kv_bytes
import tokenizers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

def checkpoint(directory):
    directory=Path(directory);scratch=directory/'sparse.gguf';fixture(scratch)
    reader=Reader(str(scratch));metadata=[(k,reader.kv_types[k],v) for k,v in reader.kv.items()]
    reader.f.close();scratch.unlink()
    literal=list(range(33,127))+list(range(161,173))+list(range(174,256))
    alphabet={b:chr(b) for b in literal};extra=0
    for b in range(256):
        if b not in alphabet: alphabet[b]=chr(256+extra);extra+=1
    vocab=[alphabet[i] for i in range(256)]+[f'[UNUSED_{i}]' for i in range(256,202048)]
    special={200000:'<|begin_of_text|>',200001:'<|end_of_text|>',200007:'<|eom|>',
             200008:'<|eot|>',200022:'<|start|>',200023:'<|message|>'}
    for i,value in special.items(): vocab[i]=value
    metadata += [('tokenizer.ggml.pre',8,'gpt-4o'),('tokenizer.ggml.tokens',9,Arr(8,vocab)),('tokenizer.ggml.merges',9,Arr(8,[]))]
    model=directory/'metadata.gguf'
    model.write_bytes(b'GGUF'+struct.pack('<IQQ',3,0,len(metadata))+b''.join(kv_bytes(*v) for v in metadata))
    full=tokenizers.Tokenizer(BPE({value:i for i,value in enumerate(vocab)},[]))
    full.pre_tokenizer=ByteLevel(add_prefix_space=False,use_regex=False)
    full.add_special_tokens(list(special.values()))
    full.decoder=tokenizers.decoders.ByteLevel()
    tokenizer=directory/'tokenizer.json';full.save(str(tokenizer))
    return model,tokenizer
