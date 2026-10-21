# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2,<3"]
# ///
"""Independent NumPy reference for the GGUF Muse graph, with F16 cached KV.

The graph order follows llama.cpp's Muse model; deterministic nonuniform norm
weights and learned residual gains expose ordering mistakes. No real checkpoint.
"""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gguf-tools'))
from qwen4_pack_to_qwen4exp import kv_bytes,w_str


def q8_payload(value):
    blocks=value.reshape(-1,32)
    scales=(np.max(np.abs(blocks),axis=1)/127).astype('<f2')
    quant=np.clip(np.rint(blocks/scales[:,None]),-127,127).astype(np.int8)
    payload=b''.join(s.tobytes()+q.tobytes() for s,q in zip(scales,quant))
    return payload,(quant*scales.astype(np.float64)[:,None]).reshape(value.shape)


def iq2_payload(shape,rng):
    # Deliberately constructed IQ2_XXS blocks, not quantization or a call into
    # ds4's decoder. These eight fixed codebook entries exercise 8/25/43 levels;
    # sign parity and all sixteen local scales are generated independently.
    grid=np.array([[8]*8,[43]+[8]*7,[25,25]+[8]*6,[8,43]+[8]*6,
                   [43,43]+[8]*6,[25,8,25]+[8]*5,[8,25,25]+[8]*5,
                   [8,8,43]+[8]*5],dtype=np.float64)
    count=int(np.prod(shape))//256
    base=rng.uniform(.0008,.0015,count).astype('<f2')
    indices=rng.integers(0,8,(count,8,4),dtype=np.uint8)
    signs=rng.integers(0,128,(count,8,4),dtype=np.uint32)
    scales=rng.integers(0,16,(count,8),dtype=np.uint32)
    bits=signs | ((np.bitwise_count(signs)&1)<<7)
    polarity=1-2*((bits[...,None]>>np.arange(8,dtype=np.uint32))&1).astype(np.float64)
    decoded=grid[indices]*polarity
    decoded*=base.astype(np.float64)[:,None,None,None]*(scales[:,:,None,None]+.5)*.25
    payload=bytearray()
    words=scales<<28
    for j in range(4): words |= signs[:,:,j]<<(7*j)
    for row in range(count):
        payload.extend(base[row].tobytes())
        for group in range(8):
            payload.extend(indices[row,group].tobytes())
            payload.extend(struct.pack('<I',int(words[row,group])))
    return bytes(payload),decoded.reshape(shape)


def checkpoint(path,packed=False):
    rng=np.random.default_rng(817)
    weights={}
    d=256 if packed else 128
    embedding=rng.integers(-32,33,(32,d),dtype=np.int8)
    weights['token_embd.weight']=embedding.astype(np.float64)/32
    weights['output_norm.weight']=rng.uniform(.8,1.2,d)
    weights['output.weight']=rng.normal(0,.08,(32,d))
    for i in range(2):
        pre=f'blk.{i}.'
        for name,n in [('attn_norm',d),('attn_q_norm',64),('attn_k_norm',64),
                       ('ffn_norm',d),('post_ffw_norm',d),
                       ('ffn_pre_exp_norm',d),('ffn_post_exp_norm',d),
                       ('attn_resid_alpha',d),('attn_resid_beta',d),
                       ('ffn_resid_alpha',d),('ffn_resid_beta',d)]:
            weights[pre+name+'.weight']=rng.uniform(.6,1.4,n)
        for name,shape in [('attn_q',(d,d)),('attn_k',(d//2,d)),('attn_v',(d//2,d)),
                           ('attn_output',(d,d)),('attn_gate',(d,d)),
                           ('ffn_latent_down',(d,d)),('ffn_latent_up',(d,d)),
                           ('ffn_gate_inp',(4,d)),('ffn_gate_exps',(4,d,d)),
                           ('ffn_up_exps',(4,d,d)),('ffn_down_exps',(4,d,d))]:
            weights[pre+name+'.weight']=rng.normal(0,.075,shape)
        weights[pre+'exp_probs_b.bias']=rng.normal(0,.01,4)
    # Round the reference weights to exactly the stored values.
    weights={k:v.astype('<f4').astype(np.float64) for k,v in weights.items()}
    tensors=[]
    for name,value in weights.items():
        kind=8 if name=='token_embd.weight' else 0
        if packed and value.ndim==3:
            kind=16;payload,value=iq2_payload(value.shape,rng);weights[name]=value
        elif packed and value.ndim==2:
            kind=8;payload,value=q8_payload(value);weights[name]=value
        elif kind==8:
            payload=b''.join(struct.pack('<e',1/32)+row.tobytes() for row in embedding.reshape(-1,32))
        else: payload=value.astype('<f4').tobytes()
        tensors.append((name,kind,list(reversed(value.shape)),payload))
    header=b'GGUF'+struct.pack('<IQQ',3,len(tensors),1)+kv_bytes('general.architecture',8,'muse-spark')
    data=bytearray()
    for name,kind,dims,payload in tensors:
        data.extend(b'\0'*(-len(data)%32)); offset=len(data); data.extend(payload)
        header+=w_str(name)+struct.pack('<I',len(dims))+struct.pack('<'+'Q'*len(dims),*dims)+struct.pack('<IQ',kind,offset)
    path.write_bytes(header+b'\0'*(-len(header)%32)+data)
    return weights


def norm(x,w=None,eps=1e-5,input_scale=False):
    if input_scale:x=x*w
    y=x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+eps)
    return y if w is None or input_scale else y*w


def reference(w):
    ids=np.arange(1,11)
    d=w['token_embd.weight'].shape[1];heads=d//64;kv_heads=heads//2
    x=norm(w['token_embd.weight'][ids])
    for layer in range(2):
        def weight(name): return w[f'blk.{layer}.{name}.weight']
        h=norm(x,weight('attn_norm'))
        q=(h@weight('attn_q').T).reshape(10,heads,64)
        k=(h@weight('attn_k').T).reshape(10,kv_heads,64)
        v=(h@weight('attn_v').T).reshape(10,kv_heads,64).astype(np.float16).astype(np.float64)
        q=norm(q,weight('attn_q_norm')); k=norm(k,weight('attn_k_norm'))
        if layer==0:
            angles=np.arange(10)[:,None,None]*500000.**(-np.arange(0,64,2)/64)
            c,s=np.cos(angles),np.sin(angles)
            def rotate(a):
                out=np.empty_like(a)
                out[...,0::2]=a[...,0::2]*c-a[...,1::2]*s
                out[...,1::2]=a[...,0::2]*s+a[...,1::2]*c
                return out
            q,k=rotate(q),rotate(k)
        k=k.astype(np.float16).astype(np.float64)
        attention=np.zeros_like(q)
        for t in range(10):
            lo=max(0,t-3) if layer==0 else 0
            score=np.einsum('hd,phd->hp',q[t],np.repeat(k[lo:t+1],2,axis=1))/8
            p=np.exp(score-score.max(axis=-1,keepdims=True)); p/=p.sum(axis=-1,keepdims=True)
            attention[t]=np.einsum('hp,phd->hd',p,np.repeat(v[lo:t+1],2,axis=1))
        attention=norm(attention).reshape(10,d)
        gate=1/(1+np.exp(-(h@weight('attn_gate').T)))
        branch=(attention*gate)@weight('attn_output').T
        x=x*weight('attn_resid_alpha')+norm(branch,eps=1e-8)*weight('attn_resid_beta')
        h=norm(x,weight('ffn_norm'))
        latent=norm(h@weight('ffn_latent_down').T,weight('ffn_pre_exp_norm'))
        probs=1/(1+np.exp(-(h@weight('ffn_gate_inp').T)))
        selected=np.argsort(-(probs+w[f'blk.{layer}.exp_probs_b.bias']),axis=-1,kind='stable')[:,:2]
        route=np.take_along_axis(probs,selected,axis=-1); route/=np.maximum(route.sum(axis=-1,keepdims=True),6.103515625e-5)
        combined=np.zeros_like(latent)
        for t in range(10):
            for slot,e in enumerate(selected[t]):
                g=weight('ffn_gate_exps')[e]@latent[t]
                u=weight('ffn_up_exps')[e]@latent[t]
                raw=weight('ffn_down_exps')[e]@(g/(1+np.exp(-g))*u)
                combined[t]+=route[t,slot]*norm(raw,weight('ffn_post_exp_norm'),eps=1e-8,input_scale=True)
        branch=combined@weight('ffn_latent_up').T
        branch=norm(branch,weight('post_ffw_norm'),eps=1e-8,input_scale=True)
        x=x*weight('ffn_resid_alpha')+branch*weight('ffn_resid_beta')
    logits=norm(x,w['output_norm.weight'])@w['output.weight'].T
    return 20*np.tanh(logits*.1767766953/20)


if __name__=='__main__':
    with tempfile.TemporaryDirectory() as tmp:
        model=Path(tmp)/'tiny.gguf'; output=Path(tmp)/'logits.f32'
        for packed in (False,True):
            expected=reference(checkpoint(model,packed=packed))
            baseline={}
            for streamed in (0,1,2,3):
                for chunk in (1,3,10):
                    subprocess.run([sys.argv[1],str(model),str(output),str(streamed),str(chunk),str(int(packed))],check=True)
                    actual=np.fromfile(output,dtype='<f4').reshape(10,32)
                    # Resident F32 uses a different existing dot kernel. Compare
                    # exact bits between streaming budgets at the same chunk.
                    key=(bool(streamed),chunk)
                    if key not in baseline: baseline[key]=actual.copy()
                    assert np.array_equal(actual,baseline[key]),(packed,streamed,chunk,'streaming budget changed output bits')
                    error=np.max(np.abs(actual-expected))
                    assert error<5e-5,(packed,streamed,chunk,error)
                    assert np.array_equal(actual.argmax(axis=-1),expected.argmax(axis=-1))
                    print(f'Muse complete graph packed={packed} streamed={streamed} chunk={chunk}: max error {error:.3g}',flush=True)
