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


def checkpoint(path):
    rng=np.random.default_rng(817)
    weights={}
    embedding=rng.integers(-32,33,(32,128),dtype=np.int8)
    weights['token_embd.weight']=embedding.astype(np.float64)/32
    weights['output_norm.weight']=rng.uniform(.8,1.2,128)
    weights['output.weight']=rng.normal(0,.08,(32,128))
    for i in range(2):
        pre=f'blk.{i}.'
        for name,n in [('attn_norm',128),('attn_q_norm',64),('attn_k_norm',64),
                       ('ffn_norm',128),('post_ffw_norm',128),
                       ('ffn_pre_exp_norm',128),('ffn_post_exp_norm',128),
                       ('attn_resid_alpha',128),('attn_resid_beta',128),
                       ('ffn_resid_alpha',128),('ffn_resid_beta',128)]:
            weights[pre+name+'.weight']=rng.uniform(.6,1.4,n)
        for name,shape in [('attn_q',(128,128)),('attn_k',(64,128)),('attn_v',(64,128)),
                           ('attn_output',(128,128)),('attn_gate',(128,128)),
                           ('ffn_latent_down',(128,128)),('ffn_latent_up',(128,128)),
                           ('ffn_gate_inp',(4,128)),('ffn_gate_exps',(4,128,128)),
                           ('ffn_up_exps',(4,128,128)),('ffn_down_exps',(4,128,128))]:
            weights[pre+name+'.weight']=rng.normal(0,.075,shape)
        weights[pre+'exp_probs_b.bias']=rng.normal(0,.01,4)
    # Round the reference weights to exactly the stored values.
    weights={k:v.astype('<f4').astype(np.float64) for k,v in weights.items()}
    tensors=[]
    for name,value in weights.items():
        kind=8 if name=='token_embd.weight' else 0
        if kind==8:
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
    x=norm(w['token_embd.weight'][ids])
    for layer in range(2):
        def weight(name): return w[f'blk.{layer}.{name}.weight']
        h=norm(x,weight('attn_norm'))
        q=(h@weight('attn_q').T).reshape(10,2,64)
        k=(h@weight('attn_k').T).reshape(10,1,64)
        v=(h@weight('attn_v').T).reshape(10,1,64).astype(np.float16).astype(np.float64)
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
        attention=norm(attention).reshape(10,128)
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
        expected=reference(checkpoint(model))
        for streamed in (0,1):
            for chunk in (1,3,10):
                subprocess.run([sys.argv[1],str(model),str(output),str(streamed),str(chunk)],check=True)
                actual=np.fromfile(output,dtype='<f4').reshape(10,32)
                error=np.max(np.abs(actual-expected))
                assert error<5e-5,(streamed,chunk,error)
                assert np.array_equal(actual.argmax(axis=-1),expected.argmax(axis=-1))
                print(f'Muse complete graph streamed={streamed} chunk={chunk}: max error {error:.3g}',flush=True)
