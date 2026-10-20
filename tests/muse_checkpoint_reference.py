# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2,<3", "gguf==0.19.0"]
# ///
"""Manual independent Muse prefix-logit oracle using bounded CPU tensor reads.

Uses ggml's Python quant decoder and NumPy math, never ds4 kernels. No sampling,
chat protocol or production inference runner. Close the resident model first
for an uncontended numerical comparison. Peak operator storage is bounded by
256 decoded rows; no full-model mmap or expert-bank cache is created.
"""
import argparse
import sys,json,time
from pathlib import Path
import numpy as np
from gguf import GGMLQuantizationType
from gguf.quants import dequantize
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'gguf-tools'))
from qwen4_pack_to_qwen4exp import Reader
from gguf_join import tensor_bytes

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model',required=True,type=Path)
parser.add_argument('--prompt-ids',required=True,type=Path)
parser.add_argument('--output',required=True,type=Path)
parser.add_argument('--experts-per-token',type=int,default=8)
parser.add_argument('--window',type=int,default=2048)
args=parser.parse_args()
if args.output.exists():parser.error('output must be new')
r=Reader(str(args.model));ids=np.fromfile(args.prompt_ids,dtype='<i4');T=len(ids)
# This oracle intentionally bounds prompt and operator memory. It never maps
# weight banks or keeps a model-sized CPU cache; selected rows are read afresh.
if not 1<=T<=32:parser.error('provide between 1 and 32 prompt IDs')
if not 1<=args.experts_per_token<=256 or args.window<1:parser.error('invalid routing/window size')
dim=r.tensors['token_embd.weight'][1][0];heads=dim//64
kv_heads=r.tensors['blk.0.attn_k.weight'][1][1]//64
layers=sum(name.endswith('.attn_norm.weight') for name in r.tensors)
if any(i<0 or i>=r.tensors['token_embd.weight'][1][1] for i in ids):parser.error('invalid prompt ID')
def read(name,first=0,count=None,expert=None):
    kind,dims,offset=r.tensors[name];width=dims[0]
    rows=int(np.prod(dims[1:])) if len(dims)>1 else 1
    if expert is not None:offset+=expert*tensor_bytes(kind,dims[:2]);rows=dims[1]
    if count is None:count=rows-first
    stride=tensor_bytes(kind,[width]);r.f.seek(r.data_start+offset+first*stride)
    data=r.f.read(count*stride);assert len(data)==count*stride
    a=np.frombuffer(data,dtype=np.uint8).reshape(count,stride)
    return dequantize(a,GGMLQuantizationType(kind)).reshape(count,width)
def vector(name):return read(name).ravel()
def linear(name,x,expert=None):
    dims=r.tensors[name][1];n=dims[1];out=np.empty((len(x),n),np.float32)
    for first in range(0,n,256):
        weight=read(name,first,min(256,n-first),expert)
        out[:,first:first+len(weight)]=x@weight.T
    return out

def norm(x,w=None,eps=1e-5,pre=False):
    if pre:x=x*w
    y=x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+np.float32(eps))
    return y if pre or w is None else y*w

def sigmoid(x):return 1/(1+np.exp(-x))
x=norm(np.concatenate([read('token_embd.weight',int(i),1) for i in ids]));start=time.monotonic()
for layer in range(layers):
    p=f'blk.{layer}.'
    def v(name):return vector(p+name+'.weight')
    def mm(name,a,expert=None):return linear(p+name+'.weight',a,expert)
    h=norm(x,v('attn_norm'));q=norm(mm('attn_q',h).reshape(T,heads,64),v('attn_q_norm'))
    k=norm(mm('attn_k',h).reshape(T,kv_heads,64),v('attn_k_norm'))
    val=mm('attn_v',h).reshape(T,kv_heads,64).astype(np.float16).astype(np.float32)
    sliding=layer%4!=1
    if sliding:
        a=np.arange(T,dtype=np.float32)[:,None,None]*np.float32(500000)**(-np.arange(0,64,2,dtype=np.float32)/64)
        c,s=np.cos(a),np.sin(a)
        def rotate(y):
            z=np.empty_like(y);z[...,::2]=y[...,::2]*c-y[...,1::2]*s;z[...,1::2]=y[...,::2]*s+y[...,1::2]*c;return z
        q,k=rotate(q),rotate(k)
    k=k.astype(np.float16).astype(np.float32);att=np.empty_like(q)
    for t in range(T):
        lo=max(0,t-args.window+1) if sliding else 0
        keys=np.repeat(k[lo:t+1],heads//kv_heads,axis=1);values=np.repeat(val[lo:t+1],heads//kv_heads,axis=1)
        score=np.einsum('hd,phd->hp',q[t],keys)/8
        probs=np.exp(score-score.max(axis=-1,keepdims=True));probs/=probs.sum(axis=-1,keepdims=True)
        att[t]=np.einsum('hp,phd->hd',probs,values)
    branch=mm('attn_output',norm(att).reshape(T,dim)*sigmoid(mm('attn_gate',h)))
    x=x*v('attn_resid_alpha')+norm(branch,eps=1e-8)*v('attn_resid_beta')
    h=norm(x,v('ffn_norm'));latent=norm(mm('ffn_latent_down',h),v('ffn_pre_exp_norm'))
    prob=sigmoid(mm('ffn_gate_inp',h));selected=np.argsort(-(prob+vector(p+'exp_probs_b.bias')),axis=-1,kind='stable')[:,:args.experts_per_token]
    routes=np.take_along_axis(prob,selected,axis=-1);routes/=np.maximum(routes.sum(axis=-1,keepdims=True),np.float32(6.103515625e-5))
    combined=np.zeros_like(latent);post=v('ffn_post_exp_norm')
    for expert in np.unique(selected):
        tokens,slots=np.where(selected==expert);a=latent[tokens]
        g,u=mm('ffn_gate_exps',a,int(expert)),mm('ffn_up_exps',a,int(expert))
        raw=mm('ffn_down_exps',g*sigmoid(g)*u,int(expert))
        combined[tokens]+=routes[tokens,slots,None]*norm(raw,post,eps=1e-8,pre=True)
    branch=norm(mm('ffn_latent_up',combined),v('post_ffw_norm'),eps=1e-8,pre=True)
    x=x*v('ffn_resid_alpha')+branch*v('ffn_resid_beta')
    print(json.dumps({'layer':layer,'seconds':time.monotonic()-start,'rms':float(np.sqrt(np.mean(x*x))),'experts':selected.tolist()}),flush=True)
logits=linear('output.weight',norm(x[-1:],vector('output_norm.weight')))
logits=20*np.tanh(logits*np.float32(.1767766953)/20)
assert np.isfinite(logits).all()
with args.output.open('xb') as fp:logits.astype('<f4').tofile(fp)
print('completed',time.monotonic()-start,flush=True)
