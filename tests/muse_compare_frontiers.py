# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2,<3"]
# ///
"""Compare same-checkpoint full-vocabulary ds4-bench frontier dumps."""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def compare_frontiers(baseline,candidate,tolerance,frontiers=(1024,2048,4096,8192)):
    results=[]
    for frontier in frontiers:
        filename=f'frontier_{frontier:06d}.logits.json'
        paths=[baseline/filename,candidate/filename]
        if any(not p.is_file() for p in paths):raise ValueError(f'missing frontier {frontier}')
        a,b=[json.loads(p.read_text()) for p in paths]
        for key in ('model','backend','quality','quant_bits','frontier_tokens','ctx','vocab'):
            if a[key]!=b[key]:raise ValueError(f'{frontier}: differing {key}')
        if a['frontier_tokens']!=frontier or a['vocab']<2:raise ValueError('invalid frontier metadata')
        x,y=[np.asarray(d['logits'],dtype=np.float64) for d in (a,b)]
        if x.shape!=(a['vocab'],) or y.shape!=x.shape:raise ValueError('invalid logits shape')
        if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('nonfinite logits')
        for d,logits in ((a,x),(b,y)):
            if d['argmax_id']!=int(logits.argmax()):raise ValueError('argmax metadata disagrees with logits')
        delta=x-y;maximum=float(np.abs(delta).max())
        if maximum>tolerance:raise ValueError(f'{frontier}: max error {maximum:.9g} exceeds {tolerance:.9g}')
        top=np.partition(x,-2)[-2:]
        results.append({'frontier':frontier,'max_error':maximum,'rms_error':float(np.sqrt(np.mean(delta*delta))),
                        'argmax_match':a['argmax_id']==b['argmax_id'],'baseline_top2_margin':float(top[1]-top[0])})
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline',type=Path);parser.add_argument('candidate',type=Path)
    parser.add_argument('--tolerance',type=float,default=.002)
    args=parser.parse_args()
    if not math.isfinite(args.tolerance) or args.tolerance<=0:parser.error('invalid tolerance')
    print(json.dumps({'status':'PASS','comparisons':compare_frontiers(args.baseline,args.candidate,args.tolerance)},indent=2))


if __name__=='__main__':main()
