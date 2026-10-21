#!/usr/bin/env python3
"""Prepare inputs and a runbook. Never execute a binary or open the model."""
import argparse
import hashlib
import json
import shlex
from pathlib import Path


def prepare(output,repo,model):
    output=output.resolve();repo=repo.resolve();model=model.expanduser().resolve()
    output.mkdir(parents=True,exist_ok=False)
    prompt=''.join(f'Archive {i}: apple banana orange river mountain copper silver garden.\n' for i in range(6000))
    (output/'benchmark.txt').write_text(prompt)
    plan={'model':str(model),'repository':str(repo),
          'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
          'native':['uv','run','--script',str(repo/'tests/muse_native_validation.py'),
                    '--model',str(model),'--output',str(output/'native')],
          'benchmarks':[]}
    for name,chunk,cache in [('cache512-chunk32',32,512),('auto-chunk512',512,None),('auto-chunk128',128,None)]:
        directory=output/name;directory.mkdir();(directory/'logits').mkdir()
        command=['env','-u','DS4_METAL_MEMORY_REPORT','DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1',str(repo/'ds4-bench'),'--model',str(model),'--metal','--ssd-streaming',
                 '--prefill-chunk',str(chunk),'--prompt-file',str(output/'benchmark.txt'),
                 '--ctx-start','1024','--ctx-max','8192','--ctx-alloc','8257','--step-mul','2',
                 '--gen-tokens','64','--teacher-forced-decode',
                 '--csv',str(directory/'speed.csv'),'--dump-frontier-logits-dir',str(directory/'logits')]
        if cache:command+=['--ssd-streaming-cache-experts',str(cache)]
        plan['benchmarks'].append({'name':name,'command':command})
    plan['compare_benchmarks']=[['uv','run','--script',str(repo/'tests/muse_compare_frontiers.py'),
        str(output/'auto-chunk512/logits'),str(output/name/'logits')]
        for name in ('cache512-chunk32','auto-chunk128')]
    server_dir=output/'server';server_dir.mkdir();(server_dir/'kv').mkdir()
    plan['server']=[str(repo/'ds4-server'),'--model',str(model),'--metal','--ssd-streaming',
        '--ctx','8192','--prefill-chunk','512','--batched-session','1',
        '--host','127.0.0.1','--port','8765','--kv-disk-dir',str(server_dir/'kv'),
        '--kv-disk-space-mb','8192','--kv-cache-min-tokens','1',
        '--kv-cache-boundary-trim-tokens','0','--kv-cache-boundary-align-tokens','1',
        '--trace',str(server_dir/'before-restart.trace')]
    plan['server_restart']=plan['server'][:-1]+[str(server_dir/'after-restart.trace')]
    api=['python3',str(repo/'tests/muse_live_api.py'),'--url','http://127.0.0.1:8765']
    plan['api']=api+['--output',str(output/'api'),'--with-reasoning']
    plan['api_multiple_tools']=['python3',str(repo/'tests/muse_live_tools.py'),'--url','http://127.0.0.1:8765',
                                '--output',str(output/'api-multiple-tools')]
    plan['api_prime']=api+['--output',str(output/'api-prime'),'--api','openai']
    plan['api_restored']=api+['--output',str(output/'api-restored'),'--api','openai','--require-initial-cache']
    agent=['uv','run','--with','pyte','--with','wcwidth','python',str(repo/'tests/test_agent_compaction.py'),
        '--binary',str(repo/'ds4-agent'),'--model',str(model),'--model-family','muse',
        '--ssd-streaming','--ctx','8192','--prefill-chunk','512','--timeout','3600']
    plan['agent_interrupt']=agent+['--interrupt-only','--output',str(output/'agent-interrupt')]
    cache_agent=agent.copy();cache_agent[cache_agent.index('--ctx')+1]='4096'
    plan['agent_cache']=cache_agent+['--compaction-only','--require-compaction-cache','--output',str(output/'agent-cache')]
    plan['agent_compaction']=agent+['--require-compaction-cache','--output',str(output/'agent-compaction')]
    plan['agent_reasoning']=agent+['--think','--temperature','1','--tokens','4096','--require-compaction-cache','--output',str(output/'agent-reasoning')]
    (output/'commands.json').write_text(json.dumps(plan,indent=2)+'\n')
    lines=['# Prepared native Muse validation','',
        'These commands open the full checkpoint. Run only after the LM Studio hold is released.',
        'Run from the repository. Run numerical, benchmark, server, and agent stages sequentially.',
        'Only the API client runs alongside its one server process. Output directories must remain new.',
        'Commands are recorded here; this preparation did not execute any of them.','',
        '## Numerical sessions','', '```sh',shlex.join(plan['native']),'```','',
        '## Context, cache and throughput sweeps','',
        'Frontiers 1024, 2048, 4096 and 8192 exercise the 2048-token sliding window.',
        'Teacher-forced continuations keep all benchmark prefixes identical. The auto-cache runs use current admission.',
        'Retain CSV, full frontier logits, stderr memory/cache reports, and elapsed wall time.',
        'The existing Metal summary reports tensor allocation peaks and final expert-cache/I/O counters.',
        'Its cleanup-time macOS footprint is not the peak working set; use the live-session diagnostic and',
        'inspect memory pressure during sustained runs before declaring that the model fits.','']
    for bench in plan['benchmarks']:
        lines+=['```sh',shlex.join(bench['command'])+' 2> '+shlex.quote(str(output/bench['name']/'stderr.log')),'```','']
    lines+=['Compare frontier logits across cache/chunk settings; investigate any drift before accepting a speedup.',
        '', '```sh',*(shlex.join(cmd) for cmd in plan['compare_benchmarks']),'```',
        '', '## Live server and disk restore','',
        'Start this foreground server in a terminal. Wait for it to finish loading before starting clients.','',
        '```sh',shlex.join(plan['server']),'```','',
        'In another terminal, run the full API suite, the two-call JSON/SSE probe, then prime the short OpenAI prefix:','',
        '```sh',shlex.join(plan['api']),shlex.join(plan['api_multiple_tools']),shlex.join(plan['api_prime']),'```','',
        'Stop this server with Ctrl-C. Restart with the same KV directory and a fresh trace file:',
        '', '```sh',shlex.join(plan['server_restart']),'```','',
        'The next client requires a cache hit on its very first request, before any in-process priming:','',
        '```sh',shlex.join(plan['api_restored']),'```','',
        'Stop the server before the next stage.','', '## Agent interruption, tools, compaction and restore','',
        'Each test starts one ds4-agent in an isolated temporary HOME/project below its output directory.',
        'The agent is explicitly instructed to create test files and compile generated C in that project.',
        'Run one command at a time. The reasoning variant repeats the compaction/recovery checks.','', '```sh',
        shlex.join(plan['agent_interrupt']),shlex.join(plan['agent_cache']),shlex.join(plan['agent_compaction']),shlex.join(plan['agent_reasoning']),
        '```','', '## Acceptance','',
        'Passing these stages is evidence, not an automatic production-ready declaration. Review numerical errors,',
        'coherent output, reasoning/tool boundaries, cache restoration, working-set pressure and useful throughput.',
        'The independent short-prefix diagnostic is not an exact golden-logit pass; review its per-layer',
        'routing diagnosis. Native chunk/cache comparisons still have to meet their strict tolerance.','']
    (output/'RUNBOOK.md').write_text('\n'.join(lines))
    return plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--model',type=Path,default=Path.home()/'.ds4/models/muse-spark-open-native.gguf')
    args=parser.parse_args();prepare(args.output,args.repo,args.model)
    print(args.output.resolve()/'RUNBOOK.md')


if __name__=='__main__':main()
