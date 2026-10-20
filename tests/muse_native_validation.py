# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2,<3"]
# ///
"""Manual, sequential native-checkpoint validation. --plan never opens a model."""
import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np


def read_vector(path, dtype, count):
    if path.stat().st_size != count * np.dtype(dtype).itemsize:
        raise ValueError(f'{path.name}: invalid artifact length')
    return np.fromfile(path, dtype=dtype)


def compare_runs(baseline, candidate, tolerance):
    """Compare every vocabulary logit, rejecting mismatched teacher prefixes."""
    left = json.loads((baseline / 'summary.json').read_text())['cases']
    right = json.loads((candidate / 'summary.json').read_text())['cases']
    if [c['name'] for c in left] != [c['name'] for c in right]:
        raise ValueError('case lists differ')
    results = []
    for a, b in zip(left, right):
        name = a['name']
        if name not in ('raw', 'chat-none', 'chat-high'):
            raise ValueError('unknown case name')
        for key in ('vocab', 'steps', 'prompt_tokens'):
            if a[key] != b[key] or a[key] <= 0:
                raise ValueError(f'{name}: invalid or differing {key}')
        for suffix, count, label in (('ids.i32', a['steps'], 'continuation'),
                                     ('prompt.i32', a['prompt_tokens'], 'prompt')):
            x = read_vector(baseline / f'{name}.{suffix}', '<i4', count)
            y = read_vector(candidate / f'{name}.{suffix}', '<i4', count)
            if x.size != count or not np.array_equal(x, y):
                raise ValueError(f'{name}: {label} IDs differ or are truncated')
        shape = (a['steps'] + 1, a['vocab'])
        x = read_vector(baseline / f'{name}.logits.f32', '<f4', math.prod(shape))
        y = read_vector(candidate / f'{name}.logits.f32', '<f4', math.prod(shape))
        if x.size != math.prod(shape) or y.size != x.size:
            raise ValueError(f'{name}: truncated logits')
        x, y = x.reshape(shape).astype(np.float64), y.reshape(shape).astype(np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f'{name}: nonfinite logits')
        delta = x - y
        maximum = float(np.abs(delta).max())
        if maximum > tolerance:
            raise ValueError(f'{name}: logits max error {maximum:.9g} exceeds {tolerance:.9g}')
        denom = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
        cosine = np.divide((x*y).sum(axis=1), denom, out=np.zeros(shape[0]), where=denom > 0)
        cosine[(np.linalg.norm(x, axis=1) == 0) & (np.linalg.norm(y, axis=1) == 0)] = 1
        top = np.partition(x, -2, axis=1)[:, -2:]
        results.append({'name': name, 'max_error': maximum,
                        'rms_error': float(np.sqrt(np.mean(delta*delta))),
                        'minimum_cosine': float(cosine.min()),
                        'argmax_matches': int((x.argmax(axis=1) == y.argmax(axis=1)).sum()),
                        'vectors': shape[0], 'minimum_baseline_top2_margin': float((top[:, 1]-top[:, 0]).min())})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='new directory for artifacts and logs')
    parser.add_argument('--driver', type=Path, default=Path(__file__).with_name('test_muse_native_live'))
    parser.add_argument('--prefill', nargs='+', type=int, default=[1, 32])
    parser.add_argument('--cache-experts', type=int, default=512)
    parser.add_argument('--context', type=int, default=4096)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--tolerance', type=float, default=.002)
    parser.add_argument('--plan', action='store_true', help='print commands only; never open the checkpoint')
    args = parser.parse_args()
    if (len(set(args.prefill)) != len(args.prefill) or any(n < 1 or n > 8192 for n in args.prefill)
            or not 1 <= args.steps <= 16 or not 8 <= args.cache_experts <= 15872
            or not 32 <= args.context <= 1048576 or not math.isfinite(args.tolerance) or args.tolerance <= 0):
        parser.error('invalid dimensions or tolerance')
    root = args.output.resolve()
    baseline = root / f'prefill-{args.prefill[0]}'
    commands = []
    for index, chunk in enumerate(args.prefill):
        directory = root / f'prefill-{chunk}'
        command = [str(args.driver.resolve()), str(args.model.resolve()), str(directory),
                   str(chunk), str(args.cache_experts), str(args.context), str(args.steps)]
        if index:
            command.append(str(baseline))
        commands.append((directory, command))
    if args.plan:
        print(json.dumps({'sequential_commands': [cmd for _, cmd in commands]}, indent=2))
        return
    # Exclusive root creation prevents accidental reuse of incomplete/old artifacts.
    root.mkdir(parents=True, exist_ok=False)
    comparisons = []
    for directory, command in commands:
        directory.mkdir()
        print(f'Running {directory.name}; full native checkpoint, two validation sessions.', flush=True)
        with (directory / 'stdout.log').open('x') as out, (directory / 'stderr.log').open('x') as err:
            subprocess.run(command, stdout=out, stderr=err, check=True)
        comparisons.append({'run': directory.name, 'cases': compare_runs(baseline, directory, args.tolerance)})
    result = {'status': 'PASS', 'tolerance': args.tolerance, 'comparisons': comparisons}
    with (root / 'comparison.json').open('x') as fp:
        json.dump(result, fp, indent=2)
        fp.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
