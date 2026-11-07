#!/usr/bin/env python
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Time the same kernel-level candidates of a few entries, to compare two compilers.

Two tuning builds of one arch that differ only in the Triton that compiled them
have the same candidate list, so candidate i of entry E is the same kernel
configuration in both. This script benchmarks a sample of candidates through
the tuner's own `testrun` (prepare_data, probe, benchmark) in the environment
it is started in, and `--compare` joins two such runs.

    # once per build; the environment (PYTHONPATH with the build's lib, as
    # .tune/host/node_env.sh sets it) decides which build is measured
    compiler_gate.py run --gpu 0 --label triton39 --out gate39.jsonl \
        --candidates 24 --entry "dtype='bfloat16';hdim=128;seqlen_q=64;seqlen_k=64;causal=False" ...
    compiler_gate.py compare gate38.jsonl gate39.jsonl

An entry is the text FlashEntry.parse_text takes. The kernel interfaces sampled
are the ones probe returns at kernel level (the bare, un-prefixed names), so
pass --iface to restrict to, e.g., bwd_kernel_dk_dv.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--gpu', type=int, default=0)
    r.add_argument('--label', required=True, help='free text, e.g. triton39')
    r.add_argument('--out', required=True, help='JSON lines file to append to')
    r.add_argument('--entry', action='append', required=True)
    r.add_argument('--iface', action='append', default=None,
                   help='kernel interface names to sample; default: all that probe lists')
    r.add_argument('--candidates', type=int, default=24,
                   help='candidates sampled per interface, evenly spaced over the list')
    r.add_argument('--module', default='flash')
    c = sub.add_parser('compare')
    c.add_argument('baseline')
    c.add_argument('candidate')
    return p.parse_args()


def _sample(count, wanted):
    if count <= wanted:
        return list(range(count))
    step = count / wanted
    return sorted({int(i * step) for i in range(wanted)})


def run(args):
    from aotriton.pgdef.exaid import ExaidProxy
    from aotriton.tune.registry import load_tune_module
    entry_cls = load_tune_module(args.module).TuneDesc().ENTRY_CLASS

    argv = [sys.executable, '-m', 'aotriton.tune.testrun', args.module, '--gpu', str(args.gpu)]
    proxy = ExaidProxy(lambda: argv)
    try:
        with tempfile.TemporaryDirectory(prefix='compiler_gate') as tmp, \
                open(args.out, 'a') as out:
            for text in args.entry:
                entry = entry_cls.parse_text(text)
                data = Path(tmp) / entry.as_posix()
                proxy.write('prepare_data', entry.as_text(), data.as_posix())
                proxy.readinfo(timeout=600)
                proxy.write('probe', data.as_posix(), '', 'kernel')
                probed = json.loads(proxy.readinfo(timeout=120))
                for iface, cands in probed.items():
                    if isinstance(cands, dict) or (args.iface and iface not in args.iface):
                        continue  # {'error': ...} or filtered out
                    for idx in _sample(len(cands), args.candidates):
                        proxy.write('benchmark', data.as_posix(), f'{iface}={idx}')
                        res = json.loads(proxy.readinfo(timeout=300))
                        rec = {'label': args.label, 'entry': entry.as_text(), 'iface': iface,
                               'index': idx, 'times': res.get('times'),
                               'adiffs': res.get('adiffs')}
                        out.write(json.dumps(rec) + '\n')
                        out.flush()
                proxy.write('cleanup', data.as_posix())
                proxy.readinfo(timeout=60)
    finally:
        proxy.shutdown()


def _load(path):
    table = {}
    for line in open(path):
        r = json.loads(line)
        if r['times']:
            table[(r['entry'], r['iface'], r['index'])] = r['times'][0]  # median
    return table


def compare(args):
    base, cand = _load(args.baseline), _load(args.candidate)
    rows = {}
    for key in base.keys() & cand.keys():
        rows.setdefault(key[:2], []).append(cand[key] / base[key])
    print(f'{"entry":70s} {"iface":22s} {"n":>4s} {"median":>7s} {"worst":>7s}  (candidate/baseline time)')
    for (entry, iface), ratios in sorted(rows.items()):
        ratios.sort()
        print(f'{entry:70s} {iface:22s} {len(ratios):4d} {ratios[len(ratios)//2]:7.3f} {ratios[-1]:7.3f}')


def main():
    args = parse_args()
    run(args) if args.command == 'run' else compare(args)


if __name__ == '__main__':
    main()
