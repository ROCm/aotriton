#!/usr/bin/env python3

import argparse
import sys
import re
import json
from abc import abstractmethod
import subprocess

SEP = re.compile(r'[ :\[\]]')
# Example: ['FAILED', 'test/test_backward.py', '', 'test_irregulars', 'Split-BiasOn-True-l1-dtype1-0.0-CausalOff-41-257-hdim32-5-3', '']
INDICES = (1, 3, 4)
NATIVE_ARCH = subprocess.check_output(['rocm_agent_enumerator -name'], shell=True).decode('utf8', errors='ignore').strip().split(':')[0]

DTYPE = {
    'dtype0' : 'float16',
    'dtype1' : 'bfloat16',
    'dtype2' : 'float32',
}

def validate_cfg(out_str):
    if not out_str.endswith('.cfg'):
        raise argparse.ArgumentTypeError(f"Invalid --out file name: '{out_str}'. Must ends with .cfg")
    return out_str

class Translator(object):
    SEP = re.compile(r'-')
    INDICES = None

    def __call__(self, utparam):
        parts = re.split(self.SEP, utparam)
        if self.INDICES is not None:
            parts = [ parts[i] for i in self.INDICES ]
        return self.parts2cfg(parts)

    @abstractmethod
    def parts2cfg(self, parts):
        pass

class Irregulars(Translator):
    def parts2cfg(self, parts):
        _bwd, bias, storage, _sm_scale, dtype, dropout_p, causal, seqlen_k, seqlen_q, hdim, nheads, batch = parts
        return {
            "causal_type" : 0 if causal == 'CausalOff' else 1,
            "d_head" : int(hdim.removeprefix('hdim')),
            "dropout_p" : float(dropout_p),
            "dtype" : DTYPE[dtype],
            "bias_type" : 1 if bias == 'BiasOn' else 0,
            "seqlen_q" : int(seqlen_q),
            "seqlen_k" : int(seqlen_k),
            "nheads" : int(nheads),     # unread by tune_flash.py, but no harm as well
            "batch" : int(batch),       # unread by tune_flash.py, but no harm as well
            # "storage_flip": None if storage == 'False' else (1, 2),
        }

class Regulars(Translator):
    def parts2cfg(self, parts):
        bias = 'BiasOff'
        _bwd, storage, _sm_scale, dtype, dropout_p, causal, seqlen_k, seqlen_q, hdim, nheads, batch = parts
        return {
            "causal_type" : 0 if causal == 'CausalOff' else 1,
            "d_head" : int(hdim.removeprefix('hdim')),
            "dropout_p" : float(dropout_p),
            "dtype" : DTYPE[dtype],
            "bias_type" : 1 if bias == 'BiasOn' else 0,
            "seqlen_q" : int(seqlen_q),
            "seqlen_k" : int(seqlen_k),
            "nheads" : int(nheads),     # unread by tune_flash.py, but no harm as well
            "batch" : int(batch),       # unread by tune_flash.py, but no harm as well
            # "storage_flip": None if storage == 'False' else (1, 2),
        }

class RegularBias(Translator):
    def parts2cfg(self, parts):
        bias = 'BiasOn'
        causal = 'CausalOff'
        _bwd, storage, _sm_scale, dtype, dropout_p, seqlen_k, seqlen_q, hdim, nheads, batch = parts
        return {
            "causal_type" : 0 if causal == 'CausalOff' else 1,
            "d_head" : int(hdim.removeprefix('hdim')),
            "dropout_p" : float(dropout_p),
            "dtype" : DTYPE[dtype],
            "bias_type" : 1 if bias == 'BiasOn' else 0,
            "seqlen_q" : int(seqlen_q),
            "seqlen_k" : int(seqlen_k),
            "nheads" : int(nheads),     # unread by tune_flash.py, but no harm as well
            "batch" : int(batch),       # unread by tune_flash.py, but no harm as well
            # "storage_flip": None if storage == 'False' else (1, 2),
        }

UT2TR = {
    'test_irregulars' : Irregulars(),
    'test_regular_bwd' : Regulars(),
    'test_op_bwd_with_matrix_bias' : RegularBias(),
}

def parse():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('pytest_outs', help='PLAIN Standard output of pytest. Do NOT use --color=yes', nargs='+')
    p.add_argument('--out', help='Output .cfg file for tune_flash.py --entry_from_json', type=validate_cfg, default=None)
    p.add_argument('--arch', help='arch field in .cfg. Guess from current GPU by default since this tool should be run on the testing machine.', default=NATIVE_ARCH)
    args = p.parse_args()
    # print(args)
    return args

def parse_pytest_out(args, f, out):
    for line in f:
        if not line.startswith('FAILED'):
            continue
        parts = re.split(SEP, line)
        utfn, utname, utparam = [ parts[i] for i in INDICES ]
        tr2cfg = {"arch": args.arch}
        tr2cfg.update(UT2TR[utname](utparam))
        print(json.dumps(tr2cfg), file=out)

def main():
    args = parse()
    def loop(out):
        for fn in args.pytest_outs:
            with open(fn) as f:
                parse_pytest_out(args, f, out)
    if args.out is None:
        loop(sys.stdout)
    else:
        with open(args.out, 'w') as out:
            loop(out)

if __name__ == '__main__':
    main()
