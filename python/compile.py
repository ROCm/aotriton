#!/usr/bin/env python

import hashlib
import importlib.util
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import triton
from triton.compiler.code_generator import kernel_suffix
from triton.compiler.make_launcher import ty_to_cpp

import shutil
import json

desc = """
Triton ahead-of-time compiler:
"""

def main():
    # command-line arguments
    parser = ArgumentParser(description=desc)
    parser.add_argument("path", help="Path to Python source containing desired kernel in its scope. File will be executed.")
    parser.add_argument("--target", type=str, default=None, help="Ahead of Time (AOT) Compile Architecture. PyTorch is required for autodetection if --target is missing.")
    parser.add_argument("--kernel_name", "-n", type=str, default="", help="Name of the kernel to compile", required=True)
    parser.add_argument("--num_warps", "-w", type=int, default=1, help="Number of warps to launch the kernel")
    parser.add_argument("--num_stages", "-ns", type=int, default=3, help="Number of stages (meta-parameter of the kernel)")
    parser.add_argument("--out_path", "-o", type=Path, default=None, help="Out filename", required=True)
    parser.add_argument("--signature", "-s", type=str, help="Signature of the kernel", required=True)
    parser.add_argument("--grid", "-g", type=str, help="Launch grid of the kernel", required=True)
    parser.add_argument("--verbose", "-v", help="Enable vebose output", action='store_true')
    args = parser.parse_args()

    out_path = args.out_path
    out_path = out_path.with_suffix('')

    # execute python sources and extract functions wrapped in JITFunction
    arg_path = Path(args.path)
    sys.path.insert(0, str(arg_path.parent))
    '''
    spec = importlib.util.spec_from_file_location(arg_path.stem, arg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kernel = getattr(mod, args.kernel_name)
    '''
    if True:
        exec_string = f'import {arg_path.stem}'
        print(exec_string)
        exec(exec_string, globals()) # importlib code path miss things
        print(globals())
        # kernel = globals()[f"{arg_path.stem}.{args.kernel_name}"]
        mod = globals()[arg_path.stem]
        kernel = getattr(mod, args.kernel_name)
        print(fused_attention_trimmed.attn_fwd)
    if False:
        mod = importlib.import_module(arg_path.stem)
        print(mod.attn_fwd)
        print(fused_attention_trimmed.attn_fwd)
        kernel = globals()[f"{arg_path.stem}.{args.kernel_name}"]
        print(f"{kernel=}")

    grid = args.grid.split(",")
    assert len(grid) == 3

    # validate and parse signature
    signature = list(map(lambda s: s.strip(" "), args.signature.split(",")))

    def hash_signature(signature: List[str]):
        m = hashlib.sha256()
        m.update(" ".join(signature).encode())
        return m.hexdigest()[:8]

    meta_sig = f"warps{args.num_warps}xstages{args.num_stages}"
    sig_hash = hash_signature(signature + [meta_sig])

    def constexpr(s):
        try:
            ret = int(s)
            return ret
        except ValueError:
            pass
        try:
            ret = float(s)
            return ret
        except ValueError:
            pass
        if s == 'True':
            return True
        if s == 'False':
            return False
        return None

    hints = {i: constexpr(s.split(":")[1]) for i, s in enumerate(signature) if ":" in s}
    hints = {k: v for k, v in hints.items() if v is not None}
    constexprs = {i: constexpr(s) for i, s in enumerate(signature)}
    constexprs = {k: v for k, v in constexprs.items() if v is not None}
    print(f"{constexprs=}")
    signature = {i: s.split(":")[0] for i, s in enumerate(signature) if i not in constexprs}
    const_sig = 'x'.join([str(v) for v in constexprs.values()])
    doc_string = [f"{kernel.arg_names[i]}={constexprs[i]}" for i in constexprs.keys()]
    doc_string += [f"num_warps={args.num_warps}", f"num_stages={args.num_stages}"]

    # compile ast into cubin
    for h in hints.values():
        assert h in [1, 16], f"Only 1 and 16 are valid hints, got {h}"
    divisible_by_16 = [i for i, h in hints.items() if h == 16]
    equal_to_1 = [i for i, h in hints.items() if h == 1]
    config = triton.compiler.instance_descriptor(divisible_by_16=divisible_by_16, equal_to_1=equal_to_1)
    for i in equal_to_1:
        constexprs.update({i: 1})
    print(f'{kernel=}')
    ccinfo = triton.compile(kernel, signature=signature, constants=constexprs, configs=[config], num_warps=args.num_warps, num_stages=args.num_stages, aot_arch=args.target)
    hsaco_path = ccinfo.asm.get('hsaco_path', None)
    if args.verbose:
        print(dir(ccinfo))
        print(f'{ccinfo.asm.keys()=}')
        print(f'{ccinfo.fn=}')
        print(f'{hsaco_path=}')

    if hsaco_path is not None:
        shutil.copy(hsaco_path, out_path.with_suffix('.hsaco'))

    with out_path.with_suffix('.json').open("w") as fp:
        json.dump(ccinfo.metadata, fp, indent=2)

if __name__ == "__main__":
    main()
