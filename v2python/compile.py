#!/usr/bin/env python
# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import hashlib
import importlib.util
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import triton

import shutil
import subprocess
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
    parser.add_argument("--waves_per_eu", type=int, default=0)
    parser.add_argument("--out_path", "-o", type=Path, default=None, help="Out filename", required=True)
    parser.add_argument("--signature", "-s", type=str, help="Signature of the kernel", required=True)
    parser.add_argument("--grid", "-g", type=str, help="Launch grid of the kernel", required=True)
    parser.add_argument("--verbose", "-v", help="Enable vebose output", action='store_true')
    parser.add_argument("--nostrip", help="Keep debugging symbols", action='store_true')
    parser.add_argument("--use_debug_branch", help="For debugging", action='store_true')
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
        # print(exec_string)
        exec(exec_string, globals()) # importlib code path miss things
        # print(globals())
        # kernel = globals()[f"{arg_path.stem}.{args.kernel_name}"]
        mod = globals()[arg_path.stem]
        kernel = getattr(mod, args.kernel_name)
        # print(fused_attention_trimmed.attn_fwd)
    if False:
        mod = importlib.import_module(arg_path.stem)
        print(mod.attn_fwd)
        # print(fused_attention_trimmed.attn_fwd)
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
    # print(f"{constexprs=}")
    signature = {i: s.split(":")[0] for i, s in enumerate(signature) if i not in constexprs}
    const_sig = 'x'.join([str(v) for v in constexprs.values()])
    doc_string = [f"{kernel.arg_names[i]}={constexprs[i]}" for i in constexprs.keys()]
    doc_string += [f"num_warps={args.num_warps}", f"num_stages={args.num_stages}"]

    # compile ast into cubin
    for h in hints.values():
        assert h in [1, 16], f"Only 1 and 16 are valid hints, got {h}"
    divisible_by_16 = [i for i, h in hints.items() if h == 16]
    divisible_by_8 = divisible_by_16
    equal_to_1 = [i for i, h in hints.items() if h == 1]
    if args.use_debug_branch:
        divisible_by_16 = (0, 1, 2, 3, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 36, 37, 39, 40, 41, 43, 44, 45, 46)
        equal_to_1=(38, 10, 14, 18, 22)
        ids_of_folded_args=(3, 36, 38, 10, 14, 18, 22, 27, 28)
        divisible_by_8=(3, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 36, 37, 39, 40, 41, 43, 44, 45, 46)
        config = triton.compiler.instance_descriptor(divisible_by_16=divisible_by_16,
                                                     divisible_by_8=divisible_by_16,
                                                     ids_of_folded_args=ids_of_folded_args,
                                                     equal_to_1=equal_to_1)
        constexprs = {3: None, 10: 1, 14: 1, 18: 1, 22: 1, 27: None, 28: None, 36: None, 37: False, 38: 1, 39: 128, 40: 64, 41: 64, 42: True, 43: False, 44: False, 45: 0, 46: False}
        signature = {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*i8', 4: 'fp32', 5: '*fp32', 6: '*fp16', 7: 'i32', 8: 'i32', 9: 'i32', 10: 'i32', 11: 'i32', 12: 'i32', 13: 'i32', 14: 'i32', 15: 'i32', 16: 'i32', 17: 'i32', 18: 'i32', 19: 'i32', 20: 'i32', 21: 'i32', 22: 'i32', 23: 'i32', 24: 'i32', 25: 'i32', 26: 'i32', 27: '*i8', 28: '*i8', 29: 'i32', 30: 'i32', 31: 'i32', 32: 'i32', 33: 'i1', 34: 'i32', 35: 'i32', 36: '*i8'}
    else:
        config = triton.compiler.instance_descriptor(divisible_by_16=divisible_by_16,
                                                     divisible_by_8=divisible_by_16,
                                                     equal_to_1=equal_to_1)
    for i in equal_to_1:
        constexprs.update({i: 1})
    print(f"{constexprs=}")
    if args.use_debug_branch:
        ccinfo = triton.compile(kernel,
                signature=signature,
                device=0,
                constants=constexprs,
                configs=[config],
                num_warps=4,
                num_stages=1,
                waves_per_eu=3,
                slice_k_tile=0,
                matrix_instr_nonkdim=0,
                kpack=1,
                enable_warp_specialization=False,
                enable_fp_fusion=True,
                debug=None,
                device_type='hip',
                aot_arch=args.target)
    else:
        # print(f'{kernel=}')
        ccinfo = triton.compile(kernel,
                signature=signature,
                constants=constexprs,
                configs=[config],
                num_warps=args.num_warps,
                num_stages=args.num_stages,
                waves_per_eu=args.waves_per_eu,
                enable_fp_fusion=True,
                aot_arch=args.target)
    hsaco_path = ccinfo.asm.get('hsaco_path', None)
    if args.verbose:
        print(dir(ccinfo))
        print(f'{ccinfo.asm.keys()=}')
        print(f'{ccinfo.fn=}')
        print(f'{hsaco_path=}')

    if hsaco_path is not None:
        if args.nostrip:
            shutil.copy(hsaco_path, out_path.with_suffix('.hsaco'))
        else:
            subprocess.run(['/opt/rocm/llvm/bin/llvm-objcopy', '--remove-section', '.debug_*', str(hsaco_path), str(out_path.with_suffix('.hsaco'))])

    with out_path.with_suffix('.json').open("w") as fp:
        json.dump(ccinfo.metadata, fp, indent=2)

if __name__ == "__main__":
    main()
