import binascii
import hashlib
import importlib.util
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import triton
from triton.compiler.code_generator import kernel_suffix
from triton.backends.amd.driver import ty_to_cpp

import shutil
import subprocess
import json

desc = """
Triton ahead-of-time compiler:
"""

from triton.backends.compiler import GPUTarget

KNOWN_TARGETS = {
        None     : None,
        'MI200'  : GPUTarget('hip', 'gfx90a', 64),
        'MI300X' : GPUTarget('hip', 'gfx942', 64),
}

def main():
    # command-line arguments
    parser = ArgumentParser(description=desc)
    parser.add_argument("path",
                        help="Path to Python source containing desired kernel in its scope. File will be executed.")
    parser.add_argument("--target", type=str, default=None,
                        choices=list(KNOWN_TARGETS.keys()),
                        help="Ahead of Time (AOT) Compile Architecture. PyTorch is required for autodetection if --target is missing.")
    parser.add_argument("--kernel_name", "-n", type=str, default="", help="Name of the kernel to compile",
                        required=True)
    parser.add_argument("--num_warps", "-w", type=int, default=1, help="Number of warps to launch the kernel")
    parser.add_argument("--num_stages", "-ns", type=int, default=3,
                        help="Number of stages (meta-parameter of the kernel)")
    parser.add_argument("--waves_per_eu", type=int, default=0)
    parser.add_argument("--out_name", "-on", type=str, default=None, help="Out name for the compiled kernel")
    parser.add_argument("--out_path", "-o", type=Path, default=None, help="Out filename")
    parser.add_argument("--signature", "-s", type=str, help="Signature of the kernel", required=True)
    parser.add_argument("--grid", "-g", type=str, help="Launch grid of the kernel", required=True)
    parser.add_argument("--verbose", "-v", help="Enable vebose output", action='store_true')
    parser.add_argument("--nostrip", help="Keep debugging symbols", action='store_true')
    args = parser.parse_args()

    out_name = args.out_name if args.out_name else args.kernel_name
    out_path = args.out_path if args.out_path else Path(out_name)

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
    constants = {i: constexpr(s) for i, s in enumerate(signature)}
    constants = {k: v for k, v in constants.items() if v is not None}
    # print(f"{constexprs=}")
    signature = {i: s.split(":")[0] for i, s in enumerate(signature) if i not in constants}
    const_sig = 'x'.join([str(v) for v in constants.values()])
    doc_string = [f"{kernel.arg_names[i]}={constants[i]}" for i in constants.keys()]
    doc_string += [f"num_warps={args.num_warps}", f"num_stages={args.num_stages}"]

    # compile ast into cubin
    for h in hints.values():
        assert h in [1, 16], f"Only 1 and 16 are valid hints, got {h}"
    divisible_by_16 = [i for i, h in hints.items() if h == 16]
    equal_to_1 = [i for i, h in hints.items() if h == 1]
    attrs = triton.compiler.AttrsDescriptor(divisible_by_16=divisible_by_16, equal_to_1=equal_to_1)
    for i in equal_to_1:
        constants.update({i: 1})
    src = triton.compiler.ASTSource(fn=kernel, constants=constants, signature=signature, attrs=attrs)
    opts = {"num_warps": args.num_warps, "num_stages": args.num_stages}
    ccinfo = triton.compile(src, target=KNOWN_TARGETS[args.target], options=opts)
    # import pdb; pdb.set_trace()
    with open(out_path.with_suffix('.hsaco'), 'bw') as f:
        f.write(ccinfo.kernel)
    with open(out_path.with_suffix('.json'), 'w') as f:
        di = ccinfo.metadata._asdict()
        del di['target']
        json.dump(di, f, indent=2)

    '''
    arg_names = []
    arg_types = []
    for i in signature.keys():
        if i not in equal_to_1:
            arg_names += [kernel.arg_names[i]]
            arg_types += [signature[i]]

    # dump C stub code
    suffix = kernel_suffix(signature.values(), attrs)
    func_name = '_'.join([out_name, sig_hash, suffix])
    hex_ = str(binascii.hexlify(ccinfo.asm["cubin"]))[2:-1]
    params = {
        "kernel_name": func_name,
        "triton_kernel_name": args.kernel_name,
        "bin_size": len(hex_),
        "bin_data": ", ".join([f"0x{x}{y}" for x, y in zip(hex_[::2], hex_[1::2])]),
        "signature": ", ".join([f"{ty_to_cpp(ty)} {name}" for name, ty in zip(arg_names, arg_types)]),
        "full_signature": ", ".join([f"{ty_to_cpp(signature[i])} {kernel.arg_names[i]}" for i in signature.keys()]),
        "arg_pointers": ", ".join([f"&{arg}" for arg in arg_names]),
        "num_args": len(arg_names),
        "kernel_docstring": doc_string,
        "shared": ccinfo.metadata.shared,
        "num_warps": args.num_warps,
        "algo_info": '_'.join([const_sig, meta_sig]),
        "gridX": grid[0],
        "gridY": grid[1],
        "gridZ": grid[2],
        "_placeholder": "",
    }
    for ext in ['h', 'c']:
        template_path = Path(__file__).parent / f"compile.{ext}"
        with out_path.with_suffix(f".{sig_hash}_{suffix}.{ext}").open("w") as fp:
            fp.write(Path(template_path).read_text().format(**params))
    '''

if __name__ == "__main__":
    main()
