# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import hashlib
from dataclasses import dataclass
from aotriton.utils import log
from pathlib import Path


@dataclass(frozen=True)
class LaunchArg:
    """One entry of a kernel's C++ launch-argument vector. Produced by
    KernelDescription.iter_launch_arguments() and consumed when building the
    prepare_arguments() function. `kind` selects the access expression form:

      'tensor_ptr'    -> params.<aname>->kparam_data_ptr()
      'tensor_stride' -> params.<tensor>->kparam_stride(<dim>)
      'scalar'        -> CAST(&params.<aname>)

    `expr` is the fully-rendered C++ expression; `aname` is the kernel argument
    name (used for the trailing comment + per-functional constexpr lookup)."""
    aname: str
    kind: str
    expr: str

def hsaco_ondisk_name(kdesc: 'KernelDescription', ksig: 'KernelSignature') -> str:
    digest = hashlib.sha256(ksig.hsaco_entry_name.encode()).hexdigest()
    return kdesc.NAME + '-' + digest + '.hsaco'

def hsaco_inaks2_name(kdesc: 'KernelDescription', ksig: 'KernelSignature') -> str:
    return ksig.hsaco_entry_name

def hsaco_dir(build_dir : Path, k : 'KernelDescription'):
    return build_dir / k.FAMILY / f'gpu_kernel_image.{k.NAME}'

def tunecc_ondisk_name(f: 'Functional') -> str:
    digest = hashlib.sha256(f.tunecc_signature.encode()).hexdigest()
    return f.name + '-' + digest + '.cc'

'''
_cfields means the data type has been translated to c types
'''
def codegen_struct_cfields(cfields, *, nalign):
    log(lambda : f'{cfields=}')
    max_len = max([len(cf.ctype) for cf in cfields]) + 1
    rows = [cf.ctype + ' ' * (max_len - len(cf.ctype)) + cf.aname for cf in cfields]
    ALIGN = ';\n' + ' ' * nalign
    return ALIGN.join(rows)

def codegen_includes(header_files):
    includes = [f'#include "{fn}"' for fn in sorted(set(header_files))]
    return '\n'.join(includes)

class MissingLutEntry(Exception):
    def __init__(self,  functional, lut_tensor):
        self._functional = functional
        self._lut_tensor = lut_tensor

    def __repr__(self):
        return f'{self._functional.tunecc_signature} has broken tuning table:\n{self._lut_tensor}'
