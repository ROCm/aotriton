# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from v3python.utils import log
from pathlib import Path

def hsaco_filename(kdesc : 'KernelDescription',
                   ksig : 'KernelSignature'):
    return kdesc.NAME + '-Sig-' + ksig.full_compact_signature + '.hsaco'

def hsaco_dir(build_dir : Path, k : 'KernelDescription'):
    return build_dir / k.FAMILY / f'gpu_kernel_image.{k.NAME}'

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
    includes = [f'#include "{fn}"' for fn in set(header_files)]
    return '\n'.join(includes)

class MissingLutEntry(Exception):
    def __init__(self,  functional, lut_tensor):
        self._functional = functional
        self._lut_tensor = lut_tensor

    def __repr__(self):
        return f'{self._functional.tunecc_signature} has broken tuning table:\n{self._lut_tensor}'

    def get_missing_lut_entries(self) -> "list[str]":
        kdesc = self._functional.meta_object
        return kdesc.get_missing_lut_entries(self._lut_tensor, self._functional)
