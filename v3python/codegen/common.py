# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

'''
_cfields means the data type has been translated to c types
'''
def codegen_struct_cfields(cfields, *, nalign):
    print(f'{cfields=}')
    max_len = max([len(cf.ctype) for cf in cfields]) + 1
    rows = [cf.ctype + ' ' * (max_len - len(cf.ctype)) + cf.aname for cf in cfields]
    ALIGN = ';\n' + ' ' * nalign
    return ALIGN.join(rows)

class MissingLutEntry(Exception):
    def __init__(self,  functional, lut_tensor):
        self._functional = functional
        self._lut_tensor = lut_tensor

    def __repr__(self):
        return f'{self._functional.filepack_signature} has broken tuning table:\n{self._lut_tensor}'

    def get_missing_lut_entries(self) -> "list[str]":
        kdesc = self._functional.meta_object
        return kdesc.get_missing_lut_entries(self.gpu, self.lut_tensor, self.fsels)
