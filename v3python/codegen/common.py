# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from v3python.base.typeinfo import (
    SIGNATURE_TO_C
)

'''
_cfields means the data type has been translated to c types
'''
def codegen_struct_cfields(cfields, *, nalign):
    # cfields = [(SIGNATURE_TO_C[ttype], aname) for ttype, aname in fields]
    print(f'{cfields=}')
    max_len = max([len(ctype) for ctype, aname in cfields]) + 1
    rows = [ctype + ' ' * (max_len - len(ctype)) + aname for ctype, aname in cfields]
    ALIGN = ';\n' + ' ' * nalign
    return ALIGN.join(rows)
