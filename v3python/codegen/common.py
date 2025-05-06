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
