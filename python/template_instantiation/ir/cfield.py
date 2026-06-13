# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

'''
For both defining the <Interface>Params class,
and defining the <Interface>Perf perf_array[];
'''
@dataclass
class cfield:
    ctype : str = ''
    aname : str = ''
    ctext : str = ''
    index : int = -1
    nbits : int = 0
