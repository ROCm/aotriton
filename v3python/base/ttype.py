# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from numpy import np
from dataclasses import dataclass
from typing import Callable, Any

def default_fmt(value):
    return str(value)

@dataclass
class TI:
    is_tensor   : bool = False
    is_bool     : bool = False
    is_int      : bool = False
    nbits       : bool = False
    ctype       : str = ''
    format_cvalue : Callable[[Any], str] = default_fmt
    is_class    : bool = False

    def nbytes(self):
        return (self.nbits - 1) // 8 + 1

int8_t      = TI(is_tensor=False, is_bool=False, is_int= True, nbits= 8, ctype='int8_t')
int16_t     = TI(is_tensor=False, is_bool=False, is_int= True, nbits=16, ctype='int16_t')
int32_t     = TI(is_tensor=False, is_bool=False, is_int= True, nbits=32, ctype='int32_t')
int64_t     = TI(is_tensor=False, is_bool=False, is_int= True, nbits=64, ctype='int64_t')
uint8_t     = TI(is_tensor=False, is_bool=False, is_int= True, nbits= 8, ctype='uint8_t')
uint16_t    = TI(is_tensor=False, is_bool=False, is_int= True, nbits=16, ctype='uint16_t')
uint32_t    = TI(is_tensor=False, is_bool=False, is_int= True, nbits=32, ctype='uint32_t')
uint64_t    = TI(is_tensor=False, is_bool=False, is_int= True, nbits=64, ctype='uint64_t')
def __fmt_bool(v):
    return 'true' if v else 'false'
bool_t      = TI(is_tensor=False, is_bool= True, is_int=False, nbits= 1, ctype='bool', format_cvalue=__fmt_bool)
typename_t  = TI(is_class=True)

MAP_NP_TO_TTYPE = {
    np.bool   : bool_t,
    np.int8   : int8_t,
    np.int16  : int16_t,
    np.int32  : int32_t,
    np.int64  : int64_t,
    np.uint8  : uint8_t,
    np.uint16 : uint16_t,
    np.uint32 : uint32_t,
    np.uint64 : uint64_t,
    # 'fp32'    : 'float',
    # '*fp32'   : 'const float*',
    # '*fp16'   : 'const __fp16*',
    # '*bf16'   : 'const __bf16*',
    # 'i8'      : 'int8_t',
    # 'i16'     : 'int16_t',
    # 'i32'     : 'int32_t',
    # 'i64'     : 'int64_t',
    # 'u8'      : 'uint8_t',
    # 'u16'     : 'uint16_t',
    # 'u32'     : 'uint32_t',
    # 'u64'     : 'uint64_t',
}

def tensor_type(rank):
    return TI(is_tensor= True, is_bool=False, is_int=False, nbits=64, ctype=f'const T{rank}*')

def guess_vparam_type(choices):
    if isinstance(choices, np.ndarray):
        return MAP_NP_TO_TTYPE(choices.dtype.type)
    if all([isinstance(v, bool) for v in choices):
        return bool_t
    if all([isinstance(v, int) for v in choices):
        if max(choices) < 16:  # Leave with some margin
            return int8_t
        if max(choices) < 8192:
            return int16_t
        return int32_t
    assert False, f"Cannot guess valued parameter type from {choices=}"
