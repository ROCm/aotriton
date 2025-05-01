# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass
from typing import Callable, Any

def default_fmt(value):
    return str(value)

DTYPE_NUMBER = {
    'fp16' : 'DType::kFloat16',
    'bf16' : 'DType::kBFloat16',
    'fp32' : 'DType::kFloat32',
    'i32'  : 'DType::kInt32',
    'u32'  : 'DType::kUInt32',
    'u64'  : 'DType::kUInt64',
}

@dataclass
class TI:
    tensor_rank : int = None
    is_bool     : bool = False
    is_int      : bool = False
    nbits       : bool = False
    ctype       : str = 'UNDEFINED_CTYPE'
    format_cvalue : Callable[[Any], str] = default_fmt
    is_class    : bool = False
    alignment   : int = 0
    element_ts  : str = None

    def nbytes(self):
        return (self.nbits - 1) // 8 + 1

    @property
    def is_tensor(self):
        return self.tensor_rank is not None

    def __str__(self):
        if not self.is_tensor:
            print('WARNING: this code path should not be triggered')
            return self.ctype
        return f'"^{self.element_ts}@{self.alignment}"'

    @property
    def element_type_enum(self):
        if not self.is_tensor:
            print('WARNING: this code path should not be triggered')
            return self.ctype
        return DTYPE_NUMBER[self.element_ts]

    @property
    def infotype(self):
        if self.is_int or self.is_bool:
            return self.ctype
        if self.is_tensor:
            return 'std::string'

int8_t      = TI(is_int= True, nbits= 8, ctype='int8_t')
int16_t     = TI(is_int= True, nbits=16, ctype='int16_t')
int32_t     = TI(is_int= True, nbits=32, ctype='int32_t')
int64_t     = TI(is_int= True, nbits=64, ctype='int64_t')
uint8_t     = TI(is_int= True, nbits= 8, ctype='uint8_t')
uint16_t    = TI(is_int= True, nbits=16, ctype='uint16_t')
uint32_t    = TI(is_int= True, nbits=32, ctype='uint32_t')
uint64_t    = TI(is_int= True, nbits=64, ctype='uint64_t')
def __fmt_bool(v):
    return 'true' if v else 'false'
bool_t      = TI(is_bool= True, nbits= 1, ctype='bool', format_cvalue=__fmt_bool)
typename_t  = TI(is_class=True, ctype='typename')
float_t     = TI(nbits=32, ctype='float')

stride_a8    = TI(is_int= True, nbits=64, ctype='uint64_t', alignment=8)
stride_a16   = TI(is_int= True, nbits=64, ctype='uint64_t', alignment=16)

# Forward declaration
class ConditionalValue(ABC):
    @abstractmethod
    def get_ttype(self, rank=None):
        pass
