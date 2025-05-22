# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from v3python.utils import log

class TypedChoice(ABC):
    HIDDEN = False    # For strides
    NBITS = None
    '''
    Identity for settled TC. Only a few TC classes are unsettled.
    '''
    def resolve(self, aname, tc_dict):
        return self

    @property
    @abstractmethod
    def itype(self):
        pass

    @property
    @abstractmethod
    def triton_compile_signature(self):
        pass

    @property
    def sql_value(self):
        return self.triton_compile_signature

    def resolve_rank(self, all_names, RANKS):
        pass

    @property
    def is_tensor(self):
        return False

    def create_constexpr(self, value):
        raise RuntimeError(f"create_constexpr is unsupported in class {self.__class__}")

'''
New design: ConditionalConstexpr is subclass of TypedChoice
'''
class ConditionalChoice(TypedChoice):
    pass

class argument_base(TypedChoice):  # Will be actual argument
    ALIGNMENT = None  # Unlike NBITS this is BYTES
    @property
    def triton_compile_signature(self):
        T = self.TRITON_TYPE_PFX
        N = self.NBITS
        A = self.ALIGNMENT
        return f'{T}{N}' if A is None else f'{T}{N}:{A}'

    '''
    TODO: Reconsider the usage of __str__
    '''
    def __str__(self):
        return '"' + self.triton_compile_signature + '"'

    '''
    infotype: type for C code to store the list of choice values
    '''
    @property
    def infotype(self):
        return 'std::string'

    @property
    def infotext(self):
        return '"' + self.triton_compile_signature + '"'

    MAP_TRITON_TYPE_PFX = {
        'i'  : 'Int',
        'u'  : 'UInt',
        'fp' : 'Float',
        'bf' : 'BFloat',
    }
    @property
    def type_enum(self):
        T = self.MAP_TRITON_TYPE_PFX[self.TRITON_TYPE_PFX]
        N = self.NBITS
        return f'DType::k{T}{N}'

########################### Basic TypedChoice ###########################

# TODO: A more compact definition?
class bool_t(argument_base):
    NBITS = 1
    @property
    def itype(self):
        return 'bool'
    @property
    def triton_compile_signature(self):
        return 'u1'
class integer_base(argument_base):
    SIGNED = None
    @property
    def itype(self):
        N = self.NBITS
        return f'int{N}_t' if self.SIGNED else f'uint{N}_t'
    @property
    def sql_value(self):
        return f'torch.int{self.NBITS}' if self.SIGNED else f'torch.uint{self.NBITS}'
class sint_base(integer_base):
    SIGNED = True
    TRITON_TYPE_PFX = 'i'
class uint_base(integer_base):
    SIGNED = False
    TRITON_TYPE_PFX = 'u'
class int8_t(sint_base):
    NBITS = 8
class int16_t(sint_base):
    NBITS = 16
class int32_t(sint_base):
    NBITS = 32
class int32a16_t(int32_t):
    ALIGNMENT = 16
class int64_t(sint_base):
    NBITS = 64
class uint8_t(uint_base):
    NBITS = 8
class uint16_t(uint_base):
    NBITS = 16
class uint32_t(uint_base):
    NBITS = 32
class uint64_t(uint_base):
    NBITS = 64
class stride_a8(uint64_t):
    ALIGNMENT = 8
    HIDDEN = True
class stride_a16(uint64_t):
    ALIGNMENT = 16
    HIDDEN = True
class float_base(argument_base):
    TRITON_TYPE_PFX = 'fp'
    @property
    def itype(self):
        # Interface only pass fp32 arguments
        return 'float'
    @property
    def sql_value(self):
        return f'torch.float{self.NBITS}'
class fp16_t(float_base):
    NBITS = 16
class bf16_t(float_base):
    TRITON_TYPE_PFX = 'bf'
    NBITS = 16
class fp32_t(float_base):
    NBITS = 32
class fp16a16_t(fp16_t):
    ALIGNMENT = 16
class bf16a16_t(bf16_t):
    ALIGNMENT = 16
class fp32a16_t(fp32_t):
    ALIGNMENT = 16

########################### Const Expression ###########################

class constexpr_base(TypedChoice):
    SIGNED = None
    def __init__(self, value):
        self._value = self.pytype(value)

    def __str__(self):
        return str(self._value)

    @property
    @abstractmethod
    def pytype(self):
        pass

    # Do not define this in superclass
    # It is only meaningful for performance choices
    @property
    def json_value(self):
        return self._value if not isinstance(self._value, np.number) else self._value.item()

    @property
    def infotext(self):
        return str(self._value)

    @property
    def triton_compile_signature(self):
        return self._value

    def create_constexpr(self, value):
        assert self.__class__ != constexpr_base, 'create_constexpr cannot be called over constexpr_base'
        return self.__class__(value)

class constexpr(object):
    class bool_t(constexpr_base):
        NBITS = 1
        @property
        def pytype(self):
            return bool
        @property
        def itype(self):
            return 'bool'
        @property
        def infotype(self):
            return 'bool'
        @property
        def infotext(self):
            return 'true' if self._value else 'false'
    class integer_base(constexpr_base):
        SIGNED = None
        @property
        def pytype(self):
            return int
        @property
        def itype(self):
            N = self.NBITS
            return f'int{N}_t' if self.SIGNED else f'uint{N}_t'
        @property
        def infotype(self):
            return 'int'
    class int_base(integer_base):
        SIGNED = True
    class uint_base(integer_base):
        SIGNED = False
    class int8_t(int_base):
        NBITS = 8
    class int16_t(int_base):
        NBITS = 16
    class int32_t(int_base):
        NBITS = 32
    class int64_t(int_base):
        NBITS = 64
    class uint8_t(uint_base):
        NBITS = 8
    class uint16_t(uint_base):
        NBITS = 16
    class uint32_t(uint_base):
        NBITS = 32
    class uint64_t(uint_base):
        NBITS = 64
    class stride1(uint64_t):
        HIDDEN = True
        def __init__(self):
            self._value = 1

########################### Tensor ###########################

class tensor(argument_base):
    NBITS = 64

    def __init__(self, elem_ty : TypedChoice, rank):
        self._elem_ty = elem_ty
        self._rank = rank
        self._specialized = {}
        self.ALIGNMENT = elem_ty.ALIGNMENT

    def resolve_rank(self, all_names, RANKS):
        default_rank = RANKS['_default']
        def specialize(aname):
            rank = RANKS.get(aname, default_rank)
            return tensor(elem_ty=self._elem_ty, rank=rank)
        # print(f'resolve_rank {self=} {self._elem_ty=} {all_names=} BEFORE {self._specialized=}')
        self._specialized.update({ aname : specialize(aname) for aname in all_names })
        log(lambda : f'resolve_rank {self=} {self._elem_ty=} {all_names=} AFTER  {self._specialized=}')

    def resolve(self, aname, tc_dict):
        log(lambda : f'{self._specialized=} {aname=}')
        return self._specialized.get(aname, self)

    @property
    def itype(self):
        assert self._rank is not None
        return f'const TensorView<{self._rank}>*'

    @property
    def triton_compile_signature(self):
        return '*' + self._elem_ty.triton_compile_signature

    @property
    def type_enum(self):
        return self._elem_ty.type_enum

    @property
    def is_tensor(self):
        return True

    # SQL do not record pointer type (already inferred by column name)
    @property
    def sql_value(self):
        elem_ty = self._elem_ty
        return elem_ty.sql_value

##################### Guessing Functions #####################
class Guess(object):
    def guess(self):
        return [ self._efactory(v) for v in self._choices ]

class GuessNumpy(Guess):
    FACTORY = {
        np.bool   : constexpr.bool_t,
        np.int8   : constexpr.int8_t,
        np.int16  : constexpr.int16_t,
        np.int32  : constexpr.int32_t,
        np.int64  : constexpr.int64_t,
        np.uint8  : constexpr.uint8_t,
        np.uint16 : constexpr.uint16_t,
        np.uint32 : constexpr.uint32_t,
        np.uint64 : constexpr.uint64_t,
    }
    def __init__(self, choices):
        self._efactory = self.FACTORY[choices.dtype.type]
        self._choices = choices

class GuessBool(Guess):
    def __init__(self, choices):
        self._efactory = constexpr.bool_t
        self._choices = choices

class GuessInt(Guess):
    def __init__(self, choices):
        if max(choices) < 16:  # Leave with some margin
            self._efactory = constexpr.int8_t
        elif max(choices) < 8192:
            self._efactory = constexpr.int16_t
        else:
            self._efactory = constexpr.int32_t
        self._choices = choices

def parse_choices(choices):
    if isinstance(choices, np.ndarray):
        # factory = FACTORY[choices.dtype.type](choices)
        factory = GuessNumpy(choices)
        return factory.guess()
    if all([isinstance(v, bool) for v in choices]):
        factory = GuessBool(choices)
        return factory.guess()
    if all([isinstance(v, int) for v in choices]):
        factory = GuessInt(choices)
        return factory.guess()
    return [ parse_complex(v) for v in choices ]

'''
Note here we coined the alignment to base type rather than the pointer type,
which actually makes more sense. For example:
struct alignas(32) avx2_t { float lane[8]; };

However, be aware u64:16 in Triton means the value is multiple of 16.
Hence the ':' syntax here is dual-purposed.
'''
ELEMENTAL_TYPE_MAP = {
    'i8'      : int8_t,
    'i16'     : int16_t,
    'i32'     : int32_t,
    'i32:16'  : int32a16_t,
    'i64'     : int64_t,
    'u8'      : uint8_t,
    'u16'     : uint16_t,
    'u32'     : uint32_t,
    'u64'     : uint64_t,
    'u64:8'   : stride_a8,
    'u64:16'  : stride_a16,
    'fp16'    : fp16_t,
    'bf16'    : bf16_t,
    'fp32'    : fp32_t,
    'fp16:16' : fp16a16_t,
    'bf16:16' : bf16a16_t,
    'fp32:16' : fp32a16_t,
}

def parse_complex(v : 'str | TypedChoice'):
    if isinstance(v, TypedChoice):  # Already typed
        return v
    assert isinstance(v, str), 'Unsupported choice {v=} with class {v.__class__=}'
    log(lambda : f'{v=} {v.__class__=}')
    if v.startswith('*'):  # Tensor
        return tensor(elem_ty=ELEMENTAL_TYPE_MAP[v[1:]](), rank=None)
    return ELEMENTAL_TYPE_MAP[v]()
