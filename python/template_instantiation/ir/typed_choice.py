# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from aotriton.utils import log

class TypedChoice(ABC):
    HIDDEN = False    # For strides
    NBITS = None

    @classmethod
    def parse(cls, spec):
        """Build a settled TypedChoice from an authoring literal: a type string
        (`'*fp16:16'`, `'i32'`), a python scalar (`0`, `False`), or an already-built
        TypedChoice. The single concrete instantiation value an Axis enumerates over."""
        if isinstance(spec, TypedChoice):
            return spec
        if isinstance(spec, str):
            return parse_complex(spec)
        # python scalars (int/bool/float/np scalar) -> constexpr via the guessers
        tcs = parse_choices([spec])
        assert len(tcs) == 1
        return tcs[0]

    @property
    @abstractmethod
    def itype(self):
        pass

    @property
    @abstractmethod
    def triton_compile_signature(self):
        pass

    @property
    def testrun_entry_signature(self) -> str:
        v = self.triton_compile_signature
        return repr(v) if isinstance(v, str) else str(v)

    @property
    def sql_value(self):
        return self.triton_compile_signature

    @property
    def is_tensor(self):
        return False

    @property
    def is_constexpr(self):
        """True for a compile-time constant choice (a constexpr_base subclass).
        The ATI IR keys some codegen decisions on this (e.g. pp_arg_doc, runtime-vs-
        constexpr scalar)."""
        return False

    def with_rank(self, rank: int) -> 'TypedChoice':
        """Return a TypedChoice whose tensor is specialized to a concrete rank. In the
        ATI IR a tensor's rank lives on the owning Axis, not the dtype, so the Axis
        calls this to settle the rank just-in-time. Non-tensor choices ignore rank and
        return self (overridden by `tensor`)."""
        return self

    def create_constexpr(self, value):
        raise RuntimeError(f"create_constexpr is unsupported in class {self.__class__}")

    # Value identity (not object identity): two TypedChoices are equal iff they are
    # the same concrete class and carry the same triton compile signature. The Axis /
    # godel enumeration and the f.choices view key on this (e.g. the override path's
    # `axis.choices.index(picked[var])`). Tensors of different RANK compare equal —
    # rank is an Axis concern, not part of the choice's value.
    def __eq__(self, other):
        if not isinstance(other, TypedChoice):
            return NotImplemented
        return (self.__class__ == other.__class__ and
                self.triton_compile_signature == other.triton_compile_signature)

    def __hash__(self):
        return hash((self.__class__, str(self.triton_compile_signature)))

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
    @property
    def sql_value(self):
        return f'torch.bfloat{self.NBITS}'
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

    @property
    def is_constexpr(self):
        return True

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
        self.ALIGNMENT = elem_ty.ALIGNMENT

    def with_rank(self, rank: int) -> 'TypedChoice':
        # Preserve subclass (tensor vs lazy_tensor) and element type; settle the rank.
        return self.__class__(elem_ty=self._elem_ty, rank=rank)

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

class lazy_tensor(tensor):
    @property
    def itype(self):
        assert self._rank is not None
        return f'LazyTensorInternal<{self._rank}>*'

##################### Guessing Functions #####################
# numpy scalar type -> constexpr TypedChoice class (the C struct width). The single
# source of truth for this mapping: GuessNumpy reads it for np.array choices, and the
# perf-schema resolver (specs/tune._resolve_tcc) reads it for numpy dtype annotations.
NUMPY_TO_CONSTEXPR = {
    np.bool_  : constexpr.bool_t,   # np.bool_ (not np.bool: removed in numpy 1.20-1.26)
    np.int8   : constexpr.int8_t,
    np.int16  : constexpr.int16_t,
    np.int32  : constexpr.int32_t,
    np.int64  : constexpr.int64_t,
    np.uint8  : constexpr.uint8_t,
    np.uint16 : constexpr.uint16_t,
    np.uint32 : constexpr.uint32_t,
    np.uint64 : constexpr.uint64_t,
}

class Guess(object):
    def guess(self):
        return [ self._efactory(v) for v in self._choices ]

class GuessNumpy(Guess):
    FACTORY = NUMPY_TO_CONSTEXPR
    def __init__(self, choices):
        self._efactory = self.FACTORY[choices.dtype.type]
        self._choices = choices

class GuessBool(Guess):
    def __init__(self, choices):
        self._efactory = constexpr.bool_t
        self._choices = choices

class GuessInt(Guess):
    def __init__(self, choices):
        if max(abs(v) for v in choices) < 16:  # size on magnitude to handle negatives
            self._efactory = constexpr.int8_t
        elif max(abs(v) for v in choices) < 8192:
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

LAZY_TENSOR_PATTERN = 'LazyTensor:'
LAZY_TENSOR_LEN = len(LAZY_TENSOR_PATTERN)

def _parse_rank_suffix(v: str):
    """Strip an optional C-style rank suffix `[N]` from a type string.

    Returns (type_str_without_suffix, rank_or_None).
    Examples:
      '*fp32:16[2]' -> ('*fp32:16', 2)
      '*u64[0]'     -> ('*u64', 0)
      '*fp16:16'    -> ('*fp16:16', None)
      'i32'         -> ('i32', None)
    """
    if v.endswith(']'):
        bracket = v.rfind('[')
        if bracket != -1:
            rank_str = v[bracket+1:-1]
            if rank_str.isdigit():
                return v[:bracket], int(rank_str)
    return v, None


def parse_complex(v : 'str | TypedChoice'):
    if isinstance(v, TypedChoice):  # Already typed
        return v
    assert isinstance(v, str), 'Unsupported choice {v=} with class {v.__class__=}'
    log(lambda : f'{v=} {v.__class__=}')
    if v.startswith('*'):  # Tensor
        v, rank = _parse_rank_suffix(v)
        tc = tensor(elem_ty=ELEMENTAL_TYPE_MAP[v[1:]](), rank=rank)
        return tc
    if v.startswith(LAZY_TENSOR_PATTERN):
        v, rank = _parse_rank_suffix(v)
        etype = v[LAZY_TENSOR_LEN+1:]
        tc = lazy_tensor(elem_ty=ELEMENTAL_TYPE_MAP[etype](), rank=rank)
        return tc
    return ELEMENTAL_TYPE_MAP[v]()
