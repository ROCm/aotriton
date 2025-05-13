# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import numpy as np
from enum import Enum
from .object_desc import ObjectFileDescription
from .conditional_value import ConditionalConstexpr
from copy import deepcopy

'''
Note: we category the Triton kernel arguments into three types.
'''
class ArgumentCategory(Enum):
    CAT_TYPE = 1  # This argument may have differnet types
    CAT_FEAT = 2  # This argument enable/disable a specific feature
    CAT_PERF = 3  # This argument is for performance tuning

class ArgumentMetadata(object):
    DTYPE_NUMBER = {
        'fp16' : 'DType::kFloat16',
        'bf16' : 'DType::kBFloat16',
        'fp32' : 'DType::kFloat32',
        'i32'  : 'DType::kInt32',
        'u32'  : 'DType::kUInt32',
        'u64'  : 'DType::kUInt64',
    }
    def __init__(self, grouped_arguments_as_set, possible_values, cat : ArgumentCategory, kdesc):
        assert grouped_arguments_as_set
        self._grouped_arguments_as_set = grouped_arguments_as_set
        for v in possible_values:
            if callable(v) and not isinstance(v, ConditionalConstexpr):
                assert 'return' in v.__annotations__, f'Callable PERF_CHOICES {grouped_arguments_as_set} in Class {kdesc} must have return type annotations'
        self._possible_values = possible_values
        self._npossible = len(possible_values)
        self._cat = cat
        self._godel_number = None
        self._kdesc = kdesc
        self._incomplete_tuning = False
        self._fallback_tuning_value = None
        self._ordered_arguments = None  # list[tuple[aname : str, aindex : int]]

    def sort_arguments(self, ALL_ARGUMENTS):
        arguments_tuple = [(aname, ALL_ARGUMENTS.index(aname)) for aname in self._grouped_arguments_as_set]
        self._ordered_arguments = sorted(arguments_tuple, key=lambda at: at[1])
        self._first_apperance = self._ordered_arguments[0][1]

    @property
    def first_apperance(self):
        assert hasattr(self, '_first_apperance'), 'ArgumentMetadata: must call sort_arguments before first_apperance'
        return self._first_apperance

    @property
    def ordered_argument_places(self):
        assert hasattr(self, '_ordered_arguments'), 'ArgumentMetadata: must call sort_arguments before ordered_argument_places'
        return [a[1] for a in self._ordered_arguments]

    @property
    def godel_number(self):
        assert hasattr(self, '_godel_number'), 'ArgumentMetadata: must call assign_godel_number on all ArgumentMetadata objects before using godel_number property'
        return self._godel_number

    @staticmethod
    def assign_godel_number(all_metadata: 'list[ArgumentMetadata]'):
        sorted_metadata = sorted(all_metadata, key=lambda m: m.first_apperance)
        for i, m in enumerate(sorted_metadata):
            m._order_among_all_choices = i
        # No need to trim m.nchoices == 1 because they have no impact on the assignment
        # (Analog: size 1 dimensions in Tensor)
        # However we need to append 1 as the last "stride"
        nchoices = np.array([m.nchoices for m in sorted_metadata] + [1])
        # Big Endian, because types are usually put at first and they should be
        # considered as more important
        # The first stride is trimmed off to ensure last stride is 1
        cumprod = np.flip(np.cumprod(np.flip(nchoices)))[1:]
        for m, c in zip(sorted_metadata, cumprod):
            m._godel_number = c

    @property
    def nchoices(self):
        return self._npossible

    @property
    def argument_names(self):
        return [a[0] for a in self._ordered_arguments]

    @property
    def repr_name(self):
        return self._ordered_arguments[0][0]

    def has_argument(self, aname):
        return aname in self.argument_names

    @property
    def default_value(self):
        return self.select(0)

    def select(self, index):
        return self._possible_values[index]

    def spawn_all_selections(self):
        return [ArgumentSelection(self, i) for i in range(self.nchoices)]

    @property
    def is_type(self) -> bool:
        return self._cat == ArgumentCategory.CAT_TYPE

    @property
    def is_feature(self) -> bool:
        return self._cat == ArgumentCategory.CAT_FEAT

    @property
    def is_functional(self) -> bool:
        return self.is_type or self.is_feature

    @property
    def is_performance(self) -> bool:
        return self._cat == ArgumentCategory.CAT_PERF

    @property
    def is_tensor(self) -> bool:
        triton_type = self._possible_values[0]
        if isinstance(triton_type, ConditionalConstexpr):
            return triton_type.is_tensor
        return isinstance(triton_type, str) and triton_type.startswith('*')

    @property
    def is_bool(self) -> bool:
        triton_type = self._possible_values[0]
        return isinstance(triton_type, bool)

    @property
    def is_int(self) -> bool:
        triton_type = self._possible_values[0]
        return isinstance(triton_type, int)

    def __get_param_cc_type(self, triton_arg):
        # Case I: use np.array as list
        if isinstance(self._possible_values, np.ndarray):
            triton_type = self._possible_values.dtype.type
            return ObjectFileDescription.SIGNATURE_TO_C[triton_type]
        # Case II: use list[str]
        triton_type = self._possible_values[0]
        # Case II-1: ['*fp16', ...]
        if self.is_tensor:
            rank = self._kdesc.get_tensor_rank(triton_arg)
            return f'const T{rank}*'

        # Case II-2: [ConditionalConstexpr]
        if isinstance(triton_type, ConditionalConstexpr):
            triton_type = triton_type.get_triton_type()
        # Case II-3: [lambda]
        elif callable(triton_type):
            triton_type = triton_type.__annotations__['return']

        # Case II-2/3: type annotation/np.dtype to cc_type
        if isinstance(triton_type, str):
            return ObjectFileDescription.SIGNATURE_TO_C[triton_type]
        elif isinstance(triton_type, np.ndarray):
            return ObjectFileDescription.SIGNATURE_TO_C[triton_type.dtype.type]
        # Case II-5: [False, True]
        elif isinstance(triton_type, bool):
            return 'bool'
        # Case II-6: [16, 32, ...]
        elif isinstance(triton_type, int):
            return 'int32_t'
        # Case II-7: [0.0, 0.1, ...]  # TODO: Unused in practice. Remove this later
        elif isinstance(triton_type, float):
            return 'float'
        assert False, f'{triton_arg} {triton_type}'

    @property
    def param_cc_fields(self):
        triton_arg = self._ordered_arguments[0][0]
        if triton_arg.startswith('stride_'):
            return []
        return [ self.__get_param_cc_type(a[0]) + ' ' + a[0] for a in self._ordered_arguments ]
        # ret = [ cc_type + ' ' + a[0] for a in self._ordered_arguments ]
        # print(f'{ret=}')

    # Return list of (ctype, aname, aindex)
    @property
    def param_cc_fields_tuple(self):
        triton_arg = self._ordered_arguments[0][0]
        if triton_arg.startswith('stride_'):
            return []
        return [ (self.__get_param_cc_type(a[0]), a[0], a[1]) for a in self._ordered_arguments ]

    @property
    def param_cc_size(self):
        for a in self._ordered_arguments:
            cc_type = self.__get_param_cc_type(a[0])
            return ObjectFileDescription.C_SIZE[cc_type]

    def codegen_godel_number_calculation(self, fout):
        if self.nchoices <= 1:
            return
        triton_arg = self._ordered_arguments[0][0]
        triton_type = self._possible_values[0]
        INDENT = 4 * ' '
        print(INDENT + '{', file=fout)
        print(2 * INDENT + 'int64_t number = 0;', file=fout)
        if self.is_tensor:
            for number, possible_type in enumerate(self._possible_values):
                elem_type = possible_type[1:].split(':')[0]
                print(2 * INDENT + f'if ({triton_arg}->dtype() == {self.DTYPE_NUMBER[elem_type]}) number = {number} ;', file=fout)
        else:
            for number, possible_type in enumerate(self._possible_values):
                value = str(possible_type).lower()
                print(2 * INDENT + f'if ({triton_arg} == {value}) number = {number} ;', file=fout)
        print(2 * INDENT + f'sum += number * {self.godel_number};', file=fout)
        print(1 * INDENT + '}', file=fout)

    @property
    def incomplete_tuning(self):
        return self._incomplete_tuning

    @property
    def fallback_tuning_value(self):
        return self._fallback_tuning_value

    def set_incomplete_tuning(self, fallback_value):
        self._incomplete_tuning = True
        self._fallback_tuning_value = fallback_value

    def get_codegen_compiled_in_features_ctype(self):
        ctype = None
        if self.is_bool:
            ctype = 'bool'
        elif self.is_int:
            ctype = 'int32_t'
        else:
            assert False, f'compiled_in_features: unknown meta type, choices {meta._possible_values}'
        return ctype

    def get_codegen_compiled_in_features_values(self):
        if self.is_bool:
            return ['true' if v else 'false' for v in self._possible_values]
        return [str(v) for v in self._possible_values]

class ArgumentSelection(object):
    def __init__(self, meta : ArgumentMetadata, selection_index : int):
        self._meta = meta
        self._selection_index = selection_index
        self._selection = self._meta.select(selection_index)
        self._is_conditional = callable(self._selection)
        self._selection_value = self._selection if not self._is_conditional else None

    @property
    def meta(self):
        return self._meta

    @property
    def nchoices(self):
        return self.meta.nchoices

    @property
    def argument_names(self):
        return self.meta.argument_names

    @property
    def repr_name(self):
        return self.meta.repr_name

    @property
    def argument_value(self):
        assert self._selection_value is not None
        return self._selection_value

    # Without None check, to support ConditionalConstexpr
    @property
    def tentative_value(self):
        return self._selection_value

    @property
    def is_conditional(self):
        return self._is_conditional

    def substitute_conditional(self, arch, fsel_dict):
        if self.is_conditional:
            # TOO Expensive
            # copy = deepcopy(self)
            # copy._selection_value = copy._selection(arch, fsel_dict)
            # assert copy._selection_value is not None
            # # print(f"substitute_conditional to {copy._selection_value} {fsel_dict=}")
            # return copy
            if self._selection_index is None:
                sub = TunedArgument(self._meta, self._selection)
            else:
                sub = ArgumentSelection(self._meta, self._selection_index)
            sub._selection_value = self._selection(arch, fsel_dict)
            return sub
        return self

    @property
    def godel_number(self):
        return self._meta.godel_number * self._selection_index

    @property
    def triton_signature(self):
        return str(self.argument_value)

    # compact_signature must be valid file name
    @property
    def compact_signature(self):
        if self.meta.is_functional and self.meta.nchoices <= 1:
            return None
        return self.triton_signature.replace('*', '^').replace(':', '@')

    @property
    def human_readable_signature(self):
        if len(self.meta.argument_names) <= 1:
            return f'{self.meta.argument_names[0]} = {self._selection_value}'
        return f'{self.meta.argument_names} = {self._selection_value}'

    def __repr__(self):
        return self.human_readable_signature

    def update_triton_api_signature(self, sig: dict):
        for place in self._meta.ordered_argument_places:
            sig[place] = self.triton_signature

    # FIXME: XXX_CHOICES's key is unordered
    '''
    Build a dict that maps "representative name" to selected value
    Consider changing it from frozenset to tuple

    Note: use tentative=True to disable None check in order to support ConditionalConstexpr
    '''
    @staticmethod
    def build_fsel_dict(fsels : 'list[ArgumentSelection]', tentative=False, all_args=False, with_meta=False):
        d = {}
        if all_args:
            for fsel in fsels:
                v = fsel.tentative_value if tentative else fsel.argument_value
                if with_meta:
                    v = (v, fsel)
                for aname in fsel.meta.argument_names:
                    d[aname] = v
        else:
            assert not with_meta, 'Unsupported'
            for fsel in fsels:
                d[fsel.meta.repr_name] = fsel.tentative_value if tentative else fsel.argument_value
        return d

class TunedArgument(ArgumentSelection):
    def __init__(self, meta : ArgumentMetadata, value):
        self._meta = meta
        assert not meta.is_functional, f'Functional argument cannot be tuned'
        self._selection_index = None
        if callable(value):
            self._selection = value
            self._is_conditional = True
            self._selection_value = None
        else:
            self._is_conditional = False
            self._selection_value = bool(value) if meta.is_bool else value
