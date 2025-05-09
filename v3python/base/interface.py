# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
import itertools
from .parameter import (
    TemplateParameter as TP,
)
from .conditional_value import (
    ConditionalConstexpr,
)
from . import typed_choice as TC
from .typed_choice import constexpr as TCC
from .functional import (
    Functional,
)
from ..utils import log

'''
Interface: common items b/w Operator, KernelDescription and MetroKernel
'''
class Interface(ABC):
    FAMILY = None               # Must be defined
    NAME = None                 # Must be defined
    TUNE_NAME = None            # Must be defined autotune/optune
    FILE_PFX = 'iface'          # Op uses iface, while Triton kernel uses 'shim'
    ARGUMENTS = None            # Must be defined
    SHARED_IFACE = None         # Optional, can be defined to share param struct
    TYPE_CHOICES = None         # Required For Operator
    FEAT_CHOICES = None         # Required For Operator
    PERF_CHOICES = None         # Required For KernelDescription/MetroKernel
    CHOICE_FILTERS = None       # Optional, Exclude unsupported combinations
    TENSOR_RANKS = None         # Operator, Required if Interface has Tensor Inputs
    TENSOR_STRIDE_INPUTS = None # Operator, Required if Interface has Tensor Inputs

    @property
    def UNTYPED_FULL_NAME(self):
        return f'{self.FAMILY}.{self.NAME}'

    def _insert_tensor_strides_to_choices(self, last_is_continuous=False):
        log(lambda:f"_insert_tensor_strides_to_choices {self.NAME} {self.TENSOR_STRIDE_INPUTS=}")
        for tensor, (strides, delete_when) in self.TENSOR_STRIDE_INPUTS.items():
            typed_strides = strides[:-1] if last_is_continuous else strides
            stride_dtype = TC.stride_a8() # 'u64:8' but hidden in cfields
            if delete_when is not None:
                feat, feat_value = delete_when
                stride_dtype = ConditionalConstexpr(feat, feat_value, 0, stride_dtype)
            self.TYPE_CHOICES[frozenset(typed_strides)] = [stride_dtype]
            constant_strides = [] if not last_is_continuous else strides[-1:]
            if constant_strides:
                self.FEAT_CHOICES[frozenset(constant_strides)] = [TCC.stride1()]
        log(lambda:f"_insert_tensor_strides_to_choices {self.NAME} {self.TYPE_CHOICES=}")
        log(lambda:f"_insert_tensor_strides_to_choices {self.NAME} {self.FEAT_CHOICES=}")

    # Early init
    def __init__(self):
        collected = self._collect_functionals_from_shared()
        self._insert_tensor_strides_to_choices(last_is_continuous=True)
        def __ttypes(anames, choices):
            for aname in anames:
                rank = self.get_tensor_rank(aname)
                break
            choices = [guess_tparam_type(v, rank=rank) for v in choices]
            if all([tt.is_tensor for tt in choices]):
                return TParam(anames, choices, ttype=create_tensor_type('any', rank))
            return TParam(anames, choices, ttype=typename_t)
        self._func_params = []
        self._func_params += [TP(k, v) for k, v in self.TYPE_CHOICES.items()]
        self._func_params += [TP(k, v) for k, v in self.FEAT_CHOICES.items()]

    def _late_init(self):
        tp_dict = self._build_tp_dict()
        log(lambda:f'_late_init {tp_dict=}')
        # Call late_init
        for m in self.list_all_params():
            m.late_init(self.ARGUMENTS, tp_dict, self.TENSOR_RANKS, self.TENSOR_STRIDE_INPUTS)
        # Sort
        self._func_params = sorted(self._func_params, key=lambda m: m.first_apperance)
        # Godel Numbers
        TP.assign_godel_number(self._func_params)
        self._godel_number = self._func_params[0].godel_number * self._func_params[0].nchoices
        # Func fields
        self._func_cfields = sum([ p.get_cfields() for p in self.list_functional_params() ], [])
        self._func_cfields = sorted(self._func_cfields, key=lambda p : p.index)

    '''
    Max Godel number assigned to this Interface
    '''
    @property
    def godel_number(self):
        return self._godel_number

    def list_functional_params(self):
        yield from self._func_params

    @abstractmethod
    def list_non_functional_params(self):
        pass

    def list_all_params(self):
        return self._func_params + self.list_non_functional_params()

    def _build_tp_dict(self):
        return { aname : param for param in self.list_all_params() for aname in param.all_names }

    @property
    def func_cfields(self):
        return self._func_cfields

    @classmethod
    def _class_name_base(klass):
        return "".join(x.capitalize() for x in klass.NAME.lower().split("_"))

    @property
    def class_name_base(self):
        return self._class_name_base()

    '''
    TODO: Auto-add 'op_' prefix to operators
    '''
    @property
    def param_class_name(self):
        if self.SHARED_IFACE is None:
            return self.class_name_base + 'Params'
        return self.SHARED_IFACE._class_name_base() + 'Params'

    @property
    def context_class_name(self):
        return self.class_name_base + 'Context'

    @property
    def metadata_class_name(self):
        return self.class_name_base + 'Metadata'

    '''
    This is also the string stored in the database
    '''
    @property
    @abstractmethod
    def enum_name(self) -> str:
        pass

    def get_tensor_rank(self, tensor_arg):
        log(lambda : f'get_tensor_rank {self=} {self.TENSOR_RANKS=}')
        return self.TENSOR_RANKS.get(tensor_arg, self.TENSOR_RANKS['_default'])

    def gen_functionals(self, target_arch):
        def create_binds_from_nths(nths):
            return [ tp.create_nth(nth) for tp, nth in zip(self._func_params, nths) ]
        for arch_number, arch in enumerate(target_arch.keys()):
            gpus = target_arch[arch]
            for nths in itertools.product(*self._func_params):
                binds = create_binds_from_nths(nths)
                yield Functional(self, arch, arch_number, binds, optimized_for=gpus)

    @abstractmethod
    def translate_dataframe(self, f : Functional, df : 'pandas.DataFrame'):
        pass

    @abstractmethod
    def translate_empty_dataframe(self, f : Functional):
        pass

    def _collect_functionals_from_shared(self):
        mklass = self.SHARED_IFACE
        if mklass is None:
            return True
        log(lambda:f'_collect_functionals_from_shared {self.__class__=} {mklass=}')
        log(lambda:f'_collect_functionals_from_shared source values:',
            lambda:f'{mklass.TYPE_CHOICES=}',
            lambda:f'{mklass.FEAT_CHOICES=}',
            lambda:f'{mklass.TENSOR_RANKS=}',
            lambda:f'{mklass.TENSOR_STRIDE_INPUTS=}',
            sep='\n')
        # Early detection
        # all_assigned = True
        # for which_choice in ['TYPE_CHOICES', 'FEAT_CHOICES']:
        #     if getattr(self, which_choice, None) is None:
        #         all_assigned = False
        #         break
        # if all_assigned:
        #     return True
        args_order = { aname : i for i, aname in enumerate(self.ARGUMENTS) }
        args_in_use = set(self.ARGUMENTS)
        log(f'_collect_functionals_from_shared {args_order=}')
        # Selection is defined in Op but not all options are available in individual kernels
        def remove_missing(args_to_determine : frozenset):
            log(lambda:f'intersection {args_to_determine=} vs {args_in_use=}')
            args_to_determine = set(args_to_determine).intersection(args_in_use)
            log(lambda:f'result {args_to_determine=}')
            sorted_args = sorted(args_to_determine, key = lambda aname : args_order[aname])
            log(lambda:f'result {sorted_args=}')
            return tuple(sorted_args)  # TODO: replace frozenset with tuple

        CHOICE_FILTERS = self.CHOICE_FILTERS
        # remove_unsupported(('Q', 'K', 'V'), [16, 32, 64]) = [16], when CHOICE_FILTERS = { 'K' : lambda x : x < 32 }
        def remove_unsupported(key, values):
            if not CHOICE_FILTERS:
                return values
            for k in key:
                if k in CHOICE_FILTERS:
                    return [ v for v in values if CHOICE_FILTERS[k](v) ]
            return values

        for which_choice in ['TYPE_CHOICES', 'FEAT_CHOICES']:
            if getattr(self, which_choice, None) is not None:
                continue
            mattr = getattr(mklass, which_choice)
            dic = {}
            for k, v in mattr.items():
                args = remove_missing(k)
                v = remove_unsupported(k, v)
                if args:
                    dic[args] = v
            setattr(self, which_choice, dic)
            log(f"{self}'s final {which_choice} is {dic}")

        for which_choice in ['TENSOR_RANKS', 'TENSOR_STRIDE_INPUTS']:
            if getattr(self, which_choice, None) is not None:
                continue
            mattr = getattr(mklass, which_choice)
            dic = {}
            for k, v in mattr.items():
                if k in args_in_use:
                    dic[k] = v
            setattr(self, which_choice, dic)
        self.TENSOR_RANKS['_default'] = mklass.TENSOR_RANKS['_default']
        log(f'_collect_functionals_from_shared final values:',
            f'{self.TYPE_CHOICES=}',
            f'{self.FEAT_CHOICES=}',
            f'{self.TENSOR_RANKS=}',
            f'{self.TENSOR_STRIDE_INPUTS=}',
            sep='\n')
        return True
