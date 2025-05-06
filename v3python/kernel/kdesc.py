# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import numpy as np
import itertools
from collections import defaultdict
import io
import os
from pathlib import Path
from ..base import (
    Tunable,
    Functional,
    ConditionalChoice,
    ConditionalConstexpr,
    ConditionalDeferredConstexpr,
    ConditionalDeferredElseTensor,
    TemplateParameter as TP,
    PerformanceTemplateParameter as PTP,
)
from ..base import typed_choice as TC
from ..op import (
    Operator,
    NO_OPERATOR,
)
from .ksignature import KernelSignature, COMPILER_OPTIONS, DEFAULT_COPT
from ..gpu_targets import AOTRITON_SUPPORTED_GPUS, cluster_gpus
from ..utils import get_template
import pandas as pd

SOURCE_PATH = Path(__file__).resolve()
AOTRITON_ENABLE_FP32 = bool(int(os.getenv('AOTRITON_ENABLE_FP32', True)))

def join_dicts(dicts : 'list[dict]') -> dict:
    return { k:v for d in dicts for k,v in d.items() }

def get_possible_choices(klass, arg_name : str) -> 'list[Any]':
    l = []
    for k in ['TYPE_CHOICES', 'FEAT_CHOICES', 'PERF_CHOICES']:
        if hasattr(klass, k):
            l += [getattr(klass, k)]
    for d in l:
        for k, v in d.items():
            if arg_name in k:
                return v
    assert False, f"cannot find {arg_name}"

def select_pattern(arguments, prefix, trim_left=None, trim_right=None, delete_when=None):
    ret = []
    for s in arguments:
        assert s.strip() == s, f'Input argument {s} within {arguments=} contains spaces at either end'
        if s.startswith(prefix):
            ret.append(s)
    return (ret[trim_left:trim_right], delete_when)

def collect_functionals_from_op(klass):
    mklass = klass.OPERATOR
    if mklass is None and klass.CONFIRM_NO_OPERATOR:
        return
    print(f'collect_functionals_from_op {klass=} {mklass=}')
    assert mklass, f'Class {klass} must define OPERATOR'
    # Early detection
    all_assigned = True
    for which_choice in ['TYPE_CHOICES', 'FEAT_CHOICES']:
        if getattr(klass, which_choice, None) is None:
            all_assigned = False
            break
    if all_assigned:
        return
    args_order = { aname : i for i, aname in enumerate(klass.ARGUMENTS) }
    args_in_use = set(klass.ARGUMENTS)
    print(f'{args_order=}')
    # Selection is defined in Op but not all options are available in individual kernels
    def remove_missing(args_to_determine : frozenset):
        print(f'intersection {args_to_determine=} vs {args_in_use=}')
        args_to_determine = set(args_to_determine).intersection(args_in_use)
        print(f'result {args_to_determine=}')
        sorted_args = sorted(args_to_determine, key = lambda aname : args_order[aname])
        print(f'result {sorted_args=}')
        return tuple(sorted_args)  # TODO: replace frozenset with tuple

    CHOICE_FILTERS = klass.CHOICE_FILTERS
    # remove_unsupported(('Q', 'K', 'V'), [16, 32, 64]) = [16], when CHOICE_FILTERS = { 'K' : lambda x : x < 32 }
    def remove_unsupported(key, values):
        if not CHOICE_FILTERS:
            return values
        for k in key:
            if k in CHOICE_FILTERS:
                return [ v for v in values if CHOICE_FILTERS[k](v) ]
        return values

    for which_choice in ['TYPE_CHOICES', 'FEAT_CHOICES']:
        if getattr(klass, which_choice, None) is not None:
            continue
        mattr = getattr(mklass, which_choice)
        dic = {}
        for k, v in mattr.items():
            args = remove_missing(k)
            v = remove_unsupported(k, v)
            if args:
                dic[args] = v
        setattr(klass, which_choice, dic)
        print(f"{klass}'s final {which_choice} is {dic}")

    for which_choice in ['TENSOR_RANKS', 'TENSOR_STRIDE_INPUTS']:
        if getattr(klass, which_choice, None) is not None:
            continue
        mattr = getattr(mklass, which_choice)
        dic = {}
        for k, v in mattr.items():
            if k in args_in_use:
                dic[k] = v
        setattr(klass, which_choice, dic)
    klass.TENSOR_RANKS['_default'] = mklass.TENSOR_RANKS['_default']

class KernelDescription(Tunable):
    TUNE_NAME = 'autotune'
    OPERATOR = None
    ARGUMENTS = []
    NAME = None
    _ARGUMENT_CHOICES = None
    HEADER_TEMPLATE = get_template('shim.h')
    SOURCE_TEMPLATE = get_template('shim.cc')

    # Type and Feature are shared from Related Op
    # TYPE_CHOICES = {
    # }
    # FEAT_CHOICES = {
    # }
    PERF_CHOICES = {
    }
    # Exclude unsupported combinations
    CHOICE_FILTERS = {
    }

    @property
    def FULL_KERNEL_NAME(self):
        return f'{self.FAMILY}.{self.NAME}'

    @property
    def enum_name(self):
        # CamelName = self.NAME.replace('_', ' ').title().replace(' ', '')
        return f'kShim_{self.class_name_base}'

    @property
    def class_name_base(self):
        return "".join(x.capitalize() for x in self.NAME.lower().split("_"))

    @property
    def param_class_name(self):
        return self.class_name_base + 'Params'

    @property
    def context_class_name(self):
        return self.class_name_base + 'Context'

    @property
    def metadata_class_name(self):
        return self.class_name_base + 'Metadata'

    @property
    def ARGUMENT_CHOICES(self):
        if self._ARGUMENT_CHOICES is None:
            self._ARGUMENT_CHOICES = join_dicts([self.TYPE_CHOICES, self.FEAT_CHOICES, self.PERF_CHOICES])
        return self._ARGUMENT_CHOICES

    @property
    def KERNEL_DATA_ARGUMENTS(self):
        if self._DATA_ARGUMENTS is None:
            def is_data_argument(a):
                for k in self.TYPE_CHOICES.keys():
                    if a in k:
                        return True
                return False
            self._DATA_ARGUMENTS = [ a for a in self.ARGUMENTS if is_data_argument(a) ]
            print(f'{self._DATA_ARGUMENTS=}')
        return self._DATA_ARGUMENTS

    def is_functional_disabled_on_arch(self, arch, fsels):
        return False

    def insert_tensor_strides_to_choices(self, last_is_continuous=False):
        for tensor, (strides, delete_when) in self.TENSOR_STRIDE_INPUTS.items():
            typed_strides = strides[:-1] if last_is_continuous else strides
            stride_dtype = TC.stride_a8() # 'u64:8' but hidden in cfields
            if delete_when is not None:
                feat, feat_value = delete_when
                stride_dtype = ConditionalConstexpr(feat, feat_value, 0, stride_dtype)
            self.TYPE_CHOICES[frozenset(typed_strides)] = [stride_dtype]
            constant_strides = [] if not last_is_continuous else strides[-1:]
            if constant_strides:
                self.FEAT_CHOICES[frozenset(constant_strides)] = [TC.constexpr.stride1()]
        print(f"{self.TYPE_CHOICES=}")
        print(f"{self.FEAT_CHOICES=}")

    def __init__(self, triton_kernel_name, triton_source_path):
        collect_functionals_from_op(self.__class__)
        self.insert_tensor_strides_to_choices(last_is_continuous=True)
        self._DATA_ARGUMENTS = None
        self._triton_source_path = Path(triton_source_path)
        self._triton_kernel_name = triton_kernel_name
        self._func_params = []
        # FIXME: Support tensor with different ranks
        def __ttypes(anames, choices):
            for aname in anames:
                rank = self.get_tensor_rank(aname)
                break
            choices = [guess_tparam_type(v, rank=rank) for v in choices]
            if all([tt.is_tensor for tt in choices]):
                return TParam(anames, choices, ttype=create_tensor_type('any', rank))
            return TParam(anames, choices, ttype=typename_t)
        self._func_params += [TP(k, v) for k, v in self.TYPE_CHOICES.items()]
        self._func_params += [TP(k, v) for k, v in self.FEAT_CHOICES.items()]
        self._perf_params = [PTP(k, v) for k, v in self.PERF_CHOICES.items()]

        tp_dict = { aname : param for param in self._func_params + self._perf_params for aname in param.all_names }

        for m in self._func_params:
            m.late_init(self.ARGUMENTS, tp_dict, self.TENSOR_RANKS, self.TENSOR_STRIDE_INPUTS)
            # for u, fallback in self.PARTIALLY_TUNED_FUNCTIONALS:
            #     if m.has_argument(u):
            #         m.set_incomplete_tuning(fallback)
        for m in self._perf_params:
            m.late_init(self.ARGUMENTS, tp_dict, self.TENSOR_RANKS, self.TENSOR_STRIDE_INPUTS)
        self._func_params = sorted(self._func_params, key=lambda m: m.first_apperance)
        # print(f'{self._func_meta}')
        TP.assign_godel_number(self._func_params)
        self._godel_number = self._func_params[0].godel_number * self._func_params[0].nchoices
        self.AUTOTUNE_KEYS_VALIDATED = []
        for key in self.ARGUMENTS:
            if key not in self.AUTOTUNE_KEYS:
                continue
            is_type = False
            for type_keys in self.TYPE_CHOICES.keys():
                if key in type_keys:
                    is_type = True
                    break
            if is_type:
                self.AUTOTUNE_KEYS_VALIDATED.append((key, self.AUTOTUNE_KEYS[key]))
        '''
        AUTOTUNE_KEYS sanity check, otherwise autotune code may be broken (already happened twice).
        '''
        for key in self.AUTOTUNE_KEYS:
            assert key in self.ARGUMENTS, f'AUTOTUNE_KEYS "{key}" cannot be found in {self.__class__.__name__}.ARGUMENTS'
        # Initialization of _func_cfields and _perf_cfields
        self._func_cfields = sum([ p.get_cfields() for p in self.list_functional_params() ], [])
        self._func_cfields = sorted(self._func_cfields, key=lambda p : p.index)
        self._perf_cfields = sum([ p.get_cfields() for p in self.list_performance_params() ], [])
        # Perf is sorted by size for more compact storage
        # Not always optimal, but good enough for now.
        self._perf_cfields = sorted(self._perf_cfields, key=lambda p : p.nbits, reverse=True)

    @property
    def triton_source_path(self):
        return self._triton_source_path

    @property
    def triton_kernel_name(self):
        return self._triton_kernel_name

    def list_functional_params(self):
        yield from self._func_params

    def list_performance_params(self):
        yield from self._perf_params

    @property
    def func_cfields(self):
        return self._func_cfields

    @property
    def perf_cfields(self):
        return self._perf_cfields

    def get_tensor_rank(self, tensor_arg):
        print(f'{self=} {self.TENSOR_RANKS=}')
        return self.TENSOR_RANKS.get(tensor_arg, self.TENSOR_RANKS['_default'])

    def gen_functionals(self, target_arch):
        def create_binds_from_nths(nths):
            return [ tp.create_nth(nth) for tp, nth in zip(self._func_params, nths) ]
        for arch_number, arch in enumerate(target_arch.keys()):
            gpus = target_arch[arch]
            for nths in itertools.product(*self._func_params):
                binds = create_binds_from_nths(nths)
                yield Functional(self, arch, arch_number, binds, optimized_for=gpus)

    def fallback_compact_dict(self, compact_dict):
        def fallback(k, v):
            return self.PARTIALLY_TUNED_FUNCTIONALS.get(k, v)
        return { k : fallback(k, v) for k, v in compact_dict.items()}

    @property
    def godel_number(self):
        return self._godel_number

    # TODO: dataframe name mangling should be deferred to database package.
    #       Possible solution is to attach a translator to DataFrame object
    def translate_dataframe(self, f : Functional, df : 'pandas.DataFrame'):
        sparse_keys = [ f'inputs${key}' for key, _ in self.AUTOTUNE_KEYS_VALIDATED ]
        nkeys = len(sparse_keys)
        # print(f'{sparse_keys=}')
        def sorted_unique_key(key):
            return np.unique(df[key].to_numpy()).tolist()
        sparse_key_possible_values = { key : sorted_unique_key(key) for key in sparse_keys }
        binning_dict = { key : algo(sparse_key_possible_values[spk]) for spk, (key, algo) in zip(sparse_keys, self.AUTOTUNE_KEYS_VALIDATED) }
        # sparse_shape is not used because lut is compact
        lut_shape = [f.noptimized_for] + [ len(sparse_key_possible_values[key]) for key in sparse_keys ]
        # lut starts with a large enough dtype
        lut_tensor = np.empty(lut_shape, dtype=np.int32)
        perf_keys = [ f'tuned_kernel${meta.repr_name}' for meta in self._perf_params ]
        copt_keys = [ f'compiler_options${key}' for key in COMPILER_OPTIONS ]
        # def discretization(key, value):
        #     # print(f'discretization {key=} {value=} to {sparse_key_possible_values[key].index(value)}')
        #     return sparse_key_possible_values[key].index(value)
        # def find_ind(series):
        #     sparse = series[sparse_keys]
        #     return tuple([ discretization(key, value) for key, value in zip(sparse_keys, sparse) ])
        # def register_sig(series):
        #     # print(f'{perf_keys=}')
        #     # print(f'{copt_keys=}')
        #     # print(f'{series=}')
        #     # import ipdb; ipdb.set_trace();
        #     key = tuple(list(series[perf_keys]) + list(series[copt_keys]))
        #     if key not in sigs_dict:
        #         ret = len(sigs)
        #         sigs_dict[key] = ret
        #         sig = KernelSignature(f,
        #                               perf_bind(series),
        #                               series[copt_keys])
        #         sigs.append(sig)
        #     return sigs_dict[key]
        # def assign_unique_combinations(df, columns_to_combine, new_column_name):
        #     """Assigns unique integer values to unique combinations of columns.

        #     Args:
        #         df: The input Pandas DataFrame.
        #         columns_to_combine: A list of column names to combine.
        #         new_column_name: The name of the new column to store the assigned values.

        #     Returns:
        #         The DataFrame with the new column added.
        #     """
        #     df[new_column_name] = pd.factorize(df[columns_to_combine].apply(tuple, axis=1))[0]
        #     return df
        # assign_unique_combinations(df, perf_keys + copt_keys, '$$sig_num')
        np_sigs, revind = np.unique(df[perf_keys + copt_keys].to_numpy(), axis=0, return_inverse=True)
        # df[i] == np_sigs[revind[i]]
        df['$$sig_num'] = revind
        sigs_dict = {}
        def perf_bind(nprow):
            return [ meta.create_direct(value) for meta, value in zip(self._perf_params, nprow) ]
        nperfs = len(perf_keys)
        def create_sig(nprow):
            return KernelSignature(f,
                                   perf_bind(nprow),
                                   nprow[nperfs:].tolist())
        sigs = [ create_sig(nprow) for nprow in np_sigs ]
        for i, ind_key in enumerate(sparse_keys):
            bucket = sparse_key_possible_values[ind_key]
            def discretization(v):
                return bucket.index(v)
            df[f'$$ind_{i}'] = df[ind_key].apply(discretization)
        for i, gpu in enumerate(f.optimized_for):
            if i > 0:
                lut_tensor[i] = lut_tensor[0]
            df_i = df[df['gpu'] == gpu]
            '''
            for idx, series in df_i.iterrows():
                lut_ind = find_ind(series)
                sig_num = register_sig(series)
                # print(f'lut_tensor[{i}][{lut_ind}] = {sig_num}')
                lut_tensor[i][lut_ind] = sig_num
            '''
            inds = tuple([df_i[f'$$ind_{i}'] for i in range(nkeys)])
            sig_nums = df_i['$$sig_num']
            lut_tensor[i][inds] = sig_nums
        nsigs = len(sigs)
        for dtype in [np.int8, np.int16, np.int32, np.int64]:
            if nsigs < np.iinfo(dtype).max:
                break
        lut_tensor = lut_tensor.astype(dtype)
        # self.sancheck_lut_tensor(lut_tensor)
        # print(f'{lut_tensor=}')
        # print(f'{sigs=}')
        # print(f'{binning_dict=}')
        return lut_tensor, sigs, binning_dict

    def translate_empty_dataframe(self, f : Functional):
        lut_tensor = np.zeros([f.noptimized_for, 1], dtype=np.int8)
        defaults = [ meta.create_nth(0) for meta in self._perf_params ]
        sigs = [ KernelSignature(f, defaults, DEFAULT_COPT) ]
        return lut_tensor, sigs, None

    @property
    def is_tunable(self):
        return hasattr(self, 'gen_autotune_configs')
