# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import numpy as np
import itertools
from collections import defaultdict
import io
import os
from pathlib import Path
from ..base import (
    Interface,
    Functional,
    ConditionalChoice,
    ConditionalConstexpr,
    ConditionalDeferredConstexpr,
    ConditionalDeferredElseTensor,
    TemplateParameter as TP,
    PerformanceTemplateParameter as PTP,
)
from .ksignature import KernelSignature, COMPILER_OPTIONS, DEFAULT_COPT
from ..gpu_targets import AOTRITON_SUPPORTED_GPUS, cluster_gpus
from ..utils import log
import pandas as pd

SOURCE_PATH = Path(__file__).resolve()
AOTRITON_ENABLE_FP32 = bool(int(os.getenv('AOTRITON_ENABLE_FP32', True)))

def join_dicts(dicts : 'list[dict]') -> dict:
    return { k:v for d in dicts for k,v in d.items() }

def get_possible_choices(klass, arg_name : str) -> 'list[Any]':
    l = []
    for k in ['TYPE_CHOICES', 'FEAT_CHOICES', 'PERF_CHOICES']:
        if hasattr(klass, k) and getattr(klass, k):
            l += [getattr(klass, k)]
    log(lambda : f'{l=}')
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

class KernelDescription(Interface):
    TUNE_NAME = 'autotune'
    FILE_PFX = 'shim'
    ARGUMENTS = []
    NAME = None
    _ARGUMENT_CHOICES = None

    @property
    def enum_name(self):
        return f'kShim_{self.class_name_base}'

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
            log(lambda : f'{self._DATA_ARGUMENTS=}')
        return self._DATA_ARGUMENTS

    def is_functional_disabled(self, functional):
        return False

    def __init__(self, triton_kernel_name, triton_source_path):
        super().__init__()
        self._DATA_ARGUMENTS = None
        self._triton_source_path = Path(triton_source_path)
        self._triton_kernel_name = triton_kernel_name
        # FIXME: Support tensor with different ranks
        self._perf_params = [PTP(k, v) for k, v in self.PERF_CHOICES.items()]
        self._late_init()
        self.__autotune_keys_init()
        # Initialization of _func_cfields and _perf_cfields
        self._perf_cfields = sum([ p.get_cfields() for p in self.gen_performance_params() ], [])
        # Perf is sorted by size for more compact storage
        # Not always optimal, but good enough for now.
        self._perf_cfields = sorted(self._perf_cfields, key=lambda p : p.nbits, reverse=True)

    def __autotune_keys_init(self):
        # print(f'{self._func_meta}')
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

    def list_non_functional_params(self):
        return list(self.gen_performance_params())

    @property
    def triton_source_path(self):
        return self._triton_source_path

    @property
    def triton_kernel_name(self):
        return self._triton_kernel_name

    def gen_performance_params(self):
        yield from self._perf_params

    @property
    def perf_cfields(self):
        return self._perf_cfields

    def fallback_compact_dict(self, compact_dict):
        def fallback(k, v):
            return self.PARTIALLY_TUNED_FUNCTIONALS.get(k, v)
        return { k : fallback(k, v) for k, v in compact_dict.items()}

    # TODO: dataframe name mangling should be deferred to database package.
    #       Possible solution is to attach a translator to DataFrame object
    def translate_dataframe(self, f : Functional, df : 'pandas.DataFrame'):
        '''
        Extract keys from kdesc
        '''
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
        lut_tensor = np.full(lut_shape, -1, dtype=np.int32)
        perf_keys = [ f'tuned_kernel${meta.repr_name}' for meta in self._perf_params ]
        copt_keys = [ f'compiler_options${key}' for key in COMPILER_OPTIONS ]
        '''
        Deduplication and assign numbers
        '''
        log(lambda : f'{df[perf_keys + copt_keys]=}')
        np_sigs, revind = np.unique(df[perf_keys + copt_keys].to_numpy(), axis=0, return_inverse=True)
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
        '''
        Bucketing autotune indices
        '''
        for i, ind_key in enumerate(sparse_keys):
            bucket = sparse_key_possible_values[ind_key]
            def discretization(v):
                return bucket.index(v)
            df[f'$$ind_{i}'] = df[ind_key].apply(discretization)
        '''
        Create LUT
        Note: df's gpu column comes directly from DB. Hence database_gpus should be used.
        '''
        for i, gpu in enumerate(f.database_gpus):
            if i > 0:
                lut_tensor[i] = lut_tensor[0]
            df_i = df[df['gpu'] == gpu]
            inds = tuple([df_i[f'$$ind_{i}'] for i in range(nkeys)])
            sig_nums = df_i['$$sig_num']
            lut_tensor[i][inds] = sig_nums
        '''
        Downcast LUT datatype
        Usually int8 is sufficient but let's be safe.
        '''
        nsigs = len(sigs)
        for dtype in [np.int8, np.int16, np.int32]:
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

    def gen_signatures_for_tuning(self, f : Functional):
        def gen_perfs(cfg) -> 'list[Bind]':
            for meta in self._perf_params:
                value = cfg.kwargs[meta.repr_name]
                yield meta.create_direct(value)
        def gen_copts(cfg) -> list[int]:
            for copt, defopt in zip(COMPILER_OPTIONS, DEFAULT_COPT):
                yield getattr(cfg, copt, defopt)
        for cfg in self.gen_autotune_configs(f):
            yield KernelSignature(f, list(gen_perfs(cfg)), list(gen_copts(cfg)))

    @property
    def is_tunable(self):
        return hasattr(self, 'gen_autotune_configs')
