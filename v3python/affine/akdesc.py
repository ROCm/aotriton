# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
import numpy as np
import itertools
import functools
import operator
from collections import defaultdict
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
from ..gpu_targets import AOTRITON_SUPPORTED_GPUS, cluster_gpus
from ..utils import log
import pandas as pd

# TODO: Support Affine kernels that do not use direct arguments
class AffineKernelDescription(Interface):
    TUNE_NAME = None
    FILE_PFX = 'affine'
    NAME = None
    CO_CSV = None
    SUPPORTED_ARCH = None
    RESIDUAL_CHOICES = None     # Affine kernel may have finer requirements
    DIRECT_KERNEL_ARGS = None
    CSV_PROPERTIES = None

    @property
    def enum_name(self):
        return f'kAffine_{self.class_name_base}'

    @abstractmethod
    def co_dir(self, functional):
        pass

    def __init__(self):
        super().__init__()
        if self.RESIDUAL_CHOICES:
            self._residual_func_params = [TP(k, v) for k, v in self.RESIDUAL_CHOICES.items()]
        else:
            self._residual_func_params = []
        self._late_init()
        # Update Godel number if RESIDUAL_CHOICES present
        if self.RESIDUAL_CHOICES:
            TP.assign_godel_number(self._func_params + self._residual_func_params)
            self._godel_number = self._func_params[0].godel_number * self._func_params[0].nchoices
        if self.CO_CSV is not None:
            desc_path = Path(self.MODULE_FILE).parent
            self._df = pd.read_csv(desc_path / self.CO_CSV, skipinitialspace=True)
        self._residual_func_cfields  = sum([ p.get_unprocessed_cfields() for p in self.list_residual_functional_params() ], [])

    def list_residual_functional_params(self):
        yield from self._residual_func_params

    def list_non_functional_params(self):
        return []

    def gen_performance_params(self):
        yield from self._perf_params

    @property
    def perf_cfields(self):
        return self._perf_cfields

    # Overrides Interface.gen_functionals because Affine kernels may not
    # support all arch select to build
    def gen_functionals(self, build_for_target_arch):
        log(lambda : f'{self.__class__=} gen_functionals')
        target_arch = {}
        # Filter out unsupported arch
        for arch, gpus in build_for_target_arch.items():
            if arch in self.SUPPORTED_ARCH:
                target_arch[arch] = gpus
        all_func_params = self._func_params + self._residual_func_params
        def create_binds_from_nths(nths):
            return [ tp.create_nth(nth) for tp, nth in zip(all_func_params, nths) ]
        log(lambda : f'{self.__class__=} gen_functionals')
        for arch_number, arch in enumerate(target_arch.keys()):
            gpus = target_arch[arch]
            for nths in itertools.product(*all_func_params):
                binds = create_binds_from_nths(nths)
                yield Functional(self, arch, arch_number, binds, optimized_for=gpus)

    def translate_dataframe(self, f : Functional, df : 'pandas.DataFrame'):
        raise RuntimeError(f'translate_dataframe should not be calle over any AffineDescription {self.NAME=}')

    # Very kernel specific logic, leave for concrete class
    # def translate_empty_dataframe(self, f : Functional):

    @property
    def is_tunable(self):
        return False

    @property
    def residual_func_cfields(self):
        return self._residual_func_cfields

    @staticmethod
    def select_df_by_dict(df : pd.DataFrame, dic : dict):
        log(lambda : f'select_df_by_dict {dic}')
        conds = [ df[k] == v for k, v in dic.items() ]
        mask = functools.reduce(operator.__and__, conds)
        log(lambda : f'select_df_by_dict {df[mask]}')
        return df[mask]
