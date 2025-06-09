# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
import numpy as np
import itertools
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
from .ksignature import KernelSignature, COMPILER_OPTIONS, DEFAULT_COPT
from ..gpu_targets import AOTRITON_SUPPORTED_GPUS, cluster_gpus
from ..utils import log
import pandas as pd

class AffineKernelDescription(Interface):
    TUNE_NAME = None
    FILE_PFX = 'affine'
    NAME = None
    CO_CSV = None
    SUPPORTED_ARCH = None
    RESIDUAL_CHOICES = None     # Affine kernel may have finer requirements

    @property
    def enum_name(self):
        return f'kAffine_{self.class_name_base}'

    def __init__(self, affine_name):
        self.NAME = affine_name
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
            df = pd.read_csv(desc_path / self.CO_CSV)
            self._asmdb = self.translate_csv_df(df)

    def list_residual_functional_params(self):
        yield from self._residual_func_params

    def list_non_functional_params(self):
        return []

    def gen_performance_params(self):
        yield from self._perf_params

    @property
    def perf_cfields(self):
        return self._perf_cfields

    def fallback_compact_dict(self, compact_dict):
        def fallback(k, v):
            return self.PARTIALLY_TUNED_FUNCTIONALS.get(k, v)
        return { k : fallback(k, v) for k, v in compact_dict.items()}

    # Overrides Interface.gen_functionals because Affine kernels may not
    # support all arch select to build
    def gen_functionals(self, build_for_target_arch):
        target_arch = {}
        # Filter out unsupported arch
        for arch, gpus in target_arch.items():
            if arch in self.SUPPORTED_ARCH:
                target_arch[arch] = gpus
        return super().gen_functionals(target_arch)

    def translate_dataframe(self, f : Functional, df : 'pandas.DataFrame'):
        raise RuntimeError(f'translate_dataframe should not be calle over any AffineDescription {self.NAME=}')

    def translate_empty_dataframe(self, f : Functional):
        # TODO
        lut_tensor = np.zeros([f.noptimized_for, 1], dtype=np.int8)
        sigs = [ self.create_asm_description(f) ]
        return lut_tensor, sigs, None

    @abstractmethod
    def create_asm_description(self, f):
        pass

    @property
    def is_tunable(self):
        return False

    def translate_csv_df(self, csv):
