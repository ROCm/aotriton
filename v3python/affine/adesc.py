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

class AffineDescription(Interface):
    TUNE_NAME = None
    FILE_PFX = 'affine'
    NAME = None
    CO_CSV = None
    SUPPORTED_ARCH = None

    @property
    def enum_name(self):
        return f'kAffine_{self.class_name_base}'

    def __init__(self, affine_name):
        self.NAME = affine_name
        super().__init__()
        if self.CO_CSV is not None:
            desc_path = Path(self.MODULE_FILE).parent
            self._csvdf = pd.read_csv(desc_path / self.CO_CSV)

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
