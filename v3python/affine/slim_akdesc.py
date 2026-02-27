# Copyright © 2023-2026 Advanced Micro Devices, Inc.
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
)
from ..gpu_targets import AOTRITON_SUPPORTED_GPUS, cluster_gpus
from ..utils import log

'''
"Slim" version of Affine kernels. The corresponding affine kernel has mature
C++ dispather and does not need to generate one.
'''
class SlimAffineKernelDescription(Interface):
    TUNE_NAME = None
    FILE_PFX = 'affine'
    NAME = None
    SUPPORTED_ARCH = None

    @property
    def enum_name(self):
        return f'kSlimAffine_{self.class_name_base}'

    # FIXME: replace with co_path_gen. gfx942/fmha_v3_fwd/ has more complicated structure
    #        Generator should yield (co file path, aks2 filename)
    #        Note: aks2 "filename" can be any valid C-string, hence it should
    #              yield the relative path that being returned by
    #              get_heuristic_kernel (in mha_bwd.cc)
    @abstractmethod
    def co_dir(self, build_dir: Path, functional):
        pass

    @property
    def perf_cfields(self):
        return []

    # Overrides Interface.gen_functionals to empty
    # because Slim Affine kernels have C++ dispatchers
    def gen_functionals(self, build_for_target_arch):
        return

    @property
    def is_tunable(self):
        return False

