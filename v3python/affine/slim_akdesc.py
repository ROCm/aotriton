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
    Functional,
)
from ..gpu_targets import AOTRITON_SUPPORTED_GPUS, cluster_gpus
from ..utils import log
from ..gpu_targets import AOTRITON_ARCH_TO_PACK

'''
"Slim" version of Affine kernels. The corresponding affine kernel has mature
C++ dispather and does not need to generate one.
'''
class SlimAffineKernelDescription(Interface):
    TUNE_NAME = None
    FILE_PFX = 'affine'
    NAME = None     # Name for AOTriton shim code
    CO_DIR = None   # Name in the Affine kernel repository
    SUPPORTED_ARCH = None
    COOKIE_CLASS = None

    @property
    def enum_name(self):
        return f'kSlimAffine_{self.class_name_base}'

    def co_gen(self, build_dir: Path, build_for_target_arch: dict[str, list[str]]):
        target_arch = {}
        # Filter out unsupported arch
        for arch, gpus in build_for_target_arch.items():
            if arch in self.SUPPORTED_ARCH:
                target_arch[arch] = gpus
        archless_package_path = Path(self.FAMILY) / "affine_kernels" / self.CO_DIR
        for arch in target_arch.keys():
            aks2_path = Path(f"amd-{AOTRITON_ARCH_TO_PACK[arch]}") / archless_package_path
            aiter_arch = build_dir / self.AFFINE_KERNEL_ROOT / arch
            aiter_arch_module = aiter_arch / self.CO_DIR
            for kernel_co in aiter_arch_module.glob("**/*.co"):
                inarchive_path = kernel_co.relative_to(aiter_arch).as_posix()
                yield (aks2_path, inarchive_path, kernel_co)

    @property
    def perf_cfields(self):
        return []

    # Overrides Interface.gen_functionals to empty generator
    # because Slim Affine kernels have C++ dispatchers
    def gen_functionals(self, build_for_target_arch):
        return
        yield

    @property
    def is_tunable(self):
        return False

    def list_non_functional_params(self):
        pass

    def translate_dataframe(self, f : Functional, df : 'pandas.DataFrame'):
        pass

    def translate_empty_dataframe(self, f : Functional):
        pass
