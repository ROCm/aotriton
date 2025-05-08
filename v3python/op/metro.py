# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from ..base import (
    Interface,
    Functional,
)
from ..kernel import KernelDescription
import numpy as np

class MetroKernel(Interface):
    TUNE_NAME = None

    def __init__(self,
                 metro_name : str,
                 kernels : list[KernelDescription]):
        self.NAME = metro_name
        super().__init__()
        self._late_init()
        self._kernels = kernels

    @property
    def class_name_base(self):
        return "".join(x.capitalize() for x in self.NAME.lower().split("_"))

    @property
    def enum_name(self):
        return f'kMetro_{self.class_name_base}'

    def list_non_functional_params(self):
        return []

    def translate_dataframe(self, f : Functional, df : 'pandas.DataFrame'):
        raise RuntimeError(f'translate_dataframe should not be calle over any MetroKernel {self.NAME=}')

    def translate_empty_dataframe(self, f : Functional):
        raise RuntimeError(f'translate_empty_dataframe should not be calle over any MetroKernel {self.NAME=}')

    def list_kernels(self):
        return self._kernels
