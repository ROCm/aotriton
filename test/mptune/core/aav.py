#!/usr/bin/env python
# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from .rocm_arch import rocm_get_gpuarch
from abc import ABC

class ArgArchVerbose(ABC):
    def __init__(self, args):
        self._args = args
        self._arch = rocm_get_gpuarch()

    @property
    def verbose(self):
        return self._args.verbose

    def print(self, text):
        if self.verbose:
            print(text)

