# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from pathlib import Path

class KernelForTuneDescription(ABC):
    '''
    PT_* can be class variable when subclassing
    '''
    @property
    @abstractmethod
    def PT_INPUT_CLASS(self):
        pass

    @property
    @abstractmethod
    def PT_REF_CLASS(self):
        pass

    def __init__(self):
        pass

    @abstractmethod
    def create_extargs(self, *, force_kernel_index=None, peek_kernel_numbers=None):
        pass

    @abstractmethod
    def generate_inputs(self, entry, *, dry_run=False):
        pass

    def __call__(self, im, inputs, extargs):
        extargs = self.create_extargs() if extargs is None else extargs
        direct_inputs = self.prepare_directs(im, inputs)
        return self.direct_call(direct_inputs, extargs)

    @abstractmethod
    def prepare_directs(self, im, inputs):
        pass

    @abstractmethod
    def direct_call(self, direct_inputs, extargs):
        pass

    @abstractmethod
    def compare(self, outputs, refs) -> list[float]:    # L1 error
        pass
