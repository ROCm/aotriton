# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from pathlib import Path

class KernelForTuneDescription(ABC):
    '''
    PT_* can be class variable when subclassing
    '''

    '''
    (data)class to supply input/output tensors/parameters
    Some kernel's output tensor may become input tensor of other kernels.
    '''
    @property
    @abstractmethod
    def PT_INPUT_CLASS(self):
        pass

    '''
    Class to generate reference outputs
    '''
    @property
    @abstractmethod
    def PT_REF_CLASS(self):
        pass

    def __init__(self):
        pass

    @abstractmethod
    def create_extargs(self, *, force_kernel_index=None, peek_kernel_numbers=None):
        pass

    '''
    Pre-condition: called with device_ctx()
    Post-condition: a custom object that contains tensors/parameters is returned
    '''
    @abstractmethod
    def generate_inputs(self, entry):
        pass

    def __call__(self, im, inputs, extargs):
        extargs = self.create_extargs() if extargs is None else extargs
        direct_inputs = self.prepare_directs(im, inputs)
        return self.direct_call(direct_inputs, extargs)

    '''
    Pre-condition:
    '''
    @abstractmethod
    def prepare_directs(self, im, inputs):
        pass

    @abstractmethod
    def fill_nan_to_outputs(self, direct_inputs):
        pass

    @abstractmethod
    def direct_call(self, direct_inputs, extargs):
        pass

    @abstractmethod
    def compare(self, outputs, refs) -> list[float]:    # L1 error
        pass
