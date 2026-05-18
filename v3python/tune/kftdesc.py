# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from pathlib import Path

class KernelForTuneDescription(ABC):
    """
    PT_* can be class variable when subclassing
    """

    @property
    @abstractmethod
    def PT_INPUT_CLASS(self):
        """
        (data)class to supply input/output tensors/parameters
        Some kernel's output tensor may become input tensor of other kernels.
        """
        pass

    @property
    @abstractmethod
    def PT_REF_CLASS(self):
        """
        Class to generate reference outputs
        """
        pass

    def __init__(self):
        pass

    @abstractmethod
    def create_extargs(self, *, hsaco_index=None, probe=False):
        pass

    @abstractmethod
    def generate_inputs(self, entry):
        """
        Pre-condition: called with device_ctx()
        Post-condition: a custom object that contains tensors/parameters is returned
        """
        pass

    def __call__(self, im, inputs, extargs):
        extargs = self.create_extargs() if extargs is None else extargs
        direct_inputs = self.prepare_directs(im, inputs)
        return self.direct_call(direct_inputs, extargs)

    @abstractmethod
    def prepare_directs(self, im, inputs):
        pass

    @abstractmethod
    def fill_nan_to_outputs(self, direct_inputs):
        pass

    @abstractmethod
    def direct_call(self, direct_inputs, extargs):
        pass

    def check_early_reject_results(self, result: dict, err) -> dict | None:
        """
        Called by run_single_test after compare() to detect graceful backend rejection.

        Returns:
            None  — not an early reject; caller uses result from compare() unchanged.
            dict  — result dict with adiff replaced by early-reject sentinel.

        Default: always returns None.  Override in BackendForTuneDescription.
        """
        return None

    @abstractmethod
    def compare(self, outputs, refs) -> list[float]:    # L1 error
        pass


class BackendForTuneDescription(KernelForTuneDescription):
    """
    Mixin for op-level kernels whose backends may gracefully reject unsupported inputs
    by returning hipErrorPeerAccessUnsupported.  Subclass this to record the negative
    sentinel adiff instead of NaN for those inputs.
    """

    def check_early_reject_results(self, result: dict, err) -> dict | None:
        from pyaotriton import hipError_t
        if err == hipError_t.hipErrorPeerAccessUnsupported:
            from .gpu_utils import record_early_reject
            return {name: record_early_reject(v) for name, v in result.items()}
        return None
