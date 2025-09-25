# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, astuple
from ..kftdesc import KernelForTuneDescription as KFTDesc
from .reference import (
    SdpaReference,
    SdpaBidiInputs,
    SdpaBidiOutputs,
)
from pyaotriton.v2.flash import (
    attn_fwd as fa_forward,
    attn_fwd_compact_varlen as fa_forward_compact_varlen,
    debug_simulate_encoded_softmax as fa_debug_simulate_encoded_softmax,
    FwdExtraArguments,
)

class SdpaCommon(SdpaReference):
    EXT_CLASS = FwdExtraArguments

    def create_extargs(self, *, force_kernel_index=None, peek_kernel_numbers=None):
        ext = self.EXT_CLASS()
        if force_kernel_index is not None:
            ext.force_kernel_index = force_kernel_index
        if peek_kernel_numbers is not None:
            ext.peek_kernel_numbers = peek_kernel_numbers
        return ext

    def __call__(self, inputs, extargs=None):
        pass

class attn_fwd(SdpaCommon):
    EXT_CLASS = FwdExtraArguments

class bwd_kernel_dk_dv(SdpaCommon):
    pass

class bwd_kernel_dq(SdpaCommon):
    pass

class bwd_kernel_fuse(SdpaCommon):
    pass
