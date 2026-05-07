# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from ..flash.kernels import (
    AttnOptionsWrapper,
    SdpaCommon,
    attn_fwd,
    bwd_kernel_dk_dv,
)


class SdpaOpCommon(SdpaCommon):
    BACKEND_COUNT = None  # must define in subclass

    def create_extargs(self, *, which_kernel=None, probe=False):
        backend_index = which_kernel.backend_index if which_kernel is not None else 0
        return self.EXT_CLASS.for_op_backend(backend_index)


class attn_fwd_op(SdpaOpCommon, attn_fwd):
    # kMetro_Triton=0, kSlimAffine_AiterFmhaV3Fwd=1
    BACKEND_COUNT = 2


class attn_bwd_op(SdpaOpCommon, bwd_kernel_dk_dv):
    # kMetro_TritonSplit=0, kShim_BwdKernelFuse=1, kSlimAffine_AiterFmhaV3Bwd=2
    BACKEND_COUNT = 3
