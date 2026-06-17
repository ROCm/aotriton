# Copyright © 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Flash attention kernel classes for tuning — composes kernel_calls and kernel_opts.

Provides the full tuning-library kernel classes (with KernelControl / attn_options)
under their original names for backward compatibility.
"""

from .kernel_calls import (
    NAN,
    eager_null_dq_acc,
    eager_delta,
    SdpaCalls,
    attn_fwd as _attn_fwd,
    bwd_kernel_dk_dv as _bwd_kernel_dk_dv,
    bwd_kernel_dq as _bwd_kernel_dq,
    bwd_kernel_fuse as _bwd_kernel_fuse,
)
from .kernel_opts import AttnOptionsWrapper, SdpaOpts


class SdpaCommon(SdpaOpts, SdpaCalls):
    pass


class attn_fwd(SdpaOpts, _attn_fwd):
    BACKEND_INDEX = 0


class bwd_kernel_dk_dv(SdpaOpts, _bwd_kernel_dk_dv):
    BACKEND_INDEX = 0


class bwd_kernel_dq(SdpaOpts, _bwd_kernel_dq):
    BACKEND_INDEX = 0


class bwd_kernel_fuse(SdpaOpts, _bwd_kernel_fuse):
    BACKEND_INDEX = 1
