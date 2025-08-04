# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from v3python.op import (
    MetroKernel,
    ConditionalKernel,
)
from .ops import OpAttnFwd, OpAttnBwd
from .attn_fwd import attn_fwd
from .bwd_preprocess import (
    bwd_preprocess,
    bwd_preprocess_varlen,
)
from .bwd_postprocess import bwd_postprocess
from .bwd_kernel_dk_dv import bwd_kernel_dk_dv
from .bwd_kernel_dq import bwd_kernel_dq
from .bwd_kernel_fuse import bwd_kernel_fuse
# from .debug_fill_dropout_rng import debug_fill_dropout_rng, debug_fill_dropout_rng_tensor
from .debug_simulate_encoded_softmax import debug_simulate_encoded_softmax
from .aiter_bwd import bwd_dq_dk_dv_v3

SOURCE_FILE = 'tritonsrc/flash.py'

__bwd_preprocess = bwd_preprocess('bwd_preprocess', SOURCE_FILE)
__bwd_preprocess_varlen = bwd_preprocess_varlen('bwd_preprocess_varlen', SOURCE_FILE)
__bwd_postprocess = bwd_postprocess('bwd_postprocess', SOURCE_FILE)
__attn_fwd = attn_fwd('attn_fwd', SOURCE_FILE)
__bwd_kernel_dk_dv = bwd_kernel_dk_dv('bwd_kernel_dk_dv', SOURCE_FILE)
__bwd_kernel_dq = bwd_kernel_dq('bwd_kernel_dq', SOURCE_FILE)
__bwd_kernel_fuse = bwd_kernel_fuse('bwd_kernel_fuse', SOURCE_FILE)
__bwd_aiter = bwd_dq_dk_dv_v3()
# # TODO: Re-implement this as part of kernel(?)
__debug_simulate_encoded_softmax = debug_simulate_encoded_softmax('debug_simulate_encoded_softmax', SOURCE_FILE)

kernels = [
    __bwd_preprocess,
    __bwd_preprocess_varlen,
    __bwd_postprocess,
    __attn_fwd,
    __bwd_kernel_dk_dv,
    __bwd_kernel_dq,
    __bwd_kernel_fuse,
    __debug_simulate_encoded_softmax,
]

affine_kernels = [
    __bwd_aiter,
]

class MetroBwdKernel(MetroKernel):
    FAMILY = OpAttnBwd.FAMILY
    SHARED_IFACE = OpAttnBwd
    ARGUMENTS = OpAttnBwd.ARGUMENTS

class MetroFwdKernel(MetroKernel):
    FAMILY = OpAttnFwd.FAMILY
    SHARED_IFACE = OpAttnFwd
    ARGUMENTS = OpAttnFwd.ARGUMENTS

operators = [
    OpAttnFwd([
        MetroFwdKernel('triton',
                       [__attn_fwd,
                        ConditionalKernel('encoded_softmax', '->data_ptr() != nullptr', __debug_simulate_encoded_softmax)]),
    ]),
    OpAttnBwd([
        MetroBwdKernel('triton_split',
                       [ConditionalKernel('num_seqlens', '> 0', __bwd_preprocess_varlen, __bwd_preprocess),
                        __bwd_kernel_dk_dv,
                        __bwd_kernel_dq]),
        __bwd_kernel_fuse,
        MetroBwdKernel('aiter_asm',
                       [ConditionalKernel('num_seqlens', '> 0', __bwd_preprocess_varlen, __bwd_preprocess),
                        __bwd_aiter,
                        __bwd_postprocess]),
    ]),
]

