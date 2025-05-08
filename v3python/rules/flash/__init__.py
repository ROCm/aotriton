# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from v3python.op import MetroKernel
from .ops import OpAttnFwd # , OpAttnBwd
from .attn_fwd import attn_fwd
# from .bwd_preprocess import bwd_preprocess, bwd_preprocess_varlen
# from .bwd_kernel_dk_dv import bwd_kernel_dk_dv
# from .bwd_kernel_dq import bwd_kernel_dq
# from .bwd_kernel_fuse import bwd_kernel_fuse
# from .debug_fill_dropout_rng import debug_fill_dropout_rng, debug_fill_dropout_rng_tensor
from .debug_simulate_encoded_softmax import debug_simulate_encoded_softmax

SOURCE_FILE = 'tritonsrc/flash.py'

# __bwd_preprocess = bwd_preprocess('bwd_preprocess', SOURCE_FILE)
# __bwd_preprocess_varlen = bwd_preprocess_varlen('bwd_preprocess_varlen', SOURCE_FILE)
__attn_fwd = attn_fwd('attn_fwd', SOURCE_FILE)
# __bwd_kernel_dk_dv = bwd_kernel_dk_dv('bwd_kernel_dk_dv', SOURCE_FILE)
# __bwd_kernel_dq = bwd_kernel_dq('bwd_kernel_dq', SOURCE_FILE)
# __bwd_kernel_fuse = bwd_kernel_fuse('bwd_kernel_fuse', SOURCE_FILE)
# # TODO: aiter
# # __bwd_aiter = bwd_aiter()
# # TODO: Remove them from code base
# # debug_fill_dropout_rng(__debug_fill_dropout_rng', SOURCE_FILE)
# # debug_fill_dropout_rng_tensor(__debug_fill_dropout_rng_tensor', SOURCE_FILE)
# # TODO: Re-implement this as part of kernel(?)
__debug_simulate_encoded_softmax = debug_simulate_encoded_softmax('debug_simulate_encoded_softmax', SOURCE_FILE)

kernels = [
    #     __bwd_preprocess,
    #     __bwd_preprocess_varlen,
    __attn_fwd,
    #     __bwd_kernel_dk_dv,
    #     __bwd_kernel_dq,
    #     __bwd_kernel_fuse,
    __debug_simulate_encoded_softmax,
]

class MetroBwdKernel(MetroKernel):
    FAMILY = OpAttnBwd.FAMILY
    SHARED_IFACE = OpAttnBwd
    ARGUMENTS = OpAttnBwd.ARGUMENTS

operators = [
    OpAttnFwd([
        __attn_fwd,
    ]),
    OpAttnBwd([
        MetroBwdKernel('triton_split',
                       [__bwd_preprocess,
                        __bwd_kernel_dk_dv,
                        __bwd_kernel_dq]),
    ]),
    # [
    #     MetroKernel('triton', [__attn_fwd], is_fallback=True),
    # ]),
    #     OpAttnBwd([
    #         MetroKernel('triton_split',
    #                     [__bwd_preprocess,
    #                      __bwd_kernel_dk_dv,
    #                      __bwd_kernel_dq],
    #                      is_fallback=True),
    #         MetroKernel('triton_fully_fused',
    #                      [__bwd_preprocess,
    #                       __bwd_kernel_fuse]),
    #         # TODO, Affine Kernels
    #         # MetroAttnBwd('aiter_semi_fused',
    #         #             [__bwd_preprocess,
    #         #               __bwd_aiter]),
    #     ]),
]
    # # TODO: Merge __bwd_preprocess_varlen into __bwd_preprocess
    #         So we don't need OpAttnBwdVarlen
    # 'AttnBwdVarlen' : [
    #     MetroAttnBwd('triton_split',
    #                  [__bwd_preprocess_varlen,
    #                   __bwd_kernel_dk_dv,
    #                   __bwd_kernel_dq],
    #                   is_fallback=True]),
    # ]
