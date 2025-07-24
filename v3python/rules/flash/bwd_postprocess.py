# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from ._common import FlashKernel, get_possible_choices, select_pattern, BinningLessOrEqual, BinningExact
from .op_attn_bwd import OpAttnBwd

class bwd_postprocess(FlashKernel):
    SHARED_IFACE = OpAttnBwd
    ARGUMENTS = [
        'DQ_ACC', 'DQ',
        'stride_accz', 'stride_acch', 'stride_accm', 'stride_acck',
        'stride_dqz', 'stride_dqh', 'stride_dqm', 'stride_dqk',
        'max_seqlen_q',
        'head_dim',
        'BLOCK_M',          # tl.constexpr starts here
        'BLOCK_DMODEL',     # TODO: Rename the triton kernel
        'PADDED_HEAD',
    ]
    PERF_CHOICES = {
        frozenset(['BLOCK_M']) : [128], # TODO: All possible values?
    }
    # bwd_postprocess is only needed by aiter kernel and thus no need to support fp32
    # Note: if AITER ASM added FP32 support later,
    # ConditionalTensor('Q', '*fp32:16', 'DQ', 'DQ_ACC') should be used for
    # DQ_ACC of AITER ASM kernel with FP32 supported.
    CHOICE_FILTERS = {
        'Q' : lambda dtype : 'fp16' in dtype or 'bf16' in dtype,
    }
    DEFAULT_NUM_WARPS=4
    DEFAULT_NUM_STAGES=1
    NAME = 'bwd_postprocess'

    AUTOTUNE_KEYS = { }
    PARTIALLY_TUNED_FUNCTIONALS = {
        'PADDED_HEAD': False,
    }
    DOWNGRADER = []
