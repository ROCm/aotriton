# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from ._common import FlashKernel, get_possible_types, select_pattern, BinningLessOrEqual, BinningExact
from .attn_fwd import attn_fwd

class bwd_preprocess(FlashKernel):
    ARGUMENTS = [
        'Out', 'DO',
        'Delta',
        'stride_oz', 'stride_oh', 'stride_om', 'stride_on',
        'stride_doz', 'stride_doh', 'stride_dom', 'stride_don',
        'seqlen_q',
        'head_dim',
        'BLOCK_M', # tl.constexpr starts here
        'D_HEAD',
        'PADDED_HEAD',
    ]
    TENSOR_STRIDE_INPUTS = {
        'Out' : select_pattern(ARGUMENTS, 'stride_o'),
        'DO' : select_pattern(ARGUMENTS, 'stride_do'),
    }
    TENSOR_RANKS = {
        '_default' : 4,
        'Delta' : 2,
    }
    TYPE_CHOICES = {
        frozenset(['Out', 'DO']) : ['*fp16:16', '*bf16:16'],
        frozenset(['Delta']) : ['*fp32:16'],
        frozenset(['seqlen_q']) : ['u64'],
        frozenset(['head_dim']) : ['i32'],
    }
    FEAT_CHOICES = {
        frozenset(['D_HEAD']) : get_possible_types(attn_fwd, 'BLOCK_DMODEL'),
        frozenset(['PADDED_HEAD']) : [False, True],
    }
    PERF_CHOICES = {
        frozenset(['BLOCK_M']) : [128], # TODO: All possible values?
    }
    DEFAULT_NUM_WARPS=4
    DEFAULT_NUM_STAGES=1
    SHIM_KERNEL_NAME = 'bwd_preprocess'

    AUTOTUNE_KEYS = { }
    PARTIALLY_TUNED_FUNCTIONALS = [('PADDED_HEAD', None)]
    DOWNGRADER = []
