# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from ._common import FlashKernel, select_pattern

class debug_fill_dropout_rng(FlashKernel):
    ARGUMENTS = [
        'R',
        'stride_rz', 'stride_rh', 'stride_rm', 'stride_rn',
        'seqlen_q', 'seqlen_k',
        'philox_seed',
        'philox_offset_base',
        'BLOCK_M',  # tl.constexpr starts here
        'BLOCK_N',
    ]
    TENSOR_STRIDE_INPUTS = {
        'R' : select_pattern(ARGUMENTS, 'stride_r'),
    }
    TENSOR_RANKS = {
        '_default' : 4,
    }
    TYPE_CHOICES = {
        frozenset(['R']) : ['*fp32:16'],
        frozenset(['seqlen_q', 'seqlen_k']) : ['i32'],
        frozenset(['philox_seed']) : ['u64'],
        frozenset(['philox_offset_base']) : ['u32'],
    }
    FEAT_CHOICES = {
    }
    PERF_CHOICES = {
        frozenset(['BLOCK_M']) : [64],
        frozenset(['BLOCK_N']) : [32],
    }
    DEFAULT_NUM_WARPS=4
    DEFAULT_NUM_STAGES=1
    SHIM_KERNEL_NAME = 'debug_fill_dropout_rng'

    AUTOTUNE_KEYS = { }
    PARTIALLY_TUNED_FUNCTIONALS = [('PADDED_HEAD', None)]
    DOWNGRADER = []
