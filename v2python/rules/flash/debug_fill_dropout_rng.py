# Copyright © 2024-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from ._common import FlashKernel, select_pattern, get_possible_choices

class debug_fill_dropout_rng(FlashKernel):
    ARGUMENTS = [
        'R',
        'stride_rz', 'stride_rh', 'stride_rm', 'stride_rn',
        'seqlen_q', 'seqlen_k',
        'philox_seed',
        'philox_offset',
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
        frozenset(['R']) : ['*fp32:16', '*i32:16'],
        frozenset(['seqlen_q', 'seqlen_k']) : ['i32'],
        frozenset(['philox_seed']) : ['u64'],
        frozenset(['philox_offset']) : ['u64'],
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

class debug_fill_dropout_rng_tensor(debug_fill_dropout_rng):
    ARGUMENTS = [
        'R',
        'stride_rz', 'stride_rh', 'stride_rm', 'stride_rn',
        'seqlen_q', 'seqlen_k',
        'philox_seed_ptr',
        'philox_offset_base_ptr',
        'BLOCK_M',  # tl.constexpr starts here
        'BLOCK_N',
    ]
    TYPE_CHOICES = {
        frozenset(['R']) : get_possible_choices(debug_fill_dropout_rng, 'R'),
        frozenset(['seqlen_q', 'seqlen_k']) : ['i32'],
        frozenset(['philox_seed_ptr']) : ['*u64'],
        frozenset(['philox_offset_base_ptr']) : ['*u64'],
    }
    TENSOR_RANKS = {
        '_default' : 4,
        'philox_seed_ptr': 0,
        'philox_offset_base_ptr': 0,
    }
    SHIM_KERNEL_NAME = 'debug_fill_dropout_rng_tensor'
