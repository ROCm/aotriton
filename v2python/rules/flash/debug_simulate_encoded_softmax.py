# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from ._common import FlashKernel, select_pattern, get_possible_choices

class debug_simulate_encoded_softmax(FlashKernel):
    ARGUMENTS = [
        'R',
        'stride_rz', 'stride_rh', 'stride_rm', 'stride_rn',
        'dropout_p', 'Num_head_q', 'Max_seqlen_q', 'Max_seqlen_k',
        "philox_seed_ptr",
        "philox_offset1",
        "philox_offset2",
        'BLOCK_M',  # tl.constexpr starts here
        'BLOCK_N',
    ]
    TENSOR_STRIDE_INPUTS = {
        'R' : select_pattern(ARGUMENTS, 'stride_r'),
    }
    TENSOR_RANKS = {
        'R' : 4,
        '_default' : 0,
    }
    TYPE_CHOICES = {
        frozenset(['R']) : FlashKernel.MAIN_DATATYPES,
        frozenset(['dropout_p']) : ['fp32'],
        frozenset(['Num_head_q', 'Max_seqlen_q', 'Max_seqlen_k']) : ['i32'],
        frozenset(['philox_seed_ptr']) : ['*u64'],
        frozenset(['philox_offset1']) : ['*u64'],
        frozenset(['philox_offset2']) : ['u64'],
    }
    FEAT_CHOICES = {
    }
    PERF_CHOICES = {
        frozenset(['BLOCK_M']) : [64],
        frozenset(['BLOCK_N']) : [32],
    }
    DEFAULT_NUM_WARPS=4
    DEFAULT_NUM_STAGES=1
    SHIM_KERNEL_NAME = 'debug_simulate_encoded_softmax'

    AUTOTUNE_KEYS = { }
    PARTIALLY_TUNED_FUNCTIONALS = []
    DOWNGRADER = []
