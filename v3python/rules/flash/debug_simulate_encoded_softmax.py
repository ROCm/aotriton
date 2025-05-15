# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import numpy as np
from ._common import FlashKernel, select_pattern, get_possible_choices, OpAttn
from v3python.op import NO_OPERATOR
from .ops import OpAttnFwd

class debug_simulate_encoded_softmax(FlashKernel):
    SHARED_IFACE = OpAttnFwd
    NAME = 'debug_simulate_encoded_softmax'
    ARGUMENTS = [
        'encoded_softmax',
        'stride_rz', 'stride_rh', 'stride_rm', 'stride_rn',
        'dropout_p',
        'Num_head_q', 'Max_seqlen_q', 'Max_seqlen_k',
        "philox_seed_ptr",
        "philox_offset1",
        "philox_offset2",
        'BLOCK_M',  # tl.constexpr starts here
        'BLOCK_N',
    ]
    # Manually copy from OpAttnFwd and override type of encoded_softmax
    # This is Hacking, should be automatically resolved by declaring this
    # kernel assumes ENABLE_DROPOUT=True
    TYPE_CHOICES = {
        frozenset(['encoded_softmax']) : OpAttn.MAIN_DATATYPES,
        frozenset(['dropout_p']) : ['fp32'],
        frozenset(['Num_head_q', 'Max_seqlen_q', 'Max_seqlen_k']) : ['i32'],
        frozenset(['philox_seed_ptr']) : ['*u64'],
        frozenset(['philox_offset1']) : ['*u64'],
        frozenset(['philox_offset2']) : ['u64'],
    }
    PERF_CHOICES = {
        frozenset(['BLOCK_M']) : np.array([64], dtype=np.int16),
        frozenset(['BLOCK_N']) : np.array([32], dtype=np.int16),
    }
    TENSOR_STRIDE_INPUTS = {
        'encoded_softmax' : select_pattern(ARGUMENTS, 'stride_r'),
    }
    TENSOR_RANKS = {
        'encoded_softmax' : 4,
        '_default' : 0,
    }
    DEFAULT_NUM_WARPS=4
    DEFAULT_NUM_STAGES=1

    AUTOTUNE_KEYS = { }
    PARTIALLY_TUNED_FUNCTIONALS = {}
    DOWNGRADER = []
