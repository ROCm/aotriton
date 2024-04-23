# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from ._common import FlashKernel, get_possible_types, select_pattern, BinningLessOrEqual, BinningExact
from .attn_fwd import attn_fwd
from .bwd_kernel_dk_dv import bwd_kernel_dk_dv

class bwd_kernel_dq(FlashKernel):
    ARGUMENTS = [
        'Q', 'K', 'V', 'B', 'sm_scale', 'Out', 'dO',
        'dQ',
        'L', 'D',
        'stride_qz', 'stride_qh', 'stride_qm', 'stride_qk',
        'stride_kz', 'stride_kh', 'stride_kn', 'stride_kk',
        'stride_vz', 'stride_vh', 'stride_vk', 'stride_vn',
        'stride_bz', 'stride_bh', 'stride_bk', 'stride_bn',
        'stride_oz', 'stride_oh', 'stride_om', 'stride_ok',
        'stride_dqz', 'stride_dqh', 'stride_dqm', 'stride_dqk',
        'seqlen_q', 'seqlen_k',
        'head_dim',
        'dropout_p',
        'philox_seed',
        'philox_offset_base',
        'BLOCK_M', # tl.constexpr starts here
        'BLOCK_DMODEL',
        'BLOCK_N',
        'CAUSAL',
        'ENABLE_DROPOUT',
        'PADDED_HEAD',
        'BIAS_TYPE',
    ]
    match_fwd = lambda aname : get_possible_types(attn_fwd, aname)
    match_kv = lambda aname : get_possible_types(bwd_kernel_dk_dv, aname)
    TENSOR_STRIDE_INPUTS = {
        'Q' : select_pattern(ARGUMENTS, 'stride_q'),
        'K' : select_pattern(ARGUMENTS, 'stride_k'),
        'V' : select_pattern(ARGUMENTS, 'stride_v'),
        'B' : select_pattern(ARGUMENTS, 'stride_b'),
        'dO' : select_pattern(ARGUMENTS, 'stride_o'),
        'dQ' : select_pattern(ARGUMENTS, 'stride_dq'),
    }
    TENSOR_RANKS = {
        '_default' : 4,
        'L': 2,
        'D': 2,
    }
    TYPE_CHOICES = {
        frozenset(['Q', 'K', 'V', 'B', 'Out', 'dO', 'dQ']) : match_fwd('Q'),
        frozenset(['sm_scale']) : match_fwd( 'sm_scale'),
        frozenset(['L', 'D']) : ['*fp32:16'],
        frozenset(['seqlen_q', 'seqlen_k']) : ['u64'],
        frozenset(['head_dim']) : ['i32'],
        frozenset(['dropout_p']) : match_fwd('dropout_p'),
        frozenset(['philox_seed']) : match_fwd('philox_seed'),
        frozenset(['philox_offset_base']) : match_fwd('philox_offset_base'),
    }
    FEAT_CHOICES = {
        frozenset(['BLOCK_DMODEL']) : [16, 32, 64, 128, 256],
        frozenset(['CAUSAL']) : match_kv('CAUSAL'),
        frozenset(['ENABLE_DROPOUT']) : match_fwd('ENABLE_DROPOUT'),
        frozenset(['PADDED_HEAD']) : [False, True],
        frozenset(['BIAS_TYPE']) : [0, 1],
    }
    PERF_CHOICES = {
        frozenset(['BLOCK_M']) : match_fwd('BLOCK_M'),
        frozenset(['BLOCK_N']) : match_fwd('BLOCK_N'),
    }
    EXPECTED_IDENTICAL_TENSOR_STRIDES = [
    ]
    DEFAULT_NUM_WARPS=4
    DEFAULT_NUM_STAGES=1
    # TODO: waves_per_eu=1
    SHIM_KERNEL_NAME = 'bwd_kernel_dq'

    AUTOTUNE_KEYS = {
        'seqlen_q' : BinningLessOrEqual,
        'seqlen_k' : BinningLessOrEqual,
    }
    PARTIALLY_TUNED_FUNCTIONALS = [('PADDED_HEAD', None)]
    DOWNGRADER = []
