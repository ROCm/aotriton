# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import itertools
from ._common import (
    FlashKernel,
    get_possible_choices,
    select_pattern,
    BinningLessOrEqual,
    BinningExact,
    Config
)
from .attn_fwd import attn_fwd

class bwd_kernel_dk_dv(FlashKernel):
    ARGUMENTS = [
        'Q', 'K', 'V', 'B', 'sm_scale', 'Out', 'DO',
        'DK', 'DV',
        'L', 'D',
        'stride_qz', 'stride_qh', 'stride_qm', 'stride_qk',
        'stride_kz', 'stride_kh', 'stride_kn', 'stride_kk',
        'stride_vz', 'stride_vh', 'stride_vk', 'stride_vn',
        'stride_bz', 'stride_bh', 'stride_bk', 'stride_bn',
        'stride_oz', 'stride_oh', 'stride_om', 'stride_ok',
        'stride_dkz', 'stride_dkh', 'stride_dkn', 'stride_dkk',
        'stride_dvz', 'stride_dvh', 'stride_dvk', 'stride_dvn',
        'num_head_q',
        'num_head_k',
        'cu_seqlens_q',
        'cu_seqlens_k',
        'num_seqlens',
        'max_seqlen_q',
        'max_seqlen_k',
        'head_dim',
        'dropout_p',
        'philox_seed_ptr',
        'philox_offset1',
        'philox_offset2',
        'BLOCK_M', # tl.constexpr starts here
        'BLOCK_DMODEL',
        'BLOCK_N',
        'CAUSAL',
        'ENABLE_DROPOUT',
        'PADDED_HEAD',
        'BIAS_TYPE',
    ]
    match_fwd = lambda aname : get_possible_choices(attn_fwd, aname)
    TENSOR_STRIDE_INPUTS = {
        'Q' : select_pattern(ARGUMENTS, 'stride_q'),
        'K' : select_pattern(ARGUMENTS, 'stride_k'),
        'V' : select_pattern(ARGUMENTS, 'stride_v'),
        'B' : select_pattern(ARGUMENTS, 'stride_b'),
        'DO' : select_pattern(ARGUMENTS, 'stride_o'),
        'DK' : select_pattern(ARGUMENTS, 'stride_dk'),
        'DV' : select_pattern(ARGUMENTS, 'stride_dv'),
    }
    TENSOR_RANKS = {
        '_default' : 4,
        'L': 2,
        'D': 2,
        'cu_seqlens_q': 1,
        'cu_seqlens_k': 1,
        'philox_seed_ptr': 0,
        'philox_offset1': 0,
    }
    TYPE_CHOICES = {
        frozenset(['Q', 'K', 'V', 'B', 'Out', 'DO', 'DK', 'DV']) : match_fwd('Q'),
        frozenset(['sm_scale']) : match_fwd( 'Sm_scale'),
        frozenset(['L', 'D']) : ['*fp32:16'],
        frozenset(['cu_seqlens_q', 'cu_seqlens_k']) : match_fwd('cu_seqlens_q'),
        frozenset(['num_seqlens', 'max_seqlen_q', 'max_seqlen_k']) : match_fwd('Num_seqlens'),
        frozenset(['head_dim', 'num_head_q', 'num_head_k']) : ['i32'],
        frozenset(['dropout_p']) : match_fwd('dropout_p'),
        frozenset(['philox_seed_ptr']) : match_fwd('philox_seed_ptr'),
        frozenset(['philox_offset1']) : match_fwd('philox_offset1'),
        frozenset(['philox_offset2']) : match_fwd('philox_offset2'),
    }
    FEAT_CHOICES = {
        frozenset(['BLOCK_DMODEL']) : match_fwd('BLOCK_DMODEL'),
        frozenset(['CAUSAL']) : [True, False],
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
    SHIM_KERNEL_NAME = 'bwd_kernel_dk_dv'

    AUTOTUNE_KEYS = {
        'max_seqlen_q' : BinningLessOrEqual,
        'max_seqlen_k' : BinningLessOrEqual,
    }
    PARTIALLY_TUNED_FUNCTIONALS = [('PADDED_HEAD', False)]
    DOWNGRADER = []

    @staticmethod
    def gen_autotune_configs(gpu, fsel_dict : 'dict[str, Any]'):
        dtype = fsel_dict['Q']
        HEAD_DIM = fsel_dict['BLOCK_DMODEL']
        MI = 'MI' in gpu
        Navi = 'Navi' in gpu or gpu.startswith('RX')
        ret = []
        # TODO: right sizes for fp32?
        BLOCK_SIZES = [16, 32, 64] if dtype != '*fp32:16' else [16, 32]
        WAVES_PER_EU = [1, 2, 3, 4]
        NUM_WARPS = [2, 4]
        NUM_STAGES = [1]
        for M, N, waves, warps, stages in itertools.product(BLOCK_SIZES,
                                                            BLOCK_SIZES,
                                                            WAVES_PER_EU,
                                                            NUM_WARPS,
                                                            NUM_STAGES):
            if M < N:
                continue  # deduplicate
            if MI and M == 64 and N == 64 and warps == 4:
                continue  # No optimal kernel according to 0.8b tuning db
            if HEAD_DIM >= 512 and M == 64 and N == 64 and warps == 1:
                continue  # Timeout
            if Navi and M * N >= 32 * 32 and warps < 4:
                continue  # Timeout
            if Navi and M * N >= 32 * 16 and warps < 2:
                continue  # Timeout
            kw = {'BLOCK_M': M, 'BLOCK_N': N, 'waves_per_eu': waves}
            yield Config(kw, num_stages=stages, num_warps=warps)
