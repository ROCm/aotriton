# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from .op_attn_fwd import OpAttnFwd
from ._common import (
    OpAttn,
    select_pattern,
    get_possible_choices,
    BinningLessOrEqual,
    BinningExact,
    ConditionalConstexpr as CC,
    ConditionalDeferredConstexpr as CDC,
    ConditionalDeferredElseTensor as CDETensor,
)

class OpAttnBwd(OpAttn):
    NAME = 'op_attn_bwd'
    ARGUMENTS = [
        'Q', 'K', 'V', 'B', 'sm_scale', 'Out', 'DO',
        'DK', 'DV', 'DQ', 'DB', 'DQ_ACC',
        'L', 'D',
        'stride_qz', 'stride_qh', 'stride_qm', 'stride_qk',
        'stride_kz', 'stride_kh', 'stride_kn', 'stride_kk',
        'stride_vz', 'stride_vh', 'stride_vk', 'stride_vn',
        'stride_bz', 'stride_bh', 'stride_bk', 'stride_bn',
        'stride_oz', 'stride_oh', 'stride_om', 'stride_ok',
        'stride_doz', 'stride_doh', 'stride_dom', 'stride_dok',
        'stride_dkz', 'stride_dkh', 'stride_dkn', 'stride_dkk',
        'stride_dvz', 'stride_dvh', 'stride_dvk', 'stride_dvn',
        'stride_dqz', 'stride_dqh', 'stride_dqm', 'stride_dqk',
        'stride_dbz', 'stride_dbh', 'stride_dbm', 'stride_dbn',
        'stride_accz', 'stride_acch', 'stride_accm', 'stride_acck',
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
        'Window_left', 'Window_right',
        'BLOCK_M', # tl.constexpr starts here
        'BLOCK_DMODEL',
        'BLOCK_N',
        'CAUSAL_TYPE',
        'ENABLE_DROPOUT',
        'PADDED_HEAD',
        'BIAS_TYPE',
    ]
    TENSOR_STRIDE_INPUTS = {
        'Q' : select_pattern(ARGUMENTS, 'stride_q'),
        'K' : select_pattern(ARGUMENTS, 'stride_k'),
        'V' : select_pattern(ARGUMENTS, 'stride_v'),
        'B' : select_pattern(ARGUMENTS, 'stride_b', delete_when=('BIAS_TYPE', 0)),
        'Out' : select_pattern(ARGUMENTS, 'stride_o'),
        'DO' : select_pattern(ARGUMENTS, 'stride_do'),
        'DK' : select_pattern(ARGUMENTS, 'stride_dk'),
        'DV' : select_pattern(ARGUMENTS, 'stride_dv'),
        'DQ' : select_pattern(ARGUMENTS, 'stride_dq'),
        'DB' : select_pattern(ARGUMENTS, 'stride_db', delete_when=('BIAS_TYPE', 0)),
        'DQ_ACC' : select_pattern(ARGUMENTS, 'stride_acc'),
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
    match_fwd = lambda aname : get_possible_choices(OpAttnFwd, aname)
    TYPE_CHOICES = {
        frozenset(['Q', 'K', 'V', 'B', 'Out', 'DO', 'DK', 'DV', 'DQ', 'DB']) : match_fwd('Q'),
        frozenset(['sm_scale']) : match_fwd( 'Sm_scale'),
        frozenset(['L']) : ['*fp32:16'],
        frozenset(['D', 'DQ_ACC']) : ['LazyTensor:*fp32:16'],
        frozenset(['cu_seqlens_q', 'cu_seqlens_k']) : match_fwd('cu_seqlens_q'),
        frozenset(['num_seqlens', 'max_seqlen_q', 'max_seqlen_k']) : match_fwd('Num_seqlens'),
        frozenset(['head_dim', 'num_head_q', 'num_head_k']) : ['i32'],
        frozenset(['dropout_p']) : match_fwd('dropout_p'),
        frozenset(['philox_seed_ptr']) : match_fwd('philox_seed_ptr'),
        frozenset(['philox_offset1']) : match_fwd('philox_offset1'),
        frozenset(['philox_offset2']) : match_fwd('philox_offset2'),
        frozenset(['Window_left', 'Window_right']) : match_fwd('Window_left'),
    }
    FEAT_CHOICES = {
        frozenset(['BLOCK_DMODEL']) : match_fwd('BLOCK_DMODEL'),
        frozenset(['CAUSAL_TYPE']) : [0, 3],
        frozenset(['ENABLE_DROPOUT']) : match_fwd('ENABLE_DROPOUT'),
        frozenset(['PADDED_HEAD']) : [False, True],
        frozenset(['BIAS_TYPE']) : [0, 1],
    }
    OPTUNE_KEYS = {
        'max_seqlen_q' : BinningLessOrEqual,
        'max_seqlen_k' : BinningLessOrEqual,
    }
