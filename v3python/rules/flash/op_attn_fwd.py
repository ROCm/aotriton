# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
from ._common import (
    OpAttn,
    select_pattern,
    BinningLessOrEqual,
    BinningExact,
    ConditionalConstexpr as CC,
    ConditionalDeferredConstexpr as CDC,
    ConditionalDeferredElseTensor as CDETensor,
)

AOTRITON_FLASH_BLOCK_DMODEL = os.getenv('AOTRITON_FLASH_BLOCK_DMODEL', default='16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 512')
AOTRITON_FLASH_BLOCK_DMODEL = [int(d) for d in AOTRITON_FLASH_BLOCK_DMODEL.split(',')]

_IF_DROPOUT = lambda elsechoice : [CC('ENABLE_DROPOUT', False, 0, elsechoice)]
_IF_CAUSAL = lambda elsechoice : [CC('CAUSAL_TYPE', 0, 0, elsechoice)]
'''
_IF_SLIDING_WINDOW:
    If CAUSAL_TYPE != 3: set param as constexpr(0) else: set to elsechoice
'''
_IF_SLIDING_WINDOW = lambda elsechoice : [CC('CAUSAL_TYPE', 3, 0, elsechoice,
                                             cond_op=lambda tc_value, _: tc_value != 3)]

class OpAttnFwd(OpAttn):
    NAME = 'op_attn_fwd'
    ARGUMENTS = [
        # Basic SDPA
        "Q", "K", "V", "B", "A", "Sm_scale", "L", "Out",
        "Q_descale", "K_descale", "P_scale", "P_descale", "V_descale",
        "stride_qz", "stride_qh", "stride_qm", "stride_qk",
        "stride_kz", "stride_kh", "stride_kn", "stride_kk",
        "stride_vz", "stride_vh", "stride_vk", "stride_vn",
        "stride_oz", "stride_oh", "stride_om", "stride_on",
        "stride_bz", "stride_bh", "stride_bm", "stride_bn",
        "stride_az", "stride_ah",
        # MQA/GQA
        "Num_head_q",
        "Num_head_k",
        # Varlen
        "Num_seqlens",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "Max_seqlen_q",
        "Max_seqlen_k",
        # Head Dimensions
        "BLOCK_DMODEL",
        "Head_dim",
        "PADDED_HEAD",
        # dropout and PRNG
        "ENABLE_DROPOUT",
        "dropout_p",
        "philox_seed_ptr",
        "philox_offset1",
        "philox_offset2",
        "philox_seed_output",
        "philox_offset_output",
        "RETURN_ENCODED_SOFTMAX",
        "encoded_softmax",
        # causal, (Planned Feature) windowed attention
        "CAUSAL_TYPE",
        "Window_left",
        "Window_right",
        # bias
        "BIAS_TYPE",
        # alibi
        "USE_ALIBI",
        # INT8
        "INT8",
        "INT8_KV",
        "USE_P_SCALE",
        # Persistent related arguments
        "PERSISTENT_TYPE",
        "persistent_atomic_counter",
        "Num_CU",
        "GRID_CU_MULTIP",
        "Batch",
        # Performance
        "BLOCK_M",
        "BLOCK_N",
        "PRE_LOAD_V",
    ]
    TENSOR_STRIDE_INPUTS = {
        'Q' : select_pattern(ARGUMENTS, 'stride_q'),
        'K' : select_pattern(ARGUMENTS, 'stride_k'),
        'V' : select_pattern(ARGUMENTS, 'stride_v'),
        'B' : select_pattern(ARGUMENTS, 'stride_b', delete_when=('BIAS_TYPE', 0)),
        'A' : select_pattern(ARGUMENTS, 'stride_a', delete_when=('USE_ALIBI', False)),
        'Out' : select_pattern(ARGUMENTS, 'stride_o'),
    }
    TENSOR_RANKS = {
        '_default' : 4,
        'A': 2,
        'L': 2,
        'cu_seqlens_q': 1,
        'cu_seqlens_k': 1,
        'philox_seed_ptr': 0,
        'philox_offset1': 0,
        'philox_seed_output': 0,
        'philox_offset_output': 0,
        'persistent_atomic_counter': 0,
    }
    TYPE_CHOICES = {
        frozenset(['Q', 'K', 'V', 'Out']) : OpAttn.MAIN_DATATYPES,
        frozenset(['B']) : [CDETensor('BIAS_TYPE', 0, 0, 'Q')],
        frozenset(['A']) : [CDETensor('USE_ALIBI', False, 0, 'Q')],
        frozenset(['encoded_softmax']) : [CDETensor('RETURN_ENCODED_SOFTMAX', False, 0, 'Q')],
        frozenset(['Sm_scale']) : ['fp32'],
        frozenset(['L']) : ['*fp32:16'],
        frozenset(['cu_seqlens_q', 'cu_seqlens_k']) : ['*i32:16'],
        frozenset(['Num_head_q', 'Num_head_k', 'Num_seqlens', 'Max_seqlen_q', 'Max_seqlen_k']) : ['i32'],
        frozenset(['Head_dim']) : [CDC('PADDED_HEAD', False, 'BLOCK_DMODEL', 'i32')],
        frozenset(['dropout_p']) : _IF_DROPOUT('fp32'),
        frozenset(['philox_seed_ptr', 'philox_seed_output', 'philox_offset_output']) : _IF_DROPOUT('*u64'),
        frozenset(['philox_offset1']) : _IF_DROPOUT('*u64'),
        frozenset(['philox_offset2']) : _IF_DROPOUT('u64'),
        frozenset(['Window_left', 'Window_right']) : _IF_SLIDING_WINDOW('i32'),
        frozenset(['persistent_atomic_counter']) : _IF_CAUSAL('*i32'),
        frozenset(['Num_CU', 'Batch']) : ['i32'],
    }
    FEAT_CHOICES = {
        frozenset(["Q_descale", "K_descale", "P_scale", "P_descale", "V_descale"]) : [0],  # INT8 For the future
        # Can support CAUSAL_TYPE = 2 (Bottom right alignment) but this will
        # further increse the number of kernels.
        # Bottom right alignment is supported through
        # windowed attention (CAUSAL_TYPE=3)
        # Note:
        #   Triton source supports [0,1,2,3] but this class is to guide build
        #   of AOTriton. So only [0, 3] are kept.
        frozenset(['CAUSAL_TYPE']) : [0, 3],
        frozenset(['BLOCK_DMODEL']) : AOTRITON_FLASH_BLOCK_DMODEL,
        frozenset(['ENABLE_DROPOUT']) : [False, True],
        frozenset(['RETURN_ENCODED_SOFTMAX']) : [False],
        frozenset(['PADDED_HEAD']) : [False, True],
        frozenset(['BIAS_TYPE']) : [0, 1],
        frozenset(['USE_ALIBI']) : [False],
        frozenset(['INT8', 'INT8_KV', 'USE_P_SCALE']) : [False],  # INT8 for the future
    }
    OPTUNE_KEYS = {
        'Max_seqlen_q' : BinningLessOrEqual,
        'Max_seqlen_k' : BinningLessOrEqual,
    }
    PARTIALLY_TUNED_FUNCTIONALS = {
        'PADDED_HEAD': False,
    }
