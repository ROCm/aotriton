# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import itertools
import os
import numpy as np
from ._common import (
    FlashKernel,
    select_pattern,
    BinningLessOrEqual,
    BinningExact,
    Config,
    ConditionalConstexpr as CC,
    ConditionalDeferredConstexpr as CDC,
    ConditionalDeferredElseTensor as CDETensor,
)

AOTRITON_FLASH_BLOCK_DMODEL = os.getenv('AOTRITON_FLASH_BLOCK_DMODEL', default='16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 512')
AOTRITON_FLASH_BLOCK_DMODEL = [int(d) for d in AOTRITON_FLASH_BLOCK_DMODEL.split(',')]

_IF_DROPOUT = lambda elsechoice : [CC('ENABLE_DROPOUT', False, 0, elsechoice)]
_IF_CAUSAL = lambda elsechoice, dtype=None : [CC('CAUSAL_TYPE', False, 0, elsechoice, else_dtype=dtype)]

class attn_fwd(FlashKernel):
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
    TYPE_CHOICES = {
        frozenset(['Q', 'K', 'V', 'Out']) : FlashKernel.MAIN_DATATYPES,
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
        frozenset(['persistent_atomic_counter']) : _IF_CAUSAL('*i32'),
        frozenset(['Num_CU', 'Batch']) : ['i32'],
    }
    FEAT_CHOICES = {
        frozenset(["Q_descale", "K_descale", "P_scale", "P_descale", "V_descale"]) : [0],  # INT8 For the future
        # Can support CAUSAL_TYPE = 2 (Bottom right alignment) but this will
        # further increse the number of kernels. Will be added later along with
        # windowed attention
        frozenset(['CAUSAL_TYPE']) : [0, 1],
        frozenset(['BLOCK_DMODEL']) : AOTRITON_FLASH_BLOCK_DMODEL,
        frozenset(['ENABLE_DROPOUT']) : [False, True],
        frozenset(['RETURN_ENCODED_SOFTMAX']) : [False],
        frozenset(['PADDED_HEAD']) : [False, True],
        frozenset(['BIAS_TYPE']) : [0, 1],
        frozenset(['USE_ALIBI']) : [False],
        frozenset(['INT8', 'INT8_KV', 'USE_P_SCALE']) : [False],  # INT8 for the future
    }
    PERF_CHOICES = {
        frozenset(['PERSISTENT_TYPE']) : _IF_CAUSAL(2, dtype='i8'),
        frozenset(['GRID_CU_MULTIP']) : np.array([2], dtype=np.int8),  # NOTE: use np.array with dtype to reduce size of the generate tuning infomation struct
        frozenset(['BLOCK_M']) : np.array([16], dtype=np.int16),
        frozenset(['BLOCK_N']) : np.array([16], dtype=np.int16),
        frozenset(['PRE_LOAD_V']) : [False], # [False, True],
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
    EXPECTED_IDENTICAL_TENSOR_STRIDES = [
        # Not needed stride_o* exist
    ]
    SHIM_KERNEL_NAME = 'attn_fwd'

    # AUTOTUNE_KEYS can have Functional choices, which will be discarded later
    AUTOTUNE_KEYS = {
        'Max_seqlen_q' : BinningLessOrEqual,
        'Max_seqlen_k' : BinningLessOrEqual,
        'CAUSAL_TYPE' : BinningExact,
        'ENABLE_DROPOUT' : BinningExact,
    }
    # List of functionals that are not fully tuned in the tuning database
    # First element of the tuple is name. Second is the value to use instead
    PARTIALLY_TUNED_FUNCTIONALS = [
        ('RETURN_ENCODED_SOFTMAX', False),
        ('PADDED_HEAD', False),
        ('BIAS_TYPE', None)
    ]

    # Python Trick: do not use @staticmethod, and also do not add 'self', and
    #               then there is no need to prefix the classname in DOWNGRADER list
    def DOWNGRADE_RETURN_ENCODED_SOFTMAX(tuned_kernel, compiler_options):
        return
        '''
        # tuned_kernel['BLOCK_M'] //= 2
        # tuned_kernel['BLOCK_N'] //= 2
        square = min(tuned_kernel['BLOCK_M'], tuned_kernel['BLOCK_N'])
        tuned_kernel['BLOCK_M'] = square // 2
        tuned_kernel['BLOCK_N'] = square // 2
        tuned_kernel['pre_load_v'] = False
        tuned_kernel['waves_per_eu'] = 0
        compiler_options['num_stages'] = 2
        '''

    DOWNGRADER = [(('RETURN_ENCODED_SOFTMAX', True), DOWNGRADE_RETURN_ENCODED_SOFTMAX)]

    @staticmethod
    def gen_autotune_configs(gpu : str, fsel_dict : 'dict[str, Any]'):
        dtype = fsel_dict['Q']
        HEAD_DIM = fsel_dict['BLOCK_DMODEL']
        CAUSAL_TYPE = fsel_dict['CAUSAL_TYPE']
        ret = []
        MI = 'MI' in gpu
        Navi = 'Navi' in gpu or gpu.startswith('RX')
        if MI:
            BLOCK_SIZES = [(32, 16), (128, 64), (64, 64), (64, 32), (128, 128)]
        elif Navi:
            BLOCK_SIZES = [(64, 32), (32, 32), (32, 16)]
            if '*fp32' not in dtype:
                BLOCK_SIZES += [(16, 16)]
            else:
                # M //= 2 will effectively yield (16,32), (16,16)
                pass
        WAVES_PER_EU = [1, 2, 3, 4]
        NUM_WARPS = [2, 4]
        PRE_LOAD_V = [False]
        NUM_STAGES = [1]
        for (M, N), waves, warps, stages, pre in itertools.product(BLOCK_SIZES,
                                                                   WAVES_PER_EU,
                                                                   NUM_WARPS,
                                                                   NUM_STAGES,
                                                                   PRE_LOAD_V):
            if warps == 1 and M * N >= 64 * 128:
                continue  # Timeout
            if stages == 2 and M * N >= 64 * 32:
                continue  # Timeout
            if Navi and HEAD_DIM == 256 and stages == 2:
                continue  # Timeout
            if HEAD_DIM >= 512 and M == 128 and N == 128 and warps == 2:
                continue  # Timeout
            if dtype == '*fp32:16':
                M //= 2
            if M < N:  # Faulty or duplicate
                continue
            if MI and M > 32 and N > 16 and stages == 2:
                continue  # No optimal kernel according to 0.8b tuning db
            if MI and M > 64 and N > 64 and warps == 1:
                continue  # No optimal kernel according to 0.8b tuning db
            if Navi and M > 32 and N > 32 and stages == 2:
                continue  # No optimal kernel according to 0.8b tuning db
            if Navi and M > 32 and N > 32 and warps == 1:
                continue  # No optimal kernel according to 0.8b tuning db
            persistent_type = 2 if CAUSAL_TYPE != 0 else 0
            kw = { 'PERSISTENT_TYPE' : persistent_type,
                   'GRID_CU_MULTIP': 2,
                   'BLOCK_M': M,
                   'BLOCK_N': N,
                   'waves_per_eu': waves,
                   'PRE_LOAD_V': pre,
                 }
            # TODO: Add Dyamic PERSISTENT_TYPE IFF causal is enabled to tuning database
            yield Config(kw, num_stages=stages, num_warps=warps)
        if MI:
            pass
            # Covered in general logic above
            # yield from [
            #     Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'waves_per_eu': 2, 'pre_load_v': False}, num_stages=1, num_warps=4),
            #     Config({'BLOCK_M': 128, 'BLOCK_N':  64, 'waves_per_eu': 2, 'pre_load_v': False}, num_stages=1, num_warps=4),
            #     Config({'BLOCK_M': 128, 'BLOCK_N':  64, 'waves_per_eu': 3, 'pre_load_v': False}, num_stages=1, num_warps=4),
            #     Config({'BLOCK_M': 128, 'BLOCK_N':  64, 'waves_per_eu': 1, 'pre_load_v': False}, num_stages=1, num_warps=4),
            #     Config({'BLOCK_M': 128, 'BLOCK_N':  32, 'waves_per_eu': 2, 'pre_load_v': False}, num_stages=1, num_warps=4),
            # ]
        elif Navi:
            pass
            # Covered in general logic above
            # yield from [
            #     Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'waves_per_eu': 4, 'pre_load_v': False}, num_stages=1, num_warps=2),
            #     Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'waves_per_eu': 2, 'pre_load_v': False}, num_stages=1, num_warps=2),
            #     Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'waves_per_eu': 4, 'pre_load_v': False}, num_stages=1, num_warps=2),
            #     Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'waves_per_eu': 2, 'pre_load_v': False}, num_stages=1, num_warps=2),
            #     Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'waves_per_eu': 4, 'pre_load_v': False}, num_stages=1, num_warps=2),
            #     Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'waves_per_eu': 2, 'pre_load_v': False}, num_stages=1, num_warps=2),
            # ]
        else:
            assert False, f'Unknown GPU {gpu}'  # Sanity check, should be removed in the future
