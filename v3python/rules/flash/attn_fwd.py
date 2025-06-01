# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import itertools
import numpy as np
from ._common import (
    FlashKernel,
    select_pattern,
    BinningLessOrEqual,
    BinningExact,
    Config,
    check_value,
)
from .ops import OpAttnFwd
from .op_attn_fwd import _IF_CAUSAL
from v3python.base import typed_choice as TC
from v3python.gpu_targets import AOTRITON_ARCH_PRODUCTION_LINE

def _parse_preload_options():
    val = int(os.getenv('AOTRITON_PRE_LOAD_OPTIONS', default='0'))
    if val == 0:
        return [False]
    elif val == 1:
        return [True]
    else:
        return [False, True]
PRE_LOAD_OPTIONS = _parse_preload_options()

class attn_fwd(FlashKernel):
    SHARED_IFACE = OpAttnFwd
    NAME = 'attn_fwd'
    # Note: There is no other FWD metro kernel right now so the arguments are shared
    ARGUMENTS = OpAttnFwd.ARGUMENTS

    PERF_CHOICES = {
        frozenset(['PERSISTENT_TYPE']) : _IF_CAUSAL(TC.constexpr.int8_t(2)),
        frozenset(['GRID_CU_MULTIP']) : np.array([2], dtype=np.int8),  # NOTE: use np.array with dtype to reduce size of the generate tuning infomation struct
        frozenset(['BLOCK_M']) : np.array([16], dtype=np.int16),
        frozenset(['BLOCK_N']) : np.array([16], dtype=np.int16),
        frozenset(['PRE_LOAD_V']) : [False], # [False, True],
    }
    EXPECTED_IDENTICAL_TENSOR_STRIDES = [
        # Not needed stride_o* exist
    ]

    # AUTOTUNE_KEYS can have Functional choices, which will be discarded later
    AUTOTUNE_KEYS = {
        'Max_seqlen_q' : BinningLessOrEqual,
        'Max_seqlen_k' : BinningLessOrEqual,
    }

    # List of functionals that are not fully tuned in the tuning database
    PARTIALLY_TUNED_FUNCTIONALS = {
        'PADDED_HEAD': False,
    }

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
    def gen_autotune_configs(f : 'Functional'):
        arch = f.arch
        dtype = check_value(f, ['Q'])
        HEAD_DIM = check_value(f, ['BLOCK_DMODEL'])
        CAUSAL_TYPE = check_value(f, ['CAUSAL_TYPE'])
        ret = []
        CDNA = AOTRITON_ARCH_PRODUCTION_LINE[arch] == 'CDNA'
        RDNA = AOTRITON_ARCH_PRODUCTION_LINE[arch] == 'RDNA'
        if CDNA:
            BLOCK_SIZES = [(32, 16), (128, 64), (64, 64), (64, 32), (128, 128)]
        elif RDNA:
            BLOCK_SIZES = [(64, 32), (32, 32), (32, 16)]
            if '*fp32' not in dtype:
                BLOCK_SIZES += [(16, 16)]
            else:
                # M //= 2 will effectively yield (16,32), (16,16)
                pass
        WAVES_PER_EU = [1, 2, 3, 4]
        NUM_WARPS = [2, 4]
        PRE_LOAD_V = PRE_LOAD_OPTIONS
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
            if RDNA and HEAD_DIM == 256 and stages == 2:
                continue  # Timeout
            if HEAD_DIM >= 512 and M == 128 and N == 128 and warps == 2:
                continue  # Timeout
            if dtype == '*fp32:16':
                M //= 2
            if M < N:  # Faulty or duplicate
                continue
            if CDNA and M > 32 and N > 16 and stages == 2:
                continue  # No optimal kernel according to 0.8b tuning db
            if CDNA and M > 64 and N > 64 and warps == 1:
                continue  # No optimal kernel according to 0.8b tuning db
            if RDNA and M > 32 and N > 32 and stages == 2:
                continue  # No optimal kernel according to 0.8b tuning db
            if RDNA and M > 32 and N > 32 and warps == 1:
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
        if CDNA:
            pass
            # Covered in general logic above
            # yield from [
            #     Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'waves_per_eu': 2, 'pre_load_v': False}, num_stages=1, num_warps=4),
            #     Config({'BLOCK_M': 128, 'BLOCK_N':  64, 'waves_per_eu': 2, 'pre_load_v': False}, num_stages=1, num_warps=4),
            #     Config({'BLOCK_M': 128, 'BLOCK_N':  64, 'waves_per_eu': 3, 'pre_load_v': False}, num_stages=1, num_warps=4),
            #     Config({'BLOCK_M': 128, 'BLOCK_N':  64, 'waves_per_eu': 1, 'pre_load_v': False}, num_stages=1, num_warps=4),
            #     Config({'BLOCK_M': 128, 'BLOCK_N':  32, 'waves_per_eu': 2, 'pre_load_v': False}, num_stages=1, num_warps=4),
            # ]
        elif RDNA:
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
