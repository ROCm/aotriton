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
from v3python.gpu_targets import AOTRITON_ARCH_WARPSIZE

def _parse_preload_options():
    val = int(os.getenv('AOTRITON_PRE_LOAD_OPTIONS', default='2'))
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
        frozenset(['NUM_XCDS']) : np.array([8], dtype=np.int8),
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

    def gen_autotune_configs(self, f : 'Functional'):
        arch = f.arch
        dtype = check_value(f, ['Q'])
        HEAD_DIM = check_value(f, ['BLOCK_DMODEL'])
        CAUSAL_TYPE = check_value(f, ['CAUSAL_TYPE'])
        ret = []
        WAVE64 = AOTRITON_ARCH_WARPSIZE[arch] == 64
        WAVE32 = AOTRITON_ARCH_WARPSIZE[arch] == 32
        if WAVE64:
            BLOCK_SIZES = [(32, 16), (128, 64), (64, 64), (64, 32), (128, 128)]
        elif WAVE32:
            BLOCK_SIZES = [(64, 32), (32, 32), (32, 16)]
            if '*fp32' not in dtype:
                BLOCK_SIZES += [(16, 16)]
            else:
                # M //= 2 will effectively yield (16,32), (16,16)
                pass
        WAVES_PER_EU = [1, 2, 3, 4]
        NUM_WARPS = [2, 4] if WAVE64 else [4, 8]
        PRE_LOAD_V = PRE_LOAD_OPTIONS
        NUM_STAGES = [1]
        if arch == 'gfx950':
            for waves, pre in itertools.product(WAVES_PER_EU, PRE_LOAD_V):
                persistent_type = 2 if CAUSAL_TYPE != 0 else 0
                kw = { 'PERSISTENT_TYPE' : persistent_type,
                       'GRID_CU_MULTIP': 2,
                       'BLOCK_M': 256,
                       'BLOCK_N': 64,
                       'waves_per_eu': waves,
                       'PRE_LOAD_V': pre,
                     }
                kw = self.update_programmatic_perfs(kw, f)
                yield Config(kw, num_stages=4, num_warps=8)
        for (M, N), waves, warps, stages, pre in itertools.product(BLOCK_SIZES,
                                                                   WAVES_PER_EU,
                                                                   NUM_WARPS,
                                                                   NUM_STAGES,
                                                                   PRE_LOAD_V):
            if HEAD_DIM >= 512 and M == 128 and N == 128 and warps == 2:
                continue  # Timeout
            if dtype == '*fp32:16':
                M //= 2
            if M < N:  # Faulty or duplicate
                continue
            persistent_type = 2 if CAUSAL_TYPE != 0 else 0
            kw = { 'PERSISTENT_TYPE' : persistent_type,
                   'GRID_CU_MULTIP': 2,
                   'BLOCK_M': M,
                   'BLOCK_N': N,
                   'waves_per_eu': waves,
                   'PRE_LOAD_V': pre,
                 }
            kw = self.update_programmatic_perfs(kw, f)
            # TODO: Add Dyamic PERSISTENT_TYPE IFF causal is enabled to tuning database
            yield Config(kw, num_stages=stages, num_warps=warps)
