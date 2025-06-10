# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import itertools
from ._common import (
    FlashKernel,
    FlashBwdKernel,
    get_possible_choices,
    select_pattern,
    BinningLessOrEqual,
    BinningExact,
    Config,
    check_value,
)
from .attn_fwd import attn_fwd
from .op_attn_bwd import OpAttnBwd
from v3python.gpu_targets import AOTRITON_ARCH_PRODUCTION_LINE
match_fwd = lambda aname : get_possible_choices(attn_fwd, aname)

class bwd_kernel_dk_dv(FlashBwdKernel):
    SHARED_IFACE = OpAttnBwd
    ARGUMENTS = [
        'Q', 'K', 'V', 'B', 'sm_scale', 'DO',
        'DK', 'DV',
        'L', 'D',
        'stride_qz', 'stride_qh', 'stride_qm', 'stride_qk',
        'stride_kz', 'stride_kh', 'stride_kn', 'stride_kk',
        'stride_vz', 'stride_vh', 'stride_vk', 'stride_vn',
        'stride_bz', 'stride_bh', 'stride_bk', 'stride_bn',
        'stride_doz', 'stride_doh', 'stride_dom', 'stride_dok',
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
        'Window_left',
        'Window_right',
        'BLOCK_M', # tl.constexpr starts here
        'BLOCK_DMODEL',
        'BLOCK_N',
        'CAUSAL_TYPE',
        'ENABLE_DROPOUT',
        'PADDED_HEAD',
        'BIAS_TYPE',
    ]
    DEFAULT_NUM_WARPS=4
    DEFAULT_NUM_STAGES=1
    NAME = 'bwd_kernel_dk_dv'

    AUTOTUNE_KEYS = {
        'max_seqlen_q' : BinningLessOrEqual,
        'max_seqlen_k' : BinningLessOrEqual,
    }
    PERF_CHOICES = {
        frozenset(['BLOCK_M']) : match_fwd('BLOCK_M'),
        frozenset(['BLOCK_N']) : match_fwd('BLOCK_N'),
    }

    PARTIALLY_TUNED_FUNCTIONALS = {
        'PADDED_HEAD': False,
    }
    DOWNGRADER = []

    @staticmethod
    def gen_autotune_configs(f : 'Functional'):
        arch = f.arch
        dtype = check_value(f, ['Q'])
        HEAD_DIM = check_value(f, ['BLOCK_DMODEL'])
        ret = []
        CDNA = AOTRITON_ARCH_PRODUCTION_LINE[arch] == 'CDNA'
        RDNA = AOTRITON_ARCH_PRODUCTION_LINE[arch] == 'RDNA'
        # TODO: right sizes for fp32?
        BLOCK_SIZES = [16, 32, 64] if dtype != '*fp32:16' else [16, 32]
        WAVES_PER_EU = [1, 2, 3, 4]
        NUM_WARPS = [4, 8] if HEAD_DIM >= 512 and RDNA else [2, 4]
        NUM_STAGES = [1]
        for M, N, waves, warps, stages in itertools.product(BLOCK_SIZES,
                                                            BLOCK_SIZES,
                                                            WAVES_PER_EU,
                                                            NUM_WARPS,
                                                            NUM_STAGES):
            if M < N:
                continue  # deduplicate
            if CDNA and M == 64 and N == 64 and warps == 4:
                continue  # No optimal kernel according to 0.8b tuning db
            if HEAD_DIM >= 512 and M == 64 and N == 64 and warps == 1:
                continue  # Timeout
            if RDNA and M * N >= 32 * 32 and warps < 4:
                continue  # Timeout
            if RDNA and M * N >= 32 * 16 and warps < 2:
                continue  # Timeout
            kw = {'BLOCK_M': M, 'BLOCK_N': N, 'waves_per_eu': waves}
            yield Config(kw, num_stages=stages, num_warps=warps)
