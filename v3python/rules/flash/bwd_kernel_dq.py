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
from .bwd_kernel_dk_dv import bwd_kernel_dk_dv
from .op_attn_bwd import OpAttnBwd
from v3python.gpu_targets import AOTRITON_ARCH_PRODUCTION_LINE
match_op = lambda aname : get_possible_choices(OpAttnBwd, aname)
match_kv = lambda aname : get_possible_choices(bwd_kernel_dk_dv, aname)

class bwd_kernel_dq(FlashBwdKernel):
    SHARED_IFACE = OpAttnBwd
    ARGUMENTS = [
        'Q', 'K', 'V', 'B', 'sm_scale', 'DO',
        'DQ', 'DB',
        'L', 'D',
        'stride_qz', 'stride_qh', 'stride_qm', 'stride_qk',
        'stride_kz', 'stride_kh', 'stride_kn', 'stride_kk',
        'stride_vz', 'stride_vh', 'stride_vk', 'stride_vn',
        'stride_bz', 'stride_bh', 'stride_bk', 'stride_bn',
        'stride_doz', 'stride_doh', 'stride_dom', 'stride_dok',
        'stride_dqz', 'stride_dqh', 'stride_dqm', 'stride_dqk',
        'stride_dbz', 'stride_dbh', 'stride_dbm', 'stride_dbn',
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
    PERF_CHOICES = {
        frozenset(['BLOCK_M']) : match_kv('BLOCK_M'),
        frozenset(['BLOCK_N']) : match_kv('BLOCK_N'),
    }
    DEFAULT_NUM_WARPS=4
    DEFAULT_NUM_STAGES=1
    # TODO: waves_per_eu=1
    NAME = 'bwd_kernel_dq'

    AUTOTUNE_KEYS = {
        'max_seqlen_q' : BinningLessOrEqual,
        'max_seqlen_k' : BinningLessOrEqual,
    }
    PARTIALLY_TUNED_FUNCTIONALS = {
        'PADDED_HEAD': False,
    }
    DOWNGRADER = []

    @staticmethod
    def gen_autotune_configs(f : 'Functional'):
        arch = f.arch
        dtype = check_value(f, ['Q'])
        ret = []
        CDNA = AOTRITON_ARCH_PRODUCTION_LINE[arch] == 'CDNA'
        RDNA = AOTRITON_ARCH_PRODUCTION_LINE[arch] == 'RDNA'
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
            kw = {'BLOCK_M': M, 'BLOCK_N': N, 'waves_per_eu': waves}
            if RDNA and M == 64  and N == 64 and stages == 2:
                continue  # No optimal kernel according to 0.8b tuning db
            if RDNA and M * N >= 32 * 32 and warps < 4:
                continue  # Timeout
            if RDNA and M * N >= 32 * 16 and warps < 2:
                continue  # Timeout
            yield Config(kw, num_stages=stages, num_warps=warps)
