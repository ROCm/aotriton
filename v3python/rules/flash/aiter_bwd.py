# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import itertools
from ._common import (
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

class bwd_dq_dk_dv_v3(FlashAffine):
    SHARED_IFACE = OpAttnBwd
    NAME = 'bwd_dq_dk_dv_v3'
    CO_CSV = 'aiter_bwd.csv'
    SUPPORTED_ARCH = ['gfx942', 'gfx950']

    def limits(self):
        ts_kv = 192 if get_gfx() == "gfx942" else 256
        seqlen_limit = 64 if get_gfx() == "gfx942" else 256
        dq_shuffle_kernel_define = "" if get_gfx() == "gfx942" else DQ_SHUFFLE_KERNEL_DEFINE
        dq_shuffle_kernel_call = "" if get_gfx() == "gfx942" else DQ_SHUFFLE_KERNEL_CALL
        dqdkdv_kernel = FMHA_BWD_KERNEL_HEADER + FMHA_BWD_API.format(
            F_AITER_ASM_DIR=get_asm_dir(),
            F_tile_size_kv=ts_kv,
            F_seqlen_limit=seqlen_limit,
            F_dq_shuffle_kernel_define=dq_shuffle_kernel_define,
            F_dq_shuffle_kernel_call=dq_shuffle_kernel_call,
        )
