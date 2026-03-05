# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import itertools
from ._common import (
    get_possible_choices,
    select_pattern,
    BinningLessOrEqual,
    BinningExact,
    Config,
    check_value,
    FlashAffine,
    ConditionalConstexpr as CC,
)
from .attn_fwd import attn_fwd
from .op_attn_bwd import OpAttnBwd
from .aiter_fwd import aiter_fmha_v3_fwd
from v3python.gpu_targets import AOTRITON_ARCH_PRODUCTION_LINE
from v3python.utils import log

class aiter_fmha_v3_bwd(aiter_fmha_v3_fwd):
    CO_DIR = 'fmha_v3_bwd'

    SHARED_IFACE = OpAttnBwd
    NAME = 'aiter_fmha_v3_bwd'
    ARGUMENTS = OpAttnBwd.ARGUMENTS
    CHOICE_FILTERS = {
        'Q' : lambda dtype : 'fp16' in dtype or 'bf16' in dtype,
        'BLOCK_DMODEL' : lambda x : x >= 64 and x <= 192,       # Note: asm kernel only have [64, 128, 192] hdim variants but others in between may be padded.
        'BIAS_TYPE' : lambda b : b == 0,
        'ENABLE_DROPOUT' : lambda dropout : dropout == False,   # TODO: support dropout = True with validated PRNG
    }

    # gfx950+16-bit dq_acc requires another dq_shuffle_kernel, but fp32 dq_acc doesn't
    SUPPORTED_ARCH = ['gfx942', 'gfx950']
    DIRECT_KERNEL_ARGS = []
