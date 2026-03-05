# Copyright © 2025-2026 Advanced Micro Devices, Inc.
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
from .op_attn_fwd import OpAttnFwd
from v3python.gpu_targets import AOTRITON_ARCH_PRODUCTION_LINE
from v3python.utils import log

class aiter_fmha_v3_fwd(FlashAffine):
    CO_DIR = 'fmha_v3_fwd'

    SHARED_IFACE = OpAttnFwd
    NAME = 'aiter_fmha_v3_fwd'
    ARGUMENTS = OpAttnFwd.ARGUMENTS
    CHOICE_FILTERS = {
        'Q' : lambda dtype : 'fp16' in dtype or 'bf16' in dtype,
        'BLOCK_DMODEL' : lambda x : x in [128, 192],
        'BIAS_TYPE' : lambda b : b == 0,
        'ENABLE_DROPOUT' : lambda dropout : dropout == False,   # TODO: support dropout = True with validated PRNG
    }

    # gfx950+16-bit dq_acc requires another dq_shuffle_kernel, but fp32 dq_acc doesn't
    SUPPORTED_ARCH = ['gfx942', 'gfx950']
    DIRECT_KERNEL_ARGS = []

    def is_functional_disabled(self, functional):
        dtype = check_value(functional, ['Q'])
        if '*fp32' in dtype:
            return True
        hdim = check_value(functional, ['BLOCK_DMODEL'])
        if hdim > 192:
            return True
        # Unnecessary since CHOICE_FILTERS ensures BIAS_TYPE == 0
        # Kept in case furture ASM kernel supports BIAS_TYPE == 1
        # is_causal = check_value(functional, ['CAUSAL', 'CAUSAL_TYPE'])
        # bias_type = check_value(functional, 'BIAS_TYPE')
        # if is_causal and bias_type != 0:
        #     return True
        df = self.translate_empty_dataframe(functional)
        if df is None:
            return True
        return False
