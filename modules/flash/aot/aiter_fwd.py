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
from .op_attn_fwd import OpAttnFwd
from aotriton.utils import log

class aiter_fmha_v3_fwd(FlashAffine):
    CO_DIR = 'fmha_v3_fwd'
    HEADER_EXTRA_INCLUDES = ['aotriton/_internal/flash/aiter.h']
    COOKIE_CLASS = 'aiter::mha_fwd_args'

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
        # Kept in case future ASM kernel supports BIAS_TYPE == 1
        # is_causal = check_value(functional, ['CAUSAL', 'CAUSAL_TYPE'])
        # bias_type = check_value(functional, 'BIAS_TYPE')
        # if is_causal and bias_type != 0:
        #     return True
        df = self.translate_empty_dataframe(functional)
        if df is None:
            return True
        return False

# DRAFT (design sketch for the future affine-kernel ATI port — NOT yet executable;
# @ati.affine.* / @ati.start do not exist yet). Kept as a note for the deferred
# Step that ports aiter_fmha_v3_* off the legacy SlimAffineKernelDescription.
#
#   @ati.start  # or @ati.kernel if start hasn't been created yet
#   @ati.disable(when=is_functional_disabled)
#   @ati.affine.arch(['gfx942', 'gfx950'])
#   @ati.affine.limitations(Q=lambda dtype: 'fp16' in dtype or 'bf16' in dtype,
#                           BLOCK_DMODEL=lambda x: x in [128, 192])
#   @ati.affine.structures(COOKIE='aiter::mha_fwd_args')
#   @ati.affine.directories(CO_DIR='fmha_v3_fwd',
#                           HEADERS=['aotriton/_internal/flash/aiter.h'])
#   @ati.affine.aiter_asm
#   def aiter_fmha_v3_fwd():
#       pass
