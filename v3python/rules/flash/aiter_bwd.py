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
    FlashAffine,
    ConditionalConstexpr as CC,
)
from .attn_fwd import attn_fwd
from .op_attn_bwd import OpAttnBwd
from v3python.gpu_targets import AOTRITON_ARCH_PRODUCTION_LINE
from v3python.affine import CSVTranslator, DirectKernelArguments
from v3python.utils import log
match_fwd = lambda aname : get_possible_choices(attn_fwd, aname)

def translate_csv_hdim(hdim):
    if hdim <= 64:
        return 64
    if hdim <= 128:
        return 128
    if hdim <= 192:
        return 192
    assert False, f'Should not call translate_csv_hdim with hdim > 192, but got {hdim}. Need to remove such functional defensively with CHOICE_FILTERS or is_functional_disabled'

def translate_csv_datatype(value):
    if value == '*bf16:16':
        return 'FmhaBwdBf16'
    if value == '*fp16:16':
        return 'FmhaBwdFp16'
    assert False, f'Should only call translate_csv_datatype with fp16/bf16, but got {value}. Need to remove such functional defensively with CHOICE_FILTERS or is_functional_disabled'
    return None

def translate_regular_to_bothpad(is_regular):
    if is_regular:
        return False
    return True

def translate_csv_tskv(f, value):
    if value > 0:
        return value
    assert f.arch in ["gfx942", "gfx950"]
    # Arch depedent ts_kv
    ts_kv = 192 if f.arch == "gfx942" else 256
    return ts_kv

class fmha_bwd_v3_args(DirectKernelArguments):
    NAME = 'fmha_bwd_v3_args'
    INCLUDE = 'aotriton/_internal/flash/aiter.h'
    NAMESPACE = 'AOTRITON_NS::v3::flash::aiter'

class aiter_fmha_v3_bwd(FlashAffine):
    CO_DIR = 'fmha_v3_bwd'

    SHARED_IFACE = OpAttnBwd
    NAME = 'aiter_fmha_v3'
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
