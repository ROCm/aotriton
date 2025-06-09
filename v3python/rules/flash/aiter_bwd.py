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
)
from .attn_fwd import attn_fwd
from .op_attn_bwd import OpAttnBwd
from v3python.gpu_targets import AOTRITON_ARCH_PRODUCTION_LINE
from v3python.affine import CSVTranslator
match_fwd = lambda aname : get_possible_choices(attn_fwd, aname)

def translate_csv_datatype(self, f, col):
    def etrans(e):
        if e == 'FmhaBwdBf16':
            return '*bf16:16'
        if e == 'FmhaBwdFp16':
            return '*fp16:16'
    return [etrans(e) for e in col]

def translate_csv_tskv(self, f, col):
    ts_kv = 192 if f.arch == "gfx942" else 256
    def etrans(e):
        if e == 'DEFERRED':
            return ts_kv
        return int(e)
    return [etrans(e) for e in col]

class bwd_dq_dk_dv_v3(FlashAffine):
    SHARED_IFACE = OpAttnBwd
    NAME = 'bwd_dq_dk_dv_v3'
    CO_CSV = 'aiter_bwd.csv'
    SUPPORTED_ARCH = ['gfx942', 'gfx950']
    RESIDUAL_CHOICES = {
        frozenset(['ikUniformStrides']) : [False, True],    # ik: "inferred" constant
        # Note: In practice, kIsSEQPad and kIsHDPad are always false when ikUniformStrides is false
        # We kept this is to make selection from CSV easiler
        frozenset(['kIsSEQPad']) : [False, True],
        frozenset(['kIsHDPad']) : [False, True],
        frozenset(['kIsAtomic32']) : [True],                # Always use FP32 for better dq accuracy.
        frozenset(['BF16Cvt']) : [0],                       # Always use RTNE when down casting from dq accumulator
        frozenset(['kIsGroupMode']) : [False, True],
    }

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

    CSV_TRANSLATORS = [
        CSVTranslator(column='HDim', iface_param='BLOCK_DMODEL'),
        CSVTranslator(column='DataType', iface_param='Q', value_translator=translate_csv_datatype),
        CSVTranslator(column='MaskType', iface_param='CAUSAL_TYPE'),  # Our value assignment matches ASM kernel
    ]

