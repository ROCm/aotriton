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

def translate_csv_tskv(f, col):
    ts_kv = 192 if f.arch == "gfx942" else 256
    def etrans(e):
        if e == 'DEFERRED':
            return ts_kv
        return int(e)
    return [etrans(e) for e in col]

class fmha_bwd_v3_args(DirectKernelArguments):
    NAME = 'fmha_bwd_v3_args'
    INCLUDE = 'aotriton/_internal/flash/aiter.h'
    NAMESPACE = 'AOTRITON_NS::v3::flash::aiter'

class fmha_bwd_v3_gen_args(fmha_bwd_v3_args):
    NAME = 'fmha_bwd_v3_gen_args'

class fmha_bwd_v3_genl_args(fmha_bwd_v3_args):
    NAME = 'fmha_bwd_v3_genl_args'

class fmha_bwd_v3_group_args(fmha_bwd_v3_args):
    NAME = 'fmha_bwd_v3_group_args'

class bwd_dq_dk_dv_v3(FlashAffine):
    CO_DIR = 'fmha_v3_bwd'

    SHARED_IFACE = OpAttnBwd
    NAME = 'bwd_dq_dk_dv_v3'
    ARGUMENTS = OpAttnBwd.ARGUMENTS
    CHOICE_FILTERS = {
        'Q' : lambda dtype : 'fp16' in dtype or 'bf16' in dtype,
        'BLOCK_DMODEL' : lambda x : x in [64, 128, 192],        # Note: asm kernel only have 3 hdim variants
        'BIAS_TYPE' : lambda b : b == 0,
        'ENABLE_DROPOUT' : lambda dropout : dropout == False,   # TODO: support dropout = True with validated PRNG
    }

    CO_CSV = 'aiter_bwd.csv'
    SUPPORTED_ARCH = ['gfx942', 'gfx950']
    RESIDUAL_CHOICES = {
        # In practice, kIsSEQPad and kIsHDPad are always false when ifUniformStrides is false
        # Hence kIsSEQPad and kIsHDPad are remove to make the table smaller
        tuple(['kIsUniformStride']) : [False, True],    # Inferred/Implicit
        tuple(['kIsSEQPad']) : [False, True],
        tuple(['kIsHDPad']) : [False, True],
        tuple(['kIsAtomic32']) : [True],                # Always use FP32 for better dq accuracy.
        tuple(['BF16Cvt']) : [0],                       # Always use RTNE when down casting from dq accumulator
        tuple(['kIsGroupMode']) : [False, True],
    }
    DIRECT_KERNEL_ARGS = [
        fmha_bwd_v3_args(),
        fmha_bwd_v3_gen_args(),
        fmha_bwd_v3_genl_args(),
        fmha_bwd_v3_group_args(),
    ]

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
        CSVTranslator(column='HDim', iface_param='BLOCK_DMODEL', value_translator=translate_csv_hdim),
        CSVTranslator(column='DataType', iface_param='Q', value_translator=translate_csv_datatype),
        CSVTranslator(column='MaskType', iface_param='CAUSAL_TYPE'),  # Our value assignment matches ASM kernel
        CSVTranslator(column='kIsAtomic32'),
        # CSVTranslator(column='kIsRegular'),
        CSVTranslator(column='BF16Cvt'),
        CSVTranslator(column='kIsSEQPad'),
        CSVTranslator(column='kIsHDPad', iface_param='PADDED_HEAD'),
        CSVTranslator(column='kIsGroupMode'),
    ]

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

    DF_DICT_PATCH = [
        # {'kIsSEQPad':  True, 'kIsHDPad': False}  # requires kIsUniformStride
        {'kIsSEQPad': False, 'kIsHDPad':  True},
        {'kIsSEQPad':  True, 'kIsHDPad':  True},
    ]
    '''
    AITER ASM kernel has extra limitations
    kIsSEQPad=False && kIsHDPad=False kernel requires kIsUniformStride
    Therefore, if kIsUniformStride=False, need to use
        kIsSEQPad=False && kIsHDPad=True, or
        kIsSEQPad=True && kIsHDPad=False, or
        kIsSEQPad=True && kIsHDPad=True
    kernel
    '''
    def translate_empty_dataframe(self, f : 'Functional'):
        complete_dict = f.build_complete_tc_dict()
        # print(f'{complete_dict=}')
        dic = {}
        for tr in self.CSV_TRANSLATORS:
            log(lambda : f'{tr=} {tr.get_iface_param()=}')
            dic[tr.column] = tr.translate_tc(complete_dict[tr.get_iface_param()])
        kIsUniformStride = complete_dict['kIsUniformStride'].triton_compile_signature
        kIsSEQPad = complete_dict['kIsSEQPad'].triton_compile_signature
        kIsHDPad = complete_dict['kIsHDPad'].triton_compile_signature
        kIsGroupMode = complete_dict['kIsGroupMode'].triton_compile_signature
        # print(f'{kIsUniformStride=}')
        if kIsUniformStride:
            '''
            kIsUniformStride=True: can use any kernel
            '''
            canSEQPad = [True] if kIsSEQPad else [False, True]
            canHDPad = [True] if kIsHDPad else [False, True]
            def locate_csv_rows():
                for pickIsSEQPad in canSEQPad:
                    for pickIsHDPad in canHDPad:
                        dic.update({'kIsSEQPad': pickIsSEQPad, 'kIsHDPad':  pickIsHDPad})
                        ret = self.select_df_by_dict(self._df, dic)
                        if not ret.empty:
                            return ret, pickIsSEQPad, pickIsHDPad
                return ret, pickIsSEQPad, pickIsHDPad
            ret, pickIsSEQPad, pickIsHDPad = locate_csv_rows()
            pp_arg_klass = fmha_bwd_v3_genl_args
            if not pickIsSEQPad and pickIsHDPad:
                pp_arg_klass = fmha_bwd_v3_gen_args
            if not pickIsSEQPad and not pickIsHDPad:
                pp_arg_klass = fmha_bwd_v3_args
            if kIsGroupMode:
                pp_arg_klass = fmha_bwd_v3_group_args
            return ret, pp_arg_klass()
        for patch in self.DF_DICT_PATCH:
            dic.update(patch)
            ret = self.select_df_by_dict(self._df, dic)
            if not ret.empty:
                break
        if kIsGroupMode:
            pp_arg_klass = fmha_bwd_v3_group_args
        else:
            pp_arg_klass = fmha_bwd_v3_genl_args
        return ret, pp_arg_klass()
