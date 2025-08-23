# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
from ...gpu_targets import gpu2arch, AOTRITON_ARCH_WARPSIZE
from ...op import Operator
from ...kernel.kdesc import (
    KernelDescription,
    get_possible_choices,
    select_pattern,
    ConditionalConstexpr,
    ConditionalDeferredConstexpr,
    ConditionalDeferredElseTensor,
    AOTRITON_ENABLE_FP32,
)
from v3python.autotune import (
    Config,
    BinningLessOrEqual,
    BinningExact,
)
from v3python.utils import log
from v3python.affine import AffineKernelDescription

class OpAttn(Operator):
    FAMILY = 'flash'
    MAIN_DATATYPES = ['*fp16:16', '*bf16:16', '*fp32:16'] if AOTRITON_ENABLE_FP32 else ['*fp16:16', '*bf16:16']
    CALL_OPTIONS_NAME = 'attn_options'

class FlashAffine(AffineKernelDescription):
    FAMILY = 'flash'
    MODULE_FILE = __file__
    AFFINE_KERNEL_ROOT = Path('third_party/aiter/hsa')
    CO_DIR = None           # Required by subclass

    def co_dir(self, functional):
        return self.AFFINE_KERNEL_ROOT / functional.arch / self.CO_DIR

def check_value(functional, repr_name):
    if not isinstance(repr_name, list):
        repr_name = [repr_name]
    tc = functional.compact_choices
    for aname in repr_name:
        if aname in tc:
            return tc[aname].triton_compile_signature
    assert False, f'Cannot find {repr_name=} in {functional=}'

class FlashKernel(KernelDescription):
    FAMILY = 'flash'
    LUT_FULL_SEQLEN_Q = [16,32,64,128,256,512,1024,2048,4096,8192]
    LUT_FULL_SEQLEN_K = [16,32,64,128,256,512,1024,2048,4096,8192]
    LUT_FULL_SEQLEN_NAVI = [16,32,64,128,256,512,1024]

    def is_functional_disabled(self, functional):
        if not hasattr(self, 'gen_autotune_configs'):  # only check acutal FA kernels
            return False
        is_causal = check_value(functional, ['CAUSAL', 'CAUSAL_TYPE'])
        bias_type = check_value(functional, 'BIAS_TYPE')
        # print(f'Functional {functional.godel_number=} {is_causal=} {bias_type=}')
        if is_causal and bias_type != 0:
            return True
        if functional.arch.startswith('gfx11'):
            hdim = check_value(functional, 'BLOCK_DMODEL')
            if hdim > 256:
                return True
        return False

    def sancheck_lut_tensor(self,
                            functional : 'Functional',
                            lut_tensor):
        # Only kernels that provide gen_autotune_configs may have entries in
        # tuning database
        if not hasattr(self, 'gen_autotune_configs'):
            return True
        arch = functional.arch
        if self.is_functional_disabled(functional):
            return True  # ignore disabled functionals
        MI = (AOTRITON_ARCH_WARPSIZE[arch] == 64)
        Navi = (AOTRITON_ARCH_WARPSIZE[arch] == 32)
        LUT_TENSOR_SIZE = (len(self.LUT_FULL_SEQLEN_Q), len(self.LUT_FULL_SEQLEN_K))
        LUT_TENSOR_SIZE_NAVI = (len(self.LUT_FULL_SEQLEN_NAVI), len(self.LUT_FULL_SEQLEN_NAVI))
        if lut_tensor.size == 1:
            to_check = lut_tensor
        else:
            to_check = lut_tensor
        log(lambda : f'{lut_tensor.shape=} ==? {LUT_TENSOR_SIZE=}')
        if MI:
            return (to_check >= 0).all() and lut_tensor.shape[1:] == LUT_TENSOR_SIZE
        elif Navi:
            return (to_check >= 0).all() and (lut_tensor.shape[1:] == LUT_TENSOR_SIZE or lut_tensor.shape[1:] == LUT_TENSOR_SIZE_NAVI)
        else:
            assert False, f"Unknown {arch}"

    '''
    TODO: new tuning framework should reuse KernelDescription
    '''
    def get_missing_lut_entries(self, lut_tensor, functional) -> list[dict]:
        from copy import deepcopy
        import json
        import numpy as np
        arch = functional.arch
        base = {'arch' : arch}
        MI = (AOTRITON_ARCH_WARPSIZE[arch] == 64)
        Navi = (AOTRITON_ARCH_WARPSIZE[arch] == 32)
        # if Navi:
        #     lut_full_seqlen_q = self.LUT_FULL_SEQLEN_NAVI
        #     lut_full_seqlen_k = self.LUT_FULL_SEQLEN_NAVI
        # else:
        #     lut_full_seqlen_q = self.LUT_FULL_SEQLEN_Q
        #     lut_full_seqlen_k = self.LUT_FULL_SEQLEN_K
        lut_full_seqlen_q = self.LUT_FULL_SEQLEN_Q
        lut_full_seqlen_k = self.LUT_FULL_SEQLEN_K
        LUT_TENSOR_SIZE = (len(self.LUT_FULL_SEQLEN_Q), len(self.LUT_FULL_SEQLEN_K))
        base['causal_type'] = check_value(functional, 'CAUSAL_TYPE')
        base['d_head'] = check_value(functional, 'BLOCK_DMODEL')
        base['dropout_p'] = 0.5 if check_value(functional, 'ENABLE_DROPOUT') else 0.0
        def dtype():
            value = check_value(functional, 'Q')
            if value.startswith('*fp16'):
                return 'float16'
            if value.startswith('*bf16'):
                return 'bfloat16'
            if value.startswith('*fp32'):
                return 'float32'
        base['dtype'] = dtype()
        base['bias_type'] = check_value(functional, 'BIAS_TYPE')
        ret = []
        if lut_tensor.size == 1:
            for seqlen_q in lut_full_seqlen_q:
                for seqlen_k in lut_full_seqlen_k:
                    d = deepcopy(base)
                    d['seqlen_q'] = seqlen_q
                    d['seqlen_k'] = seqlen_k
                    ret.append(json.dumps(d))
        else:
            # TODO: support non-mod0
            if lut_tensor.shape[1:] == LUT_TENSOR_SIZE:
                _, M_idxs, N_idxs = np.where(lut_tensor < 0)
            else:
                fake_lut = np.full(LUT_TENSOR_SIZE, -1, dtype=np.int32)
                M_idxs, N_idxs = np.where(fake_lut < 0)
            # print(f'{M_idxs=} {N_idxs=} {lut_full_seqlen_q=} {lut_full_seqlen_k=}')
            for M_id, N_id in zip(M_idxs, N_idxs):
                d = deepcopy(base)
                d['seqlen_q'] = lut_full_seqlen_q[M_id]
                d['seqlen_k'] = lut_full_seqlen_k[N_id]
                ret.append(json.dumps(d))
        return ret

class FlashBwdKernel(FlashKernel):
    def is_functional_disabled(self, functional):
        return super().is_functional_disabled(functional)
