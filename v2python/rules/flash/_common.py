# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from ...kernel_desc import KernelDescription, get_possible_choices, select_pattern
from ...autotune_config import Config
from ...autotune_binning import BinningLessOrEqual, BinningExact

class FlashKernel(KernelDescription):
    KERNEL_FAMILY = 'flash'
    LUT_FULL_SEQLEN_Q = [4,8,16,32,64,128,256,512,1024,2048,4096,8192]
    LUT_FULL_SEQLEN_K = [4,8,16,32,64,128,256,512,1024,2048,4096,8192]
    LUT_FULL_SEQLEN_NAVI = [4,8,16,32,64,128,256,512,1024]

    def sancheck_lut_tensor(self,
                            gpu,
                            lut_tensor,
                            fsels : 'list[ArgumentSelection]'):
        # Only kernels that provide gen_autotune_configs may have entries in
        # tuning database
        if not hasattr(self, 'gen_autotune_configs'):
            return True
        if gpu == 'Unidentified':  # Will not have a tuning database
            return True
        MI = 'MI' in gpu
        Navi = 'Navi' in gpu
        LUT_TENSOR_SIZE = (len(self.LUT_FULL_SEQLEN_Q), len(self.LUT_FULL_SEQLEN_K))
        LUT_TENSOR_SIZE_NAVI = (len(self.LUT_FULL_SEQLEN_NAVI), len(self.LUT_FULL_SEQLEN_NAVI))
        def check_value(repr_name):
            if not isinstance(repr_name, list):
                repr_name = [repr_name]
            for fsel in fsels:
                if fsel.repr_name in repr_name:
                    return fsel.argument_value
        is_causal = check_value(['CAUSAL', 'CAUSAL_TYPE'])
        bias_type = check_value('BIAS_TYPE')
        if lut_tensor.size == 1:
            to_check = lut_tensor
        elif is_causal and bias_type:
            to_check = lut_tensor.diagonal()
        else:
            to_check = lut_tensor
        if MI:
            return (to_check >= 0).all() and lut_tensor.shape == LUT_TENSOR_SIZE
        elif Navi:
            return (to_check >= 0).all() and (lut_tensor.shape == LUT_TENSOR_SIZE or lut_tensor.shape == LUT_TENSOR_SIZE_NAVI)
        else:
            assert False, f"Unknown {gpu}"

    def get_missing_lut_entries(self, gpu, lut_tensor, fsels) -> list[dict]:
        from copy import deepcopy
        import json
        import numpy as np
        base = {'gpu' : gpu}
        MI = 'MI' in gpu
        Navi = 'Navi' in gpu
        if Navi:
            lut_full_seqlen_q = self.LUT_FULL_SEQLEN_NAVI
            lut_full_seqlen_k = self.LUT_FULL_SEQLEN_NAVI
        else:
            lut_full_seqlen_q = self.LUT_FULL_SEQLEN_Q
            lut_full_seqlen_k = self.LUT_FULL_SEQLEN_K
        def check_value(repr_name):
            if not isinstance(repr_name, list):
                repr_name = [repr_name]
            for fsel in fsels:
                if fsel.repr_name in repr_name:
                    return fsel.argument_value
        base['causal'] = check_value(['CAUSAL', 'CAUSAL_TYPE'])
        base['d_head'] = check_value('BLOCK_DMODEL')
        base['dropout_p'] = 0.5 if check_value('ENABLE_DROPOUT') else 0.0
        def dtype():
            value = check_value('Q')
            if value.startswith('*fp16'):
                return 'float16'
            if value.startswith('*bf16'):
                return 'bfloat16'
            if value.startswith('*fp32'):
                return 'float32'
        base['dtype'] = dtype()
        base['bias_type'] = check_value('BIAS_TYPE')
        ret = []
        if lut_tensor.size == 1:
            for seqlen_q in lut_full_seqlen_q:
                for seqlen_k in lut_full_seqlen_k:
                    d = deepcopy(base)
                    d['seqlen_q'] = seqlen_q
                    d['seqlen_k'] = seqlen_k
                    ret.append(json.dumps(d))
        else:
            M_idxs, N_idxs = np.where(lut_tensor < 0)
            for M_id, N_id in zip(M_idxs, N_idxs):
                d = deepcopy(base)
                d['seqlen_q'] = lut_full_seqlen_q[M_id]
                d['seqlen_k'] = lut_full_seqlen_k[N_id]
                ret.append(json.dumps(d))
        return ret
