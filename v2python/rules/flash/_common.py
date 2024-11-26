# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from ...kernel_desc import KernelDescription, get_possible_types, select_pattern
from ...autotune_config import Config
from ...autotune_binning import BinningLessOrEqual, BinningExact

class FlashKernel(KernelDescription):
    KERNEL_FAMILY = 'flash'

    def sancheck_lut_tensor(self,
                            gpu,
                            lut_tensor,
                            fsels : 'list[ArgumentSelection]'):
        # Only kernels that provide gen_autotune_configs may have entries in
        # tuning database
        if not hasattr(self, 'gen_autotune_configs'):
            return True
        MI = 'MI' in gpu
        Navi = 'Navi' in gpu
        def check_value(repr_name):
            for fsel in fsels:
                if fsel.repr_name == repr_name:
                    return fsel.argument_value
        is_causal = check_value('CAUSAL')
        bias_type = check_value('BIAS_TYPE')
        if lut_tensor.size == 1:
            to_check = lut_tensor
        elif is_causal and bias_type:
            to_check = lut_tensor.diagonal()
        else:
            to_check = lut_tensor
        if MI or Navi:
            return (to_check >= 0).all() and lut_tensor.shape == (12, 12)
        else:
            assert False, f"Unknown {gpu}"

    def get_missing_lut_entries(self, gpu, lut_tensor, fsels) -> list[dict]:
        SEQLEN_Q = [4,8,16,32,64,128,256,512,1024,2048,4096,8192]
        SEQLEN_K = [4,8,16,32,64,128,256,512,1024,2048,4096,8192]
        from copy import deepcopy
        import json
        import numpy as np
        base = {}
        def check_value(repr_name):
            for fsel in fsels:
                if fsel.repr_name == repr_name:
                    return fsel.argument_value
        base['causal'] = check_value('CAUSAL')
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
        M_idxs, N_idxs = np.where(lut_tensor < 0)
        for M_id, N_id in zip(M_idxs, N_idxs):
            d = deepcopy(base)
            d['seqlen_q'] = SEQLEN_Q[M_id]
            d['seqlen_k'] = SEQLEN_K[N_id]
            ret.append(json.dumps(d))
        return ret
