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
        if MI:
            return (to_check >= 0).all() and lut_tensor.shape == (12, 12)
        elif Navi:
            return (to_check >= 0).all() and lut_tensor.shape == (10, 10)
        else:
            assert False, f"Unknown {gpu}"
