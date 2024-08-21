# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from ...kernel_desc import KernelDescription, get_possible_types, select_pattern
from ...autotune_config import Config
from ...autotune_binning import BinningLessOrEqual, BinningExact

class FlashKernel(KernelDescription):
    KERNEL_FAMILY = 'flash'

    def sancheck_lut_tensor(self,
                            lut_tensor,
                            fsels : 'list[ArgumentSelection]'):
        def check_value(repr_name):
            for fsel in fsels:
                if fsel.repr_name == repr_name:
                    return fsel.argument_value
        is_causal = check_value('CAUSAL')
        if is_causal:
            to_check = lut_tensor.diagonal()
        else:
            to_check = lut_tensor
        return (to_check >= 0).all()
