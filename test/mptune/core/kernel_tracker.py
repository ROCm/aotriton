#!/usr/bin/env python
# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

# In case extargs.total_number_of_kernels never returns a positive number
# Thus the number does not need to be too large since total_number_of_kernels
# will eventually get updated by the output of extargs
CPP_AUTOTUNE_MAX_KERNELS = 20

@dataclass
class KernelIndexProress:
    kernel_index : int = 0
    last_success_kernel : int = None
    total_kernel: int = CPP_AUTOTUNE_MAX_KERNELS
