# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# TODO: Replace with loop
from .flash import (
    kernels as flash_kernels,
    operators as flash_operators,
    affine_kernels as flash_affine_kernels,
)

kernels = flash_kernels
operators = flash_operators
affine_kernels = flash_affine_kernels
