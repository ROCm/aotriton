# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

AOTRITON_SUPPORTED_GPUS = {
    'MI200'     : 'GPU_ARCH_AMD_GFX90A',
    'MI300X'    : 'GPU_ARCH_AMD_GFX942',
    'Navi31'    : 'GPU_ARCH_AMD_GFX1100',
    'Navi32'    : 'GPU_ARCH_AMD_GFX1101',
}

AOTRITON_GPU_ARCH_TUNING_STRING = {
    'MI200'     : 'gfx90a',
    'MI300X'    : 'gfx942',
    'Navi31'    : 'gfx1100',
    'Navi32'    : 'gfx1101',
}

AOTRITON_GPU_WARPSIZE = {
    'MI200'     : 64,
    'MI300X'    : 64,
    'Navi31'    : 32,
    'Navi32'    : 32,
}
