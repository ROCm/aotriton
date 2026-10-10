# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os


# Shared across the flash kernel descriptions (attn_fwd / bwd_kernel_* / bwd_preprocess*).
MAIN_DTYPES = ['*fp16:16', '*bf16:16', '*fp32:16']


def block_dmodel_values():
    """BLOCK_DMODEL values, overridable via AOTRITON_FLASH_BLOCK_DMODEL."""
    env = os.getenv('AOTRITON_FLASH_BLOCK_DMODEL',
                    default='16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 512')
    return [int(d) for d in env.split(',')]


def flash_disabled(f, *, gfx950_bad_hdims=()):
    """True if `f` must be excluded (shared fwd/bwd predicate).

    gfx950_bad_hdims: per-kernel BLOCK_DMODEL values gfx950 miscompiles."""
    causal = f.choices.CAUSAL_TYPE
    hdim = f.choices.BLOCK_DMODEL
    bias_type = f.choices.BIAS_TYPE
    if causal != 0 and bias_type != 0:
        return True
    if f.arch.startswith('gfx11') and hdim > 256:
        return True
    if f.arch == 'gfx950' and hdim in gfx950_bad_hdims:
        return True
    return False
