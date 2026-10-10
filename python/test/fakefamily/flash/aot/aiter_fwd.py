# Copyright © 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fake AITER-ASM affine backend for op_attn_fwd: minimal @ati.affine.*
fixture exercising SHARED_IFACE resolution."""

import aotriton.template_instantiation as ati


def _aiter_fwd_disabled(functional):
    """ASM-kernel exclusions: no fp32, no hdim > 192."""
    if '*fp32' in functional.choices.Q:
        return True
    if functional.choices.BLOCK_DMODEL > 192:
        return True
    return False


@ati.start
@ati.disable(when=_aiter_fwd_disabled)
@ati.affine.shared_operator('op_attn_fwd')
@ati.affine.arch(['gfx942', 'gfx950'])
@ati.affine.limitations(Q=lambda dtype: 'fp16' in dtype or 'bf16' in dtype,
                        BLOCK_DMODEL=lambda x: x in [128, 192],
                        BIAS_TYPE=lambda b: b == 0,
                        ENABLE_DROPOUT=lambda dropout: dropout == False)
@ati.affine.structures(cookie='aiter::mha_fwd_args')
@ati.affine.directories(co_dir='fmha_v3_fwd',
                        headers=['aotriton/_internal/flash/aiter.h'])
@ati.affine.aiter_asm(name='aiter_fmha_v3_fwd')
def aiter_fmha_v3_fwd():
    pass
