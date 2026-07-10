# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fake AITER-ASM affine backend for op_attn_bwd: minimal @ati.affine.*
fixture supplying DQ_ACC, exercising affine/backend linking."""

import aotriton.template_instantiation as ati


def _aiter_bwd_disabled(functional):
    """ASM-kernel exclusions: no fp32, no hdim > 192."""
    if '*fp32' in functional.choices.Q:
        return True
    if functional.choices.BLOCK_DMODEL > 192:
        return True
    return False


@ati.start
@ati.disable(when=_aiter_bwd_disabled)
@ati.affine.shared_operator('op_attn_bwd')
@ati.affine.arch(['gfx942', 'gfx950'])
# asm bwd has [64, 128, 192] hdim variants (others in between may be padded).
@ati.affine.limitations(Q=lambda dtype: 'fp16' in dtype or 'bf16' in dtype,
                        BLOCK_DMODEL=lambda x: 64 <= x <= 192,
                        BIAS_TYPE=lambda b: b == 0,
                        ENABLE_DROPOUT=lambda dropout: dropout == False)
@ati.affine.structures(cookie='aiter::mha_bwd_args')
@ati.affine.directories(co_dir='fmha_v3_bwd',
                        headers=['aotriton/_internal/flash/aiter.h'])
# The extra operand this backend contributes to OpAttnBwdParams (the union places
# it between DB and L via the after/before anchors).
@ati.affine.supplies(
    ati.tensor('DQ_ACC', 'LazyTensor:*fp32:16', strides='stride_acc?', contiguous=-1),
    after='DB', before='L')
@ati.affine.aiter_asm(name='aiter_fmha_v3_bwd')
def aiter_fmha_v3_bwd():
    pass
