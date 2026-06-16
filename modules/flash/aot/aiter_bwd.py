# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI description of the flash backward affine (AITER ASM) kernel.

A SLIM affine backend: a thin C++ shim translating OpAttnBwdParams into the AITER
`aiter::mha_bwd_args` cookie and packaging pre-built fmha_v3_bwd .co files.

It SUPPLIES the DQ_ACC operand (the workspace accumulator only this backend uses):
the operator's params-struct union picks it up so DQ_ACC lands in OpAttnBwdParams
without any hand-injection. DQ_ACC is a LazyTensor (rank 4, strides stride_acc?).
"""

import aotriton.template_instantiation as ati
from ._common import check_value


def _aiter_bwd_disabled(functional):
    """ASM-kernel exclusions: no fp32, no hdim > 192."""
    dtype = check_value(functional, ['Q'])
    if '*fp32' in dtype:
        return True
    hdim = check_value(functional, ['BLOCK_DMODEL'])
    if hdim > 192:
        return True
    return False


@ati.kernel
@ati.disable(when=_aiter_bwd_disabled)
@ati.affine.aiter_asm(name='aiter_fmha_v3_bwd')
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
def aiter_fmha_v3_bwd():
    pass
