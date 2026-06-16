# Copyright © 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI description of the flash forward affine (AITER ASM) kernel.

A SLIM affine backend: a thin C++ shim translating OpAttnFwdParams into the AITER
`aiter::mha_fwd_args` cookie and packaging pre-built fmha_v3_fwd .co files. Declared
declaratively via the @ati.affine.* surface; the operator (op_attn_fwd) it serves is
referenced by name and resolved to SHARED_IFACE by infer_shared_iface.
"""

import aotriton.template_instantiation as ati
from ._common import check_value


def _aiter_fwd_disabled(functional):
    """ASM-kernel exclusions: no fp32, no hdim > 192."""
    dtype = check_value(functional, ['Q'])
    if '*fp32' in dtype:
        return True
    hdim = check_value(functional, ['BLOCK_DMODEL'])
    if hdim > 192:
        return True
    return False


@ati.kernel
@ati.disable(when=_aiter_fwd_disabled)
@ati.affine.aiter_asm(name='aiter_fmha_v3_fwd')
@ati.affine.shared_operator('op_attn_fwd')
@ati.affine.arch(['gfx942', 'gfx950'])
@ati.affine.limitations(Q=lambda dtype: 'fp16' in dtype or 'bf16' in dtype,
                        BLOCK_DMODEL=lambda x: x in [128, 192],
                        BIAS_TYPE=lambda b: b == 0,
                        ENABLE_DROPOUT=lambda dropout: dropout == False)
@ati.affine.structures(cookie='aiter::mha_fwd_args')
@ati.affine.directories(co_dir='fmha_v3_fwd',
                        headers=['aotriton/_internal/flash/aiter.h'])
def aiter_fmha_v3_fwd():
    pass
