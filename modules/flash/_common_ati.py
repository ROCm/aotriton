# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Shared ATI description helpers for the flash family.

`flash_disabled` is the single functional-disable predicate shared by the fwd and
bwd kernels (the user interface to is_functional_disabled). The only per-kernel
variation is the set of head dims the gfx950 compiler mishandles, so that is a
parameter; everything else (causal+matrix-bias unsupported, gfx11 hdim>256) is
common.
"""


def flash_disabled(f, *, gfx950_bad_hdims=()):
    """True if functional `f` must be excluded for compiler/numerical correctness.

    Shared by fwd and bwd. `gfx950_bad_hdims` is the per-kernel set of BLOCK_DMODEL
    values the gfx950 compiler has a known numerical error on (fwd: {16};
    bwd: {48, 80})."""
    causal = f.choices.CAUSAL_TYPE
    hdim = f.choices.BLOCK_DMODEL
    bias_type = f.choices.BIAS_TYPE
    # causal + matrix bias is unsupported
    if causal != 0 and bias_type != 0:
        return True
    # gfx11xx cannot handle hdim > 256
    if f.arch.startswith('gfx11') and hdim > 256:
        return True
    # gfx950 compiler numerical errors on certain head dims
    if f.arch == 'gfx950' and hdim in gfx950_bad_hdims:
        return True
    return False
