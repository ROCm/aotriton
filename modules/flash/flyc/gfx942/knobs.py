# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Knob classes of the gfx942 attention kernels (forward, dQ, dK/dV).

Plain data and flydsl-free: the code generator imports them to record the
resolved knobs. The policy that fills them is in the `tuning_*` modules.

Each kernel has several bodies (narrow, wide, ...), all compiled as the one
`@flyc.kernel` def named below, in the entry file named below. `body` says
which one a build uses; the entry file's primary builder maps it to a source.
"""

from dataclasses import dataclass, replace

from fmha_knobs import (
    GRID_AXIS_TILE_FASTEST,
    FlycQKnobs,
)

__all__ = ["FmhaKnobs"]

# dK/dV `mask_at`: where the band mask lands.
MASK_SCORE = 'score'        # score -> -inf before exp2
MASK_PROB = 'prob'          # P -> 0 after exp2
# dK/dV `dropout_scale_at`: where dropout's 1/(1-p) meets P for dV.
P_DROP_AT_STORE = 'store'   # dropout zeroes P; its 1/(1-p) joins dV at the store
P_DROP_PER_ELEM = 'elem'    # P scaled by 1/(1-p) per element


class _PolicyGridAxisOrder:
    """gfx942's grid order is per functional, so `GRID_AXIS_ORDER` is a policy
    field here: `with_flyc_fields` keeps a resolved one, and falls back to
    `FLYC_GRID_AXIS_ORDER` only when it is unset."""

    def with_flyc_fields(self):
        order = self.GRID_AXIS_ORDER
        knobs = super().with_flyc_fields()
        return knobs if order is None else replace(knobs, GRID_AXIS_ORDER=order)


@dataclass(frozen=True)
class FmhaKnobs(_PolicyGridAxisOrder, FlycQKnobs):
    """The forward's knobs. `None` means "policy decides"."""

    body: str | None = None
    num_waves: int | None = None
    block_n: int | None = None
    waves_per_eu: int | None = None
    # > 0 enables the lazy running max (numerical lessons, #17); 0 is exact.
    rescale_threshold: float | None = None

    FLYC_SOURCE = "gfx942/fwd.py"
    FLYC_KERNEL_NAME = "fmha_fwd_gfx942_kernel"
    # Non-causal; causal narrow and xwide builds take HEAD_FASTEST.
    FLYC_GRID_AXIS_ORDER = GRID_AXIS_TILE_FASTEST
