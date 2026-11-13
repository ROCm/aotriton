# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Knobs of the gfx942 backward dQ/dB kernels: fmha_bwd_dq_gfx942.py (head dims
32-128) and fmha_bwd_dq_gfx942_wide.py (160-512).

flydsl-free, like fmha_tuning_gfx942.py: the code generator imports it.
The builder takes its defaults from here, so the two cannot drift.
"""

from dataclasses import dataclass

from fmha_tuning_gfx942 import FmhaInputMetadata, resolve_fields


# The compiled head dims; an off-ladder head dim runs padded on the next rung.
LADDER = (32, 64, 96, 128, 160, 192, 224, 256, 384, 512)

# (source file, @flyc.kernel name, builder) per body.
BODIES = {
    'narrow': ('fmha_bwd_dq_gfx942.py', 'fmha_bwd_dq_gfx942_kernel', 'build_fmha_bwd_dq_gfx942'),
    'wide': ('fmha_bwd_dq_gfx942_wide.py', 'fmha_bwd_dq_gfx942_wide_kernel', 'build_fmha_bwd_dq_gfx942_wide'),
}

# csrc/flyc_common.h's kFlycGridAxis*.
GRID_AXIS_HEAD_FASTEST = 0
GRID_AXIS_TILE_FASTEST = 1


def default_block_n(head_dim):
    """KV rows per tile: 48 MFMAs per tile at 64 and 128."""
    return 64 if head_dim <= 64 else 32


def default_grid_axis_order(causal):
    """Head-fastest: tile-fastest measured no better (B2 H32 S8k d128: 518 either
    way; causal 315 vs 444), and csrc/flyc_bwd_dq.cc does not serve it."""
    return GRID_AXIS_HEAD_FASTEST


def default_lpt(causal):
    """Longest q tiles first under causal masks (B2 H32 S8k causal: d128 444 -> 471,
    d64 367 -> 418 TFLOPS)."""
    return causal


def body(head_dim):
    """Which body serves a compiled head dim."""
    return 'narrow' if head_dim <= 128 else 'wide'


def rung(hdim_qk, hdim_vo):
    """The compiled head dim serving a call, or None past the ladder."""
    h = max(hdim_qk, hdim_vo)
    return next((r for r in LADDER if r >= h), None)


def build(head_dim, **kwargs):
    """The body's builder; flydsl is imported only here, never at module scope."""
    import importlib

    src, _, name = BODIES[body(head_dim)]
    return getattr(importlib.import_module(src[:-3]), name)(head_dim, **kwargs)


def knobs(head_dim, causal):
    """The build of one functional, as a JSON-able dict. `block_m` (rows per grid
    block) and `GRID_AXIS_ORDER` are what the C++ grid_calculator reads."""
    if head_dim not in LADDER:
        raise ValueError(f'head_dim {head_dim} is not on the gfx942 dQ ladder {LADDER}')
    if body(head_dim) == 'wide':
        # 32 query rows per wave up to 256, 16 above; a causal block pairs two q tiles.
        wave_m = 32 if head_dim <= 256 else 16
        return dict(block_m=4 * wave_m * (2 if causal else 1),
                    wave_m=wave_m,
                    num_waves=4,
                    block_n=wave_m,
                    waves_per_eu=1,
                    GRID_AXIS_ORDER=GRID_AXIS_TILE_FASTEST)
    return dict(block_m=256,
                num_waves=8,
                block_n=default_block_n(head_dim),
                waves_per_eu=2,
                lpt=default_lpt(causal),
                GRID_AXIS_ORDER=default_grid_axis_order(causal))


def builder_kwargs(knob_dict):
    """A `knobs()` result as the builder's keyword arguments."""
    kw = {k: v for k, v in knob_dict.items() if k not in ('block_m', 'GRID_AXIS_ORDER')}
    kw['grid_axis_order'] = knob_dict['GRID_AXIS_ORDER']
    return kw


# The tuning keywords each body's builder takes.
BODY_KNOBS = {
    'narrow': ('num_waves', 'block_n', 'waves_per_eu', 'lpt'),
    'wide': ('num_waves', 'block_n', 'waves_per_eu', 'wave_m'),
}


@dataclass(frozen=True)
class DqKnobs:
    """How to compute it. None means "policy decides"; `resolve` fills it in.
    `block_m` (q rows per grid block) and `GRID_AXIS_ORDER` feed the C++ grid."""
    body: str | None = None
    block_m: int | None = None
    num_waves: int | None = None
    block_n: int | None = None
    waves_per_eu: int | None = None
    lpt: bool | None = None
    wave_m: int | None = None
    GRID_AXIS_ORDER: int | None = None

    def resolve(self, meta: FmhaInputMetadata) -> 'DqKnobs':
        policy = dict(knobs(meta.head_dim, meta.causal_type != 0), body=body(meta.head_dim))
        return resolve_fields(self, policy)

    def builder_kwargs(self):
        """The tuning half of the body builder's keywords."""
        if self.body is None:
            raise ValueError('knobs must be resolved: call DqKnobs().resolve(meta) first')
        kw = {k: getattr(self, k) for k in BODY_KNOBS[self.body]}
        kw['grid_axis_order'] = self.GRID_AXIS_ORDER
        return kw
