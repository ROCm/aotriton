# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Knobs of the gfx942 forward (fmha_fwd_gfx942.py).

flydsl-free: aot/flyc_attn_fwd.py calls it from the code generator, which has
no flydsl. The builder takes its defaults from here, so the two cannot drift.
"""

from dataclasses import asdict, dataclass, fields, replace


@dataclass(frozen=True)
class FmhaInputMetadata:
    """What to compute, read off a functional's axes; shared by the forward,
    dQ and dK/dV. `head_dim` is the compiled tile (BLOCK_DMODEL)."""
    head_dim: int
    causal_type: int = 0
    dtype_str: str = 'bf16'
    bias: bool = False
    dropout: bool = False
    padded_head: bool = False

    def builder_kwargs(self):
        """The functional half of every gfx942 body builder's keywords."""
        return dict(causal_type=self.causal_type,
                    dtype_str=self.dtype_str,
                    bias=self.bias,
                    dropout=self.dropout,
                    padded_head=self.padded_head)


def resolve_fields(knobs, policy):
    """`knobs` with every None field taken from the `policy` dict."""
    return replace(knobs, **{f.name: policy[f.name] for f in fields(knobs)
                             if getattr(knobs, f.name) is None and f.name in policy})


def knob_dict(knobs):
    """A resolved knobs dataclass as the JSON dict recorded beside the hsaco."""
    return {k: v for k, v in asdict(knobs).items() if v is not None}


# The compiled head dims, served by three bodies with different register plans:
#   narrow (fmha_fwd_gfx942.py)        <= 128: 8 waves x 32 rows, 2 waves/SIMD
#   wide   (fmha_fwd_gfx942_wide.py)   <= 256: 4 waves x 32 rows, 1 wave/SIMD
#   xwide  (fmha_fwd_gfx942_xwide.py)  > 256:  8 waves x 16 rows, 16x16x16 MFMA
LADDER = (32, 64, 96, 128, 160, 192, 224, 256, 384, 512)

# (source file, @flyc.kernel name, builder) per body.
BODIES = {
    'narrow': ('fmha_fwd_gfx942.py', 'fmha_fwd_gfx942_kernel', 'build_fmha_fwd_gfx942'),
    'wide': ('fmha_fwd_gfx942_wide.py', 'fmha_fwd_gfx942_wide_kernel', 'build_fmha_fwd_gfx942_wide'),
    'xwide': ('fmha_fwd_gfx942_xwide.py', 'fmha_fwd_gfx942_xwide_kernel', 'build_fmha_fwd_gfx942_xwide'),
}

# csrc/flyc_common.h's kFlycGridAxis*.
GRID_AXIS_HEAD_FASTEST = 0
GRID_AXIS_TILE_FASTEST = 1


def default_block_n(head_dim):
    """Keep 32 MFMAs per KV tile: small heads take a longer tile."""
    return 64 if head_dim <= 64 else 32


def default_grid_axis_order(causal):
    """Tile-fastest shares K/V in L2 across a head's q tiles; causal needs
    head-fastest to spread its uneven tiles (B2 H64 S32k d128: 523 vs 495 TFLOPS
    non-causal; B4 H32 S8k d128 causal: 402 vs 295)."""
    return GRID_AXIS_HEAD_FASTEST if causal else GRID_AXIS_TILE_FASTEST


def body(head_dim):
    """Which body serves a compiled head dim."""
    if head_dim <= 128:
        return 'narrow'
    return 'wide' if head_dim <= 256 else 'xwide'


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
        raise ValueError(f'head_dim {head_dim} is not on the gfx942 ladder {LADDER}')
    if body(head_dim) == 'wide':
        # A causal block runs q tiles g and T-1-g, so it covers 2 * 128 rows; the
        # tile-fastest grid keeps K/V in L2 and the pairing evens out block lengths.
        return dict(block_m=256 if causal else 128,
                    num_waves=4,
                    block_n=32,
                    waves_per_eu=1,
                    rescale_threshold=0.0,
                    GRID_AXIS_ORDER=GRID_AXIS_TILE_FASTEST)
    if body(head_dim) == 'xwide':
        return dict(block_m=128,
                    num_waves=8,
                    block_n=16,
                    waves_per_eu=2,
                    rescale_threshold=0.0,
                    GRID_AXIS_ORDER=default_grid_axis_order(causal))
    return dict(block_m=256,
                num_waves=8,
                block_n=default_block_n(head_dim),
                waves_per_eu=2,
                rescale_threshold=0.0,
                GRID_AXIS_ORDER=default_grid_axis_order(causal))


# The tuning keywords each body's builder takes.
BODY_KNOBS = {
    'narrow': ('num_waves', 'block_n', 'waves_per_eu', 'rescale_threshold'),
    'wide': ('num_waves', 'block_n', 'waves_per_eu', 'rescale_threshold'),
    'xwide': ('num_waves', 'block_n', 'waves_per_eu', 'rescale_threshold'),
}


@dataclass(frozen=True)
class FwdKnobs:
    """How to compute it. None means "policy decides"; `resolve` fills it in.
    `block_m` (rows per grid block) and `GRID_AXIS_ORDER` feed the C++ grid."""
    body: str | None = None
    block_m: int | None = None
    num_waves: int | None = None
    block_n: int | None = None
    waves_per_eu: int | None = None
    rescale_threshold: float | None = None
    GRID_AXIS_ORDER: int | None = None

    def resolve(self, meta: FmhaInputMetadata) -> 'FwdKnobs':
        policy = dict(knobs(meta.head_dim, meta.causal_type != 0), body=body(meta.head_dim))
        return resolve_fields(self, policy)

    def builder_kwargs(self):
        """The tuning half of the body builder's keywords."""
        if self.body is None:
            raise ValueError('knobs must be resolved: call FwdKnobs().resolve(meta) first')
        kw = {k: getattr(self, k) for k in BODY_KNOBS[self.body]}
        kw['grid_axis_order'] = self.GRID_AXIS_ORDER
        return kw
