# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Shared pieces of the flyc BACKWARD descriptions (`flyc_bwd_dkdv.py`,
`flyc_bwd_dq.py`).

Separate from `_common.py`, which is the *Triton* flash descriptions' shared
vocabulary — `flash_disabled` is imported from there below rather than
reimplemented, but nothing flyc-specific goes back the other way.

The forward (`flyc_attn_fwd.py`) keeps its own copies. Two backward kernels
that are two halves of one operation genuinely share a hints dataclass and a
disable rule; the forward shares neither with them (it has a different ladder,
and its own extra `philox_seed_output`/`philox_offset_output` surface), so
hoisting its versions here would be grouping by spelling rather than by fact.
"""

from dataclasses import dataclass

from ._common import flash_disabled, check_value


@dataclass
class FlycBwdHints:
    """Tuning inputs a flyc backward builder may read that are NOT functional axes.

    The backward analogue of `flyc_attn_fwd.FlycFwdHints`, and shared by both
    kernels because dK/dV and dQ are two halves of one backward pass: a caller
    who knows the sequence lengths knows them for both, and a schedule that
    starts varying with them will vary for both.

    Defaults are what every build passes today. `resolve_knobs` reads none of
    these in either module, so every build is hint-independent until FlyDSL's
    backward tuner grows a seqlen dependence — and `codegen/root.py`'s
    `write_flyc_hsaco` emits an empty HINTS column, so nothing could vary them
    yet even if it did.
    """
    seqlen_q: int = 0        # 0 = unknown/any; a real value once the tuner uses it
    seqlen_k: int = 0
    num_heads: int = 0
    batch: int = 0


def flyc_bwd_disabled(f, *, head_dims):
    """Everything a flyc backward kernel cannot serve, in one predicate.

    `head_dims` is a `{arch: ladder}` mapping: the caller's own
    `FLYC_BWD_*_HEAD_DIMS`, one ladder per arch it serves, since dK/dV and dQ
    each have their own per-arch ladders (dK/dV compiles 320 and 448 as well on
    gfx1201; gfx950's ladder is the same `(32, ..., 512)` tuple for both
    kernels, but is still looked up per arch here rather than assumed).

    Arch lives here rather than in a declaration for the same reason it does in
    the forward's `_flyc_fwd_disabled`: it is one more exclusion among several,
    and splitting it across two mechanisms means two places to look when a
    functional unexpectedly has no flyc kernel.
    """
    if f.arch not in head_dims:
        return True
    # Also excludes causal+bias, which both kernels assert against directly
    # ("bias and causal are mutually exclusive, as in the forward").
    if flash_disabled(f):
        return True
    # f16/bf16 WMMA only.
    if '*fp32' in check_value(f, ['Q']):
        return True
    # Off-ladder head dims are rejected outright by `resolve_knobs`/`resolve()`,
    # not rounded: rounding is the *interface*'s job, because it also has to
    # arrange the runtime extent and the padded-head contract that make the
    # rounding safe.
    if check_value(f, ['BLOCK_DMODEL']) not in head_dims[f.arch]:
        return True
    return False
