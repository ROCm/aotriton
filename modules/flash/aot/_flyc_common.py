# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Shared pieces of the flyc BACKWARD descriptions (`flyc_bwd_dkdv.py`,
`flyc_bwd_dq.py`).

Separate from `_common.py`, which is the *Triton* flash descriptions' shared
vocabulary — this module may import from there, but nothing flyc-specific goes
back the other way.

The forward (`flyc_attn_fwd.py`) keeps its own copies. Two backward kernels
that are two halves of one operation genuinely share a hints dataclass and a
disable rule; the forward shares neither with them (it has a different ladder,
and its own extra `philox_seed_output`/`philox_offset_output` surface), so
hoisting its versions here would be grouping by spelling rather than by fact.
"""

from dataclasses import dataclass


@dataclass
class FlycBwdHints:
    """Tuning inputs a flyc backward builder may read that are NOT functional axes.

    The backward analogue of `flyc_attn_fwd.FlycFwdHints`, and shared by both
    kernels because dK/dV and dQ are two halves of one backward pass: a caller
    who knows the sequence lengths knows them for both, and a schedule that
    starts varying with them will vary for both.

    Defaults are what every build passes today. `resolve_knobs` reads none of
    these in either module, so every build is hint-independent until FlyDSL's
    backward tuner grows a seqlen dependence.
    """
    seqlen_q: int = 0        # 0 = unknown/any; a real value once the tuner uses it
    seqlen_k: int = 0
    num_heads: int = 0
    batch: int = 0
