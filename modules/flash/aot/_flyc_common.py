# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Shared pieces of the flyc descriptions: mostly the BACKWARD ones'
(`flyc_bwd_dkdv.py`, `flyc_bwd_dq.py`); the head-dim ladders and the input
metadata serve the forward too.

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


# The compiled head-dim ladder per arch, shared by every flyc kernel (forward, dQ,
# dK/dV). Literals: the generator parses these modules and has no flydsl; the
# builders re-validate the tile, so a drift fails loudly at build.
#
# The kernels serve more rungs than these (gfx1201's tuning ladders have 384, and
# dK/dV 320 and 448; the MFMA kernels have 384, JIT), but no surveyed LLM uses a head
# dim in 257-511 (HF configs, 2026-11), so those run padded on 512.
#
# WMMA (gfx1201): from gfx1201.tuning_fwd._BLOCK_DMODEL_LADDER and its dQ/dK/dV twins.
FLYC_WMMA_HEAD_DIMS = frozenset({16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 512})
# MFMA (gfx950, gfx942; gfx90a would share it): the MFMA shapes limit it to
# multiples of 32 (PV_MFMA_N). 96 IS included, as in gfx950.tuning_fwd.LADDER. The
# note above that LADDER saying 96 "computes the wrong answer" predates the fix:
# ROCm/FlyDSL commit 98493cc7 ("gfx950 parity: fix the head_dim 96 wait-state
# hazard, 579 -> 932 TF"), which our vendored kernel carries
# (gfx950.dualwave_helpers.mfma_operand_wait_state).
FLYC_MFMA_HEAD_DIMS = frozenset({32, 64, 96, 128, 160, 192, 224, 256, 512})

# Which arches the flyc kernels serve, and each one's ladder.
FLYC_HEAD_DIMS_BY_ARCH = {
    'gfx1201': FLYC_WMMA_HEAD_DIMS,
    'gfx950': FLYC_MFMA_HEAD_DIMS,
    'gfx942': FLYC_MFMA_HEAD_DIMS,
}

# The BLOCK_DMODEL marker's options: one decorator stack serves every arch.
FLYC_HEAD_DIM_OPTIONS = sorted(set().union(*FLYC_HEAD_DIMS_BY_ARCH.values()))


def flyc_bwd_disabled(f):
    """Everything a flyc backward kernel cannot serve, in one predicate.

    Arch lives here rather than in a declaration for the same reason it does in
    the forward's `_flyc_fwd_disabled`: it is one more exclusion among several,
    and splitting it across two mechanisms means two places to look when a
    functional unexpectedly has no flyc kernel.
    """
    if f.arch not in FLYC_HEAD_DIMS_BY_ARCH:
        return True
    # Also excludes causal+bias, which both kernels assert against directly
    # ("bias and causal are mutually exclusive, as in the forward").
    if flash_disabled(f):
        return True
    # f16/bf16 WMMA only.
    if '*fp32' in check_value(f, ['Q']):
        return True
    # Flyc kernels are not specialized for hdim that cannot be addressed by MFMA/WMMA directly.
    if check_value(f, ['BLOCK_DMODEL']) not in FLYC_HEAD_DIMS_BY_ARCH[f.arch]:
        return True
    return False


def flyc_input_metadata(choices):
    """The problem a flyc functional describes, for every kernel and arch."""
    from fmha_meta import FmhaInputMetadata
    # One head-dim axis, so V's tile is QK's (block_dmodel_v left None). The
    # real head dims arrive at run time; PADDED_HEAD says whether they may be
    # narrower than the tile.
    tile = choices.BLOCK_DMODEL
    # `num_heads` reaches a kernel only through STRIDE_TOKEN, which the arch
    # tuners keep behind strides_constexpr=False; 1 keeps it off the axes.
    return FmhaInputMetadata(num_heads=1,
                             dtype_str='bf16' if '*bf16' in choices.arg('Q') else 'f16',
                             causal_type=choices.CAUSAL_TYPE,
                             block_dmodel=tile,
                             padded_head=choices.PADDED_HEAD,
                             bias=bool(choices.BIAS_TYPE),
                             dropout=bool(choices.ENABLE_DROPOUT))
