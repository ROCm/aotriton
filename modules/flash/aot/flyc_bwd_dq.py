# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI description of the flash bwd dQ/dB FlyDSL backend (gfx1201 and gfx950).

The dQ/dB half of the flyc backward, and the sibling of `flyc_bwd_dkdv.py` --
see that file for the shared rationale (no perf axes, no functional axes of its
own, a declared kernel-argument list, and why the backend is a metro with a
Triton `bwd_preprocess` in front).

Two things are specific to this kernel:

* **It writes dB.** dB is `dS`, which the kernel already forms for its last
  GEMM. dK/dV has the same quantity but walks Q tiles for a fixed KV block, so
  its eight elements are eight q *rows* at one kv column and the store would go
  down a dB column; this kernel writes along a row instead. That is why dB is
  the dQ kernel's output in both DSLs, and why both arches write it whenever
  bias is present, matching `bwd_kernel_dq.py`'s DB operand. gfx1201 gets there
  with no knob at all -- FlyDSL 077626cd dropped `return_dbias`, so `bias=True`
  always emits the store -- while gfx950 kept a `store_db` `const_expr` that
  defaults *off*, which the builder below has to pin from `BIAS_TYPE`.
* **`block_m` and `block_n` are both real knobs** here, so nothing has to be
  re-derived for the grid the way `flyc_bwd_dkdv.py` re-derives `block_n`.
"""

from dataclasses import asdict

import aotriton.template_instantiation as ati
from ._flyc_common import FlycBwdHints, flyc_bwd_disabled


# The compiled tile ladder, from fmha_tuning_bwd_dq_gfx1201._BLOCK_DMODEL_LADDER.
# A literal, not an import, for the reason flyc_bwd_dkdv.py records. Narrower
# than dK/dV's -- no 320 or 448 -- which is exactly why the ladder is a per-kernel
# constant rather than something shared in _flyc_common.py.
FLYC_BWD_DQ_HEAD_DIMS = frozenset({16, 32, 48, 64, 80, 96, 128, 160, 192, 224,
                                   256, 384, 512})

# gfx950's ladder, from fmha_tuning_bwd_dq_gfx950.BWD_DQ_LADDER (itself
# `fmha_tuning_gfx950.LADDER`, imported there unchanged). A literal for the same
# reason as FLYC_BWD_DQ_HEAD_DIMS above -- including 96, per the forward's note
# on that rung (the stale-comment/live-tuple discrepancy lives in the forward's
# tuning module; this ladder is the same tuple, so the same ruling applies).
FLYC_GFX950_BWD_DQ_HEAD_DIMS = frozenset({32, 64, 96, 128, 160, 192, 224, 256, 384, 512})

_FLYC_BWD_DQ_HEAD_DIMS_BY_ARCH = {
    'gfx1201': FLYC_BWD_DQ_HEAD_DIMS,
    'gfx950': FLYC_GFX950_BWD_DQ_HEAD_DIMS,
}


def _flyc_bwd_dq_disabled(f):
    return flyc_bwd_disabled(f, head_dims=_FLYC_BWD_DQ_HEAD_DIMS_BY_ARCH)


@ati.start
@ati.disable(when=_flyc_bwd_dq_disabled,
             I_understand_this_overrides_cited_disable=True)
@ati.cite('op_attn_bwd.triton_split.bwd_kernel_dq')
#
# --- the kernarg ABI, in `bwd_dq_kernel` order -------------------------------
#
# The kernel, NOT the Python launcher: the two differ by `stream` and by
# `batch_size` (see flyc_bwd_dkdv.py). Same convention as the forward and dK/dV
# since FlyDSL 1b58fb93 -- inputs, outputs, LSE/Delta, seqinfo, scalars, strides
# -- with (DQ, DB) where dK/dV has (DK, DV).
@ati.tensor('Q',     'T_io', rank=4, strides='stride_q_*',  wires_to='Q')
@ati.tensor('K',     'T_io', rank=4, strides='stride_k_*',  wires_to='K')
@ati.tensor('V',     'T_io', rank=4, strides='stride_v_*',  wires_to='V')
@ati.tensor('B',     'T_io', rank=4, strides='stride_b_*',  wires_to='B')
@ati.tensor('DO',    'T_io', rank=4, strides='stride_do_*', wires_to='DO')
@ati.tensor('DQ',    'T_io', rank=4, strides='stride_dq_*', wires_to='DQ')
# DB has its OWN stride triple, not B's. FlyDSL made that split deliberately
# (fmha_abi_gfx1201.bias_args: "B and DB are separate tensors and may be laid
# out differently ... deriving one from the other happens to work whenever both
# are contiguous, and writes to the wrong addresses the moment either is a
# view"). `stride_b_*` does not match `stride_db_*`, so the two globs stay
# disjoint on their own.
@ati.tensor('DB',    'T_io', rank=4, strides='stride_db_*', wires_to='DB')
@ati.tensor('LSE',   '*fp32:16', rank=2, wires_to='L')
@ati.tensor('Delta', '*fp32:16', rank=2, wires_to='D')
@ati.tensor('seqinfo_q0', '*i32:16', rank=1, wires_to='seqinfo_q0')
@ati.tensor('seqinfo_q1', '*i32:16', rank=1, wires_to='seqinfo_q1')
@ati.tensor('seqinfo_k0', '*i32:16', rank=1, wires_to='seqinfo_k0')
@ati.tensor('seqinfo_k1', '*i32:16', rank=1, wires_to='seqinfo_k1')
# `varlen_bits` is a plain rename; `num_seqlens` has no operand to wire to and
# is computed from the decoded layout word and Q's extents. See
# flyc_bwd_dkdv.py.
@ati.scalar('varlen_bits', 'i32', wires_to='varlen_bits')
@ati.scalar('num_seqlens', 'i32', wires_to=ati.context_helper('flyc_num_seqlens'))
@ati.scalar('max_seqlen_q', 'i32', wires_to='max_seqlen_q')
@ati.scalar('max_seqlen_k', 'i32', wires_to='max_seqlen_k')
@ati.scalar('window_left',  'i32', wires_to='Window_left')
@ati.scalar('window_right', 'i32', wires_to='Window_right')
@ati.tensor('philox_seed_ptr', '*u64', rank=0)
@ati.tensor('philox_offset1', '*u64', rank=0)
@ati.scalar('philox_offset2', 'u64')
@ati.scalar('idropout_p',    'i32',  wires_to=ati.context_helper('flyc_idropout_p'))
@ati.scalar('dropout_scale', 'fp32', wires_to=ati.context_helper('flyc_dropout_scale'))
@ati.scalar('num_head_q', 'i32',  wires_to='num_head_q')
@ati.scalar('num_head_k', 'i32',  wires_to='num_head_k')
@ati.scalar('hdim_qk',    'i32',  wires_to='hdim_qk')
@ati.scalar('hdim_vo',    'i32',  wires_to='hdim_vo')
@ati.scalar('sm_scale',   'fp32', wires_to='sm_scale')
#
# --- functional-axis markers, NOT kernel arguments ----------------------------
#
# See flyc_attn_fwd.py for the full rationale (same mechanism, same reason:
# this backend's compiled ladder, FLYC_BWD_DQ_HEAD_DIMS, is a strict subset of
# the operator's BLOCK_DMODEL axis). Kept out of the ordered kernarg block
# above -- BLOCK_DMODEL/PADDED_HEAD are not part of `bwd_dq_kernel`'s signature
# at all, they are build-time Python values the operator owns.
#
# One shared marker for both arches (this decorator stack does not vary by
# arch), so `options=` is the union of both ladders -- see flyc_attn_fwd.py's
# BLOCK_DMODEL marker for why a single arch's ladder would under-document.
@ati.scalar('BLOCK_DMODEL',
            options=sorted(FLYC_BWD_DQ_HEAD_DIMS | FLYC_GFX950_BWD_DQ_HEAD_DIMS),
            wires_to=ati.context_helper('flyc_block_dmodel'))
@ati.scalar('PADDED_HEAD', options=[False, True],
            wires_to=ati.context_helper('flyc_padded_head'))
@ati.flyc.hints(FlycBwdHints)
@ati.flyc.kernel()
def flyc_bwd_dq(arch, choices, hints):
    """Build one dQ/dB hsaco for the functional described by `choices`.

    Build-time only (`aotriton.flyc_compile`); the generator calls this for its
    knobs and never calls the returned `build`. See `flyc_attn_fwd.py` for the
    `choices` / `hints` contract, and for the invariant this owes (`meta` a pure
    function of `choices` alone) -- true here on both arches for the same reason
    it is there.
    """
    if arch == 'gfx1201':
        from fmha_tuning_bwd_dq_gfx1201 import (
            BwdDqInputMetadata, BwdDqKnobs, resolve_knobs,
        )

        tile = choices.BLOCK_DMODEL
        meta = BwdDqInputMetadata(
            # Unread by resolve_knobs, whose policy keys on head_dim and causal.
            num_heads=1,
            head_dim=tile,
            # Both the same, because AOTriton has ONE BLOCK_DMODEL axis and this
            # kernel has one compiled tile (unlike dK/dV, which carries a second
            # block_dmodel_v). The tile covers the wider of the two extents and the
            # narrower axis rides as a masked RUNTIME extent -- the kernel's
            # `vo_cols` is `MaskedAxis(hdim_vo, active=PADDED_HEAD)`.
            #
            # Which makes PADDED_HEAD load-bearing rather than cosmetic: with the
            # flag off the V/O axis is not bounded at all and a narrower V is walked
            # to the full tile. AOTriton sets it as
            # `hdim_qk != hdim_rounded || hdim_vo != hdim_rounded`
            # (modules/flash/csrc/attn_bwd.cc), which is exactly the condition
            # upstream's resolve_knobs derives when asked to choose -- so passing
            # `choices.PADDED_HEAD` below is the same answer, not a coincidence.
            head_dim_v=tile,
            causal=choices.CAUSAL_TYPE != 0,
            causal_type=choices.CAUSAL_TYPE,
            dtype_str='bf16' if '*bf16' in choices.arg('Q') else 'f16',
            bias=bool(choices.BIAS_TYPE),
            dropout=bool(choices.ENABLE_DROPOUT),
            # philox_width left at None -- Philox.for_arch(), matching the forward.
            # The backward has to reproduce the forward's mask exactly.
        )
        knobs = resolve_knobs(meta, BwdDqKnobs(
            block_dmodel=tile,
            padded_head=choices.PADDED_HEAD,
        ))
        assert knobs.block_dmodel == tile, (
            f'resolve_knobs returned block_dmodel={knobs.block_dmodel} for '
            f'BLOCK_DMODEL={tile}; the compiled tile must be the operator axis')

        def build(knobs=knobs):
            """Deferred: constructs the FlyDSL module. Imports flydsl transitively,
            so ONLY `aotriton.flyc_compile` (run by ninja) may call this.

            `knobs` is bound here as a default rather than captured: the name is
            rebound to the JSON knob dict below, and a live closure would follow
            it and hand a `dict` to a function expecting `BwdDqKnobs`. The driver
            calls this with no arguments."""
            from fmha_bwd_dq_gfx1201_kernel import build_bwd_dq_module_primary
            return build_bwd_dq_module_primary(meta, knobs)

        # Two plain strings: the vendored file, relative to
        # modules/flash/flyc/, and the `@flyc.kernel` def's own name inside it.
        # Read by codegen/flytune.py and flyc_compile.py off the `build`
        # closure WITHOUT ever calling it.
        build.flyc_source = 'fmha_bwd_dq_gfx1201_kernel.py'
        build.flyc_kernel_name = 'bwd_dq_kernel'

        # `block_m` is already a knob here, so the dict needs nothing added for
        # it -- unlike flyc_bwd_dkdv.py, which has to re-derive its grid's
        # block_n. `GRID_AXIS_ORDER` still needs supplying by hand though:
        # gfx1201's BwdDqKnobs has no such field, and this kernel walks
        # (head, q_tile, seq) -- HEAD_FASTEST -- same as the forward.
        knobs = asdict(knobs)  # past here `knobs` is the dict, not BwdDqKnobs
        knobs['GRID_AXIS_ORDER'] = 0  # HEAD_FASTEST; fmha_tuning_gfx950.GRID_AXIS_HEAD_FASTEST
        return build, knobs

    elif arch == 'gfx950':
        from fmha_tuning_bwd_dq_gfx950 import bwd_dq_knobs
        # BwdDqKnobs reuses fmha_tuning_gfx950.FmhaInputMetadata verbatim
        # (not its own metadata class) -- see that module: only the forward
        # and dK/dV declare their own.
        from fmha_tuning_gfx950 import FmhaInputMetadata as Gfx950InputMetadata

        tile = choices.BLOCK_DMODEL
        meta = Gfx950InputMetadata(
            num_heads=1,
            head_dim=tile,
            # No `causal_type` field here either (same shape as the forward's
            # gfx950 branch); the kernel only ever compiles CAUSAL_TYPE in
            # {0, 3}, so this is 1:1, no mapping.
            causal=choices.CAUSAL_TYPE != 0,
            window=choices.CAUSAL_TYPE != 0,
            dtype_str='bf16' if '*bf16' in choices.arg('Q') else 'f16',
            bias=bool(choices.BIAS_TYPE),
            dropout=bool(choices.ENABLE_DROPOUT),
        )
        knobs = bwd_dq_knobs(
            arch,
            block_dmodel=tile,
            padded_head=choices.PADDED_HEAD,
            # Pinned for the same reason as the forward's gfx950 branch:
            # this kernel's own real signature carries no folded constexpr
            # parameter, but the pin keeps the build configuration uniform
            # across all three gfx950 descriptions rather than varying it
            # kernel-by-kernel for no functional reason.
            paged=False,
            num_kv_splits=1,
            # dB is this kernel's second output, and `store_db` is the
            # `const_expr` that decides whether the store exists at all
            # (`fmha_bwd_dq_gfx950.py`'s `if const_expr(STORE_DB)`). Upstream's
            # `_BWD_DQ_FALLBACK` defaults it to False -- inference builds have
            # no gradient to write -- and it is not derived from `meta.bias`,
            # so an unpinned build with BIAS_TYPE=1 compiles a kernel that
            # returns success and leaves dB untouched. Same defect shape as the
            # forward's `return_lse` (see flyc_attn_fwd.py), with one real
            # difference: BIAS_TYPE *is* a functional axis, so this one is
            # genuinely a build-time question and specialising on it is right.
            #
            # gfx1201 reaches the same place without a knob -- FlyDSL 077626cd
            # dropped its `return_dbias` and `bias=True` always emits the store
            # (see this module's docstring). Both arches therefore write dB
            # whenever bias is present, and neither implements Triton's extra
            # runtime out: `bwd_kernel_dq.py` also clears `store_db` when the
            # caller passes all-zero dB strides, which is how a bias that does
            # not require grad asks for the store to be skipped.
            store_db=bool(choices.BIAS_TYPE),
        ).resolve(meta)
        assert knobs.store_db == bool(choices.BIAS_TYPE), (
            f'resolve() returned store_db={knobs.store_db} for '
            f'BIAS_TYPE={choices.BIAS_TYPE}; dB is written exactly when bias is')
        assert knobs.block_dmodel == tile, (
            f'resolve() returned block_dmodel={knobs.block_dmodel} for '
            f'BLOCK_DMODEL={tile}; the compiled tile must be the operator axis')
        assert not knobs.paged and knobs.num_kv_splits == 1, (
            'AOT gfx950 pins paged=False, num_kv_splits=1 uniformly across the '
            'three descriptions; resolve() must not have overridden it')

        def build():
            """Deferred: constructs the FlyDSL module. Imports flydsl transitively,
            so ONLY `aotriton.flyc_compile` (run by ninja) may call this."""
            from fmha_bwd_dq_gfx950 import build_fmha_bwd_dq_gfx950_module_primary
            return build_fmha_bwd_dq_gfx950_module_primary(meta, knobs)

        build.flyc_source = 'fmha_bwd_dq_gfx950.py'
        build.flyc_kernel_name = 'fmha_bwd_dq_gfx950_kernel'

        # GRID_AXIS_ORDER is a flat resolved field on Gfx950Knobs/BwdDqKnobs
        # already -- no mirroring needed.
        return build, asdict(knobs)

    else:
        assert False, f'flyc_bwd_dq: no builder branch for arch {arch!r}'
