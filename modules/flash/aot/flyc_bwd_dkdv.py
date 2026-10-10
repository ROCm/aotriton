# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI description of the flash bwd dK/dV FlyDSL backend (gfx1201 and gfx950).

The backward half of `flyc_attn_fwd.py`, and it works the same way: no perf
axes (`fmha_tuning_bwd_dkdv_gfx1201.resolve_knobs` is the sole producer of a
schedule), no functional axes of its own (it inherits `op_attn_bwd`'s and
narrows them with `@ati.disable`), and a declared kernel-argument list, because
the hsaco's layout comes from `bwd_dkdv_kernel` and nothing else can supply
it.

What differs from the forward, beyond the operand list:

* **It does not stand alone.** dK/dV recomputes P from the logsumexp the
  forward stored and needs `Delta = rowsum(dO * O)`, which FlyDSL has no kernel
  for. AOTriton's Triton `bwd_preprocess` computes exactly that, in fp32, so
  the backend is a metro (`metro_bwd_flyc` in `__init__.py`) that runs the
  Triton preprocess and then this kernel and `flyc_bwd_dq`.
* **The tile-width knob is derived, not resolved, on this arch.** See the knob
  note at the bottom.
"""

from dataclasses import asdict

import aotriton.template_instantiation as ati
from ._flyc_common import FlycBwdHints, flyc_bwd_disabled


# The compiled tile ladder, from fmha_tuning_bwd_dkdv_gfx1201._BLOCK_DMODEL_LADDER.
# A literal, not an import: the generator parses this module and has no flydsl.
# `build` (build time) does import it, and `resolve_knobs` re-validates the tile,
# so a drift here fails loudly at build rather than silently emitting the wrong
# kernel. Wider than the forward's -- 320 and 448 compile here.
FLYC_BWD_DKDV_HEAD_DIMS = frozenset({16, 32, 48, 64, 80, 96, 128, 160, 192, 224,
                                     256, 320, 384, 448, 512})

# gfx950's ladder, from fmha_tuning_bwd_dkdv_gfx950.LADDER (that module's own
# copy, NOT imported from fmha_tuning_gfx950 -- see that file's header). A
# literal for the same reason as FLYC_BWD_DKDV_HEAD_DIMS above -- including 96,
# per the forward's note on that rung (the same tuple, same ruling).
FLYC_GFX950_BWD_DKDV_HEAD_DIMS = frozenset({32, 64, 96, 128, 160, 192, 224, 256, 384, 512})

_FLYC_BWD_DKDV_HEAD_DIMS_BY_ARCH = {
    'gfx1201': FLYC_BWD_DKDV_HEAD_DIMS,
    'gfx950': FLYC_GFX950_BWD_DKDV_HEAD_DIMS,
}


def _flyc_bwd_dkdv_disabled(f):
    return flyc_bwd_disabled(f, head_dims=_FLYC_BWD_DKDV_HEAD_DIMS_BY_ARCH)


@ati.start
# The cited disable is the WEAKER of the two, despite the kwarg's name: it is
# flash_disabled(f, gfx950_bad_hdims={48, 80}), and the predicate above rejects
# every arch but the two this kernel serves, which leaves parts of the cited
# predicate unreachable. Same argument the forward records; widening the arch
# gate expires it.
@ati.disable(when=_flyc_bwd_dkdv_disabled,
             I_understand_this_overrides_cited_disable=True)
# `cite` fills the GAPS: any argument below that this description does not fully
# claim is cloned from the Triton dK/dV kernel by apparel name, so the two
# backends cannot drift on a shared operand's type.
@ati.cite('op_attn_bwd.triton_split.bwd_kernel_dk_dv')
#
# --- the kernarg ABI, in `bwd_dkdv_kernel` order -----------------------------
#
# The kernel, NOT the Python launcher. The two differ by `stream` and by
# `batch_size`, which left the kernarg in FlyDSL 67a3ace0 and survives only
# host-side, where it is half of the grid's z extent. AOTriton computes its own
# grid, so nothing here declares it; see FlycBwdDkdvContext::grid_calculator().
#
# The order is frozen and load-bearing. FlyDSL 1b58fb93 put the forward and all
# three backward kernels on one convention, which is why this block reads like
# flyc_attn_fwd.py's with the outputs swapped in: inputs, outputs, LSE/Delta,
# seqinfo, scalars, strides.
#
# `rank=4` with only THREE strides is the declaration that the last dimension is
# unit-stride and has no argument -- FlyDSL requires stride(3) == 1 and reads no
# D-axis stride at all. `contiguous=-1` would be actively WRONG here: it indexes
# the matched stride list, which has three entries, so it would mark the seq
# stride as the unit one.
@ati.tensor('Q',     'T_io', rank=4, strides='stride_q_*',  wires_to='Q')
@ati.tensor('K',     'T_io', rank=4, strides='stride_k_*',  wires_to='K')
@ati.tensor('V',     'T_io', rank=4, strides='stride_v_*',  wires_to='V')
# B is (B, H, Sq, Sk): its three strides are batch, head and QUERY ROW, because
# the axis the KV tile walks is the contiguous one. Same shape as the forward's.
@ati.tensor('B',     'T_io', rank=4, strides='stride_b_*',  wires_to='B')
@ati.tensor('DO',    'T_io', rank=4, strides='stride_do_*', wires_to='DO')
@ati.tensor('DK',    'T_io', rank=4, strides='stride_dk_*', wires_to='DK')
@ati.tensor('DV',    'T_io', rank=4, strides='stride_dv_*', wires_to='DV')
# LSE and Delta are always compact: the kernel derives both pitches from
# LSE_LAYOUT, num_head_q and the token count, so neither has stride arguments
# by design. They share one offset computation in the kernel, which is why
# FlyDSL's host ABI checks them with one function (`row_tensor_arg`).
#
# `Delta` is AOTriton's `D`, and `D` is a LazyTensor -- the metro's Triton
# preprocess step materialises it. `LazyTensorInternal::kparam_data_ptr()`
# forces it first, so the flyc kernarg vector reads no differently from a plain
# tensor's.
@ati.tensor('LSE',   '*fp32:16', rank=2, wires_to='L')
@ati.tensor('Delta', '*fp32:16', rank=2, wires_to='D')
# varlen seqinfo. Identity wiring since upstream's varlen_bits port adopted the
# same role-based names; see flyc_attn_fwd.py for the history.
@ati.tensor('seqinfo_q0', '*i32:16', rank=1, wires_to='seqinfo_q0')
@ati.tensor('seqinfo_q1', '*i32:16', rank=1, wires_to='seqinfo_q1')
@ati.tensor('seqinfo_k0', '*i32:16', rank=1, wires_to='seqinfo_k0')
@ati.tensor('seqinfo_k1', '*i32:16', rank=1, wires_to='seqinfo_k1')
#
# --- the layout word, and the count derived from it ---------------------------
#
# `varlen_bits` IS AOTriton's `varlen_bits`: FlyDSL and Triton share one varlen
# ABI, so the same u32 word describes the layout to both and this is a plain
# rename like the ones below it.
@ati.scalar('varlen_bits', 'i32', wires_to='varlen_bits')
# `num_seqlens` has no operand to wire to. It is computed host side, from the
# decoded `varlen_bits` and the Q tensor's own extents: FlyDSL wants how many
# sequences are packed into a 1THD tensor, zero when the Q side is not packed,
# and branches on it directly. AOTriton has no field carrying that -- only the
# layout word it is derived from. `ati.context_helper` is how that host-side
# code is declared, for the reason flyc_attn_fwd.py sets out at length; the
# implementation is shared across all three flyc kernels via csrc/flyc_common.h,
# and it is also where an underivable sequence count is rejected. See
# flyc_attn_fwd.py.
@ati.scalar('num_seqlens', 'i32', wires_to=ati.context_helper('flyc_num_seqlens'))
#
# --- plain renames -----------------------------------------------------------
#
# Note the case: op_attn_bwd spells these lowercase where op_attn_fwd
# capitalises them (`max_seqlen_q` vs `Max_seqlen_q`), so these are NOT the same
# strings as the forward description's.
@ati.scalar('max_seqlen_q', 'i32', wires_to='max_seqlen_q')
@ati.scalar('max_seqlen_k', 'i32', wires_to='max_seqlen_k')
@ati.scalar('window_left',  'i32', wires_to='Window_left')
@ati.scalar('window_right', 'i32', wires_to='Window_right')
# PRNG: 1:1 with the Triton kernel. Declared one per line, in kernel order,
# because philox_offset2 is a scalar sitting BETWEEN the two pointers -- the
# grouped `@ati.tensor([...])` form would push it to the end and get the
# declared kernarg order wrong. Unlike the forward there is no
# philox_seed_output/philox_offset_output: only the forward reports what it drew.
@ati.tensor('philox_seed_ptr', '*u64', rank=0)
@ati.tensor('philox_offset1', '*u64', rank=0)
@ati.scalar('philox_offset2', 'u64')
#
# --- the two dropout arguments that are not a rename -------------------------
#
# AOTriton carries a float `dropout_p`; FlyDSL wants the i32 threshold and the
# 1/(1-p) scale that `philox.dropout_threshold` and `dropout_args` produce.
@ati.scalar('idropout_p',    'i32',  wires_to=ati.context_helper('flyc_idropout_p'))
@ati.scalar('dropout_scale', 'fp32', wires_to=ati.context_helper('flyc_dropout_scale'))
#
# --- plain renames, continued ------------------------------------------------
@ati.scalar('num_head_q', 'i32',  wires_to='num_head_q')
@ati.scalar('num_head_k', 'i32',  wires_to='num_head_k')
@ati.scalar('hdim_qk',    'i32',  wires_to='hdim_qk')
@ati.scalar('hdim_vo',    'i32',  wires_to='hdim_vo')
@ati.scalar('sm_scale',   'fp32', wires_to='sm_scale')
#
# --- functional-axis markers, NOT kernel arguments ----------------------------
#
# See flyc_attn_fwd.py for the full rationale (same mechanism, same reason:
# this backend's compiled ladder, FLYC_BWD_DKDV_HEAD_DIMS, is a strict subset
# of the operator's BLOCK_DMODEL axis). Kept out of the ordered kernarg block
# above -- BLOCK_DMODEL/PADDED_HEAD are not part of `bwd_dkdv_kernel`'s
# signature at all, they are build-time Python values the operator owns.
#
# One shared marker for both arches (this decorator stack does not vary by
# arch), so `options=` is the union of both ladders -- see flyc_attn_fwd.py's
# BLOCK_DMODEL marker for why a single arch's ladder would under-document.
@ati.scalar('BLOCK_DMODEL',
            options=sorted(FLYC_BWD_DKDV_HEAD_DIMS | FLYC_GFX950_BWD_DKDV_HEAD_DIMS),
            wires_to=ati.context_helper('flyc_block_dmodel'))
@ati.scalar('PADDED_HEAD', options=[False, True],
            wires_to=ati.context_helper('flyc_padded_head'))
@ati.flyc.hints(FlycBwdHints)
@ati.flyc.kernel()
def flyc_bwd_dkdv(arch, choices, hints):
    """Build one dK/dV hsaco for the functional described by `choices`.

    Executed by `aotriton.flyc_compile` at build time, in a venv that has
    flydsl — never by the generator, which only reads the decorators above and
    the `knobs` half of this function's return. See `flyc_attn_fwd.py` for the
    full account of `choices` (a `ChoiceView`, backed by a real `Functional`
    generator-side and by parsed `--signature` text driver-side) and `hints`,
    and for the invariant this owes (`meta` a pure function of `choices` alone)
    -- true here on both arches for the same reason it is there.
    """
    if arch == 'gfx1201':
        # ONLY the flydsl-free tuning module at call time. The FlyDSL-bearing
        # import lives inside `build()` below, so the code generator -- which
        # calls this function but never the callable -- never imports flydsl.
        from fmha_tuning_bwd_dkdv_gfx1201 import (
            ROWS_PER_WAVE, BwdDkDvKnobs, BwdDkDvMetadata, resolve_knobs,
        )

        tile = choices.BLOCK_DMODEL
        meta = BwdDkDvMetadata(
            # `num_heads` is not a functional axis. The forward pins it to 1 for a
            # reason specific to its STRIDE_TOKEN/STRIDES_CONSTEXPR arm; here it is
            # simply unread by resolve_knobs, whose policy keys on head_dim,
            # head_dim_v and causal only.
            num_heads=1,
            head_dim=tile,
            # AOTriton has ONE BLOCK_DMODEL axis, so the compiled QK and VO widths
            # are the same; the real extents ride as runtime hdim_qk/hdim_vo.
            head_dim_v=tile,
            # FlyDSL's causal_type IS AOTriton's CAUSAL_TYPE (0 none / 1 top-left /
            # 2 bottom-right / 3 window), and the operator's axis offers {0, 3}.
            causal=choices.CAUSAL_TYPE != 0,
            causal_type=choices.CAUSAL_TYPE,
            dtype_str='bf16' if '*bf16' in choices.arg('Q') else 'f16',
            bias=bool(choices.BIAS_TYPE),
            dropout=bool(choices.ENABLE_DROPOUT),
            # philox_width left at None -- Philox.for_arch(), which is what the
            # forward compiled against. The backward must reproduce the forward's
            # mask exactly, so this is the one setting that is not a free choice.
        )
        knobs = resolve_knobs(meta, BwdDkDvKnobs(
            block_dmodel=tile,
            block_dmodel_v=tile,
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
            it and hand a `dict` to a function expecting `BwdDkDvKnobs`. The
            driver calls this with no arguments."""
            from fmha_bwd_dkdv_gfx1201_kernel import build_bwd_dkdv_module_primary
            return build_bwd_dkdv_module_primary(meta, knobs)

        # Two plain strings: the vendored file, relative to
        # modules/flash/flyc/, and the `@flyc.kernel` def's own name inside it.
        # Read by codegen/flytune.py and flyc_compile.py off the `build`
        # closure WITHOUT ever calling it.
        build.flyc_source = 'fmha_bwd_dkdv_gfx1201_kernel.py'
        build.flyc_kernel_name = 'bwd_dkdv_kernel'

        # BLOCK_N is what the grid's x extent divides Max_seqlen_k by, and unlike
        # the dQ kernel's it is NOT a knob: fmha_bwd_dkdv_gfx1201_kernel.py derives
        # it inside the builder as `ROWS_PER_WAVE * NUM_TEAMS`, which would leave
        # grid_calculator() with no way to ask for it.
        #
        # Mirrored here from that derivation, whose three inputs ARE all knobs:
        #
        #     TEAM_WAVES = 2 * CONTRACTION_SHARDS if SPLIT_HEAD_DIM else 1
        #     NUM_TEAMS  = NUM_WAVES // TEAM_WAVES
        #     BLOCK_N    = ROWS_PER_WAVE * NUM_TEAMS
        #
        # A divergence between the two copies is a wrong grid rather than a build
        # failure, so this is a real coupling to the vendored file -- check it on
        # every re-sync, the same way the kernarg order is checked.
        #
        # Filed under 'block_kv', not 'block_n': there is no upstream field
        # this key belongs to on this arch (we compute it, we don't rename
        # someone else's), and gfx950's BwdDkDvKnobs already owns a real
        # 'block_kv' field for the same concept (see the gfx950 branch below)
        # -- so grid_calculator() reads one key regardless of arch.
        team_waves = 2 * knobs.contraction_shards if knobs.split_head_dim else 1
        block_kv = ROWS_PER_WAVE * (knobs.num_waves // team_waves)
        # Past this line `knobs` is the JSON dict, not BwdDkDvKnobs -- which is
        # why `build()` above binds the dataclass as a default argument, and why
        # every field read off it happens before here.
        knobs = asdict(knobs)
        knobs['block_kv'] = block_kv
        # gfx1201's own knob class has no GRID_AXIS_ORDER field: unlike
        # the forward and dQ, this kernel walks (tile, head, seq) --
        # TILE_FASTEST -- because it is one workgroup per KV head with the GQA
        # query heads summed inside; see grid_calculator()'s own comment.
        knobs['GRID_AXIS_ORDER'] = 1  # TILE_FASTEST; fmha_tuning_gfx950.GRID_AXIS_TILE_FASTEST
        return build, knobs

    elif arch == 'gfx950':
        from fmha_tuning_bwd_dkdv_gfx950 import BwdDkDvInputMetadata, bwd_dkdv_knobs

        tile = choices.BLOCK_DMODEL
        meta = BwdDkDvInputMetadata(
            num_heads=1,
            head_dim=tile,
            head_dim_v=tile,
            causal=choices.CAUSAL_TYPE != 0,
            window=choices.CAUSAL_TYPE != 0,
            dtype_str='bf16' if '*bf16' in choices.arg('Q') else 'f16',
            bias=bool(choices.BIAS_TYPE),
            dropout=bool(choices.ENABLE_DROPOUT),
            # There is no varlen field to set: the kernel decodes the
            # `varlen_bits` word unconditionally, and a dense call is
            # bits == 0.
            #
            # lse_layout_th stays at its False default, and that default no
            # longer decides anything. It used to: the row read was specialised
            # at build time, so a binary compiled False and handed (T, H) bits
            # read the wrong elements -- which is what test_varlen.py's
            # lse_layout=['HT', 'TH'] axis was hitting, 1193 failures of it.
            # Since the row read branches on the runtime pitch instead, one
            # build serves both layouts and this field survives only in the
            # metadata and the cache key, as a record of which layout a build
            # was tuned against.
            #
            # It is deliberately NOT promoted to a functional axis: compiling
            # both variants would double the dK/dV binary count and buy no
            # correctness, because the binary already handles both.
        )
        knobs = bwd_dkdv_knobs(
            arch,
            block_dmodel=tile,
            block_dmodel_v=tile,
            padded_head=choices.PADDED_HEAD,
            strides_constexpr=False,
            # NOTE: unlike Gfx950Knobs (forward, dQ), BwdDkDvKnobs is a
            # standalone dataclass with no `paged`/`num_kv_splits` fields at
            # all -- there is nothing to pin here. This kernel's real
            # signature also carries no folded constexpr parameter (only the
            # forward does), so the forward's pin has no counterpart here.
        ).resolve(meta)
        assert knobs.block_dmodel == tile, (
            f'resolve() returned block_dmodel={knobs.block_dmodel} for '
            f'BLOCK_DMODEL={tile}; the compiled tile must be the operator axis')

        def build():
            """Deferred: constructs the FlyDSL module. Imports flydsl transitively,
            so ONLY `aotriton.flyc_compile` (run by ninja) may call this."""
            from fmha_bwd_dkdv_gfx950 import build_fmha_bwd_dkdv_gfx950_module_primary
            return build_fmha_bwd_dkdv_gfx950_module_primary(meta, knobs)

        build.flyc_source = 'fmha_bwd_dkdv_gfx950.py'
        build.flyc_kernel_name = 'fmha_bwd_dkdv_gfx950_kernel'

        # block_kv and GRID_AXIS_ORDER are flat resolved fields on BwdDkDvKnobs
        # already -- no mirroring needed, unlike the gfx1201 branch
        # above. The two arches derive this value differently (gfx950 resolves
        # it as a real knob; gfx1201 has no such field and derives it from
        # three others, see the note there), but both land on the same
        # 'block_kv' knob, so grid_calculator() reads one key regardless
        # of arch (see flyc_bwd_dkdv.cc).
        return build, asdict(knobs)

    else:
        assert False, f'flyc_bwd_dkdv: no builder branch for arch {arch!r}'
