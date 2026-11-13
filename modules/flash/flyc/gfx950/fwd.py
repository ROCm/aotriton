# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Flash Attention for gfx950 -- AOTriton's `attn_fwd` surface on the dualwave schedule.

The gfx950 counterpart of `gfx1201/fwd.py`. Same goal --
functional equivalence with AOTriton's Triton `attn_fwd`, so this can replace
it rather than sit beside it -- reached from the opposite direction: gfx1201
had the features and needed the speed, gfx950 has the speed and needs the
features.

**The schedule is not ported from gfx1201, and must not be.** aiw's shape is a
consequence of RDNA constraints that do not exist here: gfx11/gfx12 WMMA cannot
co-execute with VALU, so every feature's VALU lands directly on the critical
path, and RDNA has no global->LDS path at all. Neither holds on gfx950, where
MFMA/VALU co-execution is the whole premise of the dual-wave software pipeline
in `kernels/attention/flash_attn_gfx950.py`. That pipeline is what produces
~1150 TFLOPS, and it is what this file keeps. Features port across
architectures; schedules do not.

--- What this file is ------------------------------------------------------

The body below is the production dualwave pipeline, copied rather than
imported. The *helpers* subclass cleanly (see `dualwave_helpers.py`), but
the pipeline itself is inline in `build_flash_attn_dualwave_swp_module`, and
every remaining phase -- windows, bias, dropout -- rewrites parts of it. A copy
that diverges on purpose is honest; an import that has to be special-cased at
six points is not.

`kernels/attention/flash_attn_gfx950.py` therefore stays the production kernel
and the performance oracle. Any build of this file with the feature knobs off
should match it numerically, and the two can be diffed at the ISA level.

--- Argument order is the ABI ----------------------------------------------

The launch order below is the contract, not a convenience: the eventual caller
dispatches the compiled hsaco directly.

Strides arrive in **BHSD slot order** -- batch, head, sequence -- for all four
tensors. Axis 3 is `D`, contiguous by contract, so it is never passed. Only the
*shape* is fixed; any memory layout with D innermost is accepted, which is why
the strides are read rather than derived.

**Named for the axis, not the slot**, matching the gfx1201 kernel. These were
`stride_q0/q1/q2` on the reasoning in `sdpa-feature-gap.md`, which warns
against the `z/h/m/k` suffixes inherited from the maths because they caused
real mix-ups during AOTriton's development. That objection was to *cryptic*
letters, and numeric slots trade one unreadable convention for another -- which
gets actively dangerous here, where BHSD-shaped views of BSHD memory are the
common case and `stride_q2` reads as "Q's third something".

Nothing at runtime distinguishes a head stride from a sequence stride, so a
caller that swaps them gets finite garbage rather than an error. Spelling the
axis out is the only check there is, and the split-K combine bug -- heads
aliasing tokens because a shared helper took `stride_q_n` to be the BSHD token
pitch -- is what its absence costs.

--- head_dim: 32xD tiles, 8xD inputs ----------------------------------------

The compiled tiles are multiples of 32 (`LADDER`), because the PV MFMA writes
32 D columns and the LDS staging granule cannot go below that. A head_dim off
that grid is served by the next tile up, with the real extent passed at runtime
and the surplus columns masked.

**The input contract is 8xD, not 32xD.** Loads and stores are 8 columns wide,
so a head_dim that is a multiple of 8 is a whole number of chunks: the kernel
never touches a column it was not given, whatever the layout, and a plainly
contiguous `(B, H, S, 24)` needs no padding of any kind. Every multiple of 8
from 8 to 512 is covered by `test_grid8_contiguous_is_exact_and_writes_nothing_past_o`.

An odd head_dim -- 100, or a prime -- still works, but only in an allocation
with slack: the kernel rounds each row up to `ceil8(head_dim)`, so those extra
columns must belong to the caller. `_args` checks this and refuses otherwise
rather than corrupting the neighbouring row.

**The contract grants the D axis and nothing else.** Stated as a promise the
caller makes, it is: `tensor[b, h, s, :]` is `ceil8(head_dim)` contiguous
elements, for every `(b, h, s)`, the last one included. It is not a promise
that anything *else* is contiguous, on any axis. A caller may legitimately put
a guard page -- or an unmapped hole, or another tensor -- between
`tensor[b, h, s, :]` and `tensor[b, h, s+1, :]`, between one head and the next,
or between one batch and the next; the strides say where the next row starts,
never that the bytes in between are readable. Only `stride(3) == 1` and the
`ceil8` slack behind it are ours to touch.

That cuts both ways and both ways are load-bearing:

- **Round the D axis up, always.** The `ceil8` columns are granted on *every*
  row, so a buffer descriptor must cover the last row's tail chunk too. One
  that stops at `head_dim` does not merely decline the pad: gfx950 range-checks
  a multi-dword buffer op per dword and drops any dword not wholly inside
  `num_records`, so it takes the real column `head_dim - 1` with it.
  `_slab_span_elems` in `dualwave_helpers.py` is where that is spent, and
  carries the measurement of what the mistake cost.
- **Round no other axis, ever, by any amount.** There is no slack on the
  sequence, head or batch axes to rely on, so a bound may only ever end on an
  element the D grant covers. `_bias_slab_num_records_bytes` holds the same
  line on a bias's KV axis and buys its last column back by narrowing the
  *load* instead (`narrow_tail`), which is the shape any future fix of this
  kind should take.

Note what this does *not* buy, so it is not mistaken for a guarantee: a linear
`num_records` cannot express a per-row extent, so it cannot stop the pad access
on an interior row either way. A caller who puts a guard page immediately after
column `head_dim - 1` is outside the contract and faults regardless of how the
descriptor is bounded -- before this rule at row 0, after it at the last row.
What the bound does protect is the region past the end of the slab, which for
the last `(batch, head)` slab is past the end of the tensor.

--- Phase status -----------------------------------------------------------

P0 (this ABI, runtime scale/head counts/strides, LSE), P1 (runtime
`hdim_qk`/`hdim_vo` with `PADDED_HEAD`), P3 (generalized sliding windows --
`causal_type=3` plus the runtime `window_left`/`window_right` pair) and P4 (the
five `VarlenBits` modes, including the LSE token pitch, row origin and the
`_HT`/`_TH` layout bits) and P5 (a `(B, H, Sq, Sk)` bias, on both kernel
bodies) and P6 (philox dropout, both bodies) are in -- the AOTriton feature
surface is complete. See `sdpa-close-gap-gfx950.md`.

**The dropout mask is a function of element coordinates only**, never of
`BLOCK_M`/`BLOCK_N`. That is the reproducibility contract of
`sdpa-dropout-plan.md` section 3, and from P6 onward it is a constraint on the
tuner rather than a property of today's code: a mask made here is regenerated
by the backward pass and by the debug mask kernel, and all three must agree bit
for bit. `test_dropout_mask_does_not_depend_on_the_tiling` builds one problem
with two supported wave geometries and requires identical output.
"""

import fmha_utils as fmha
from .dualwave_helpers import (
    ParityGemmHelper,
    ParityKernelContext,
    ParityKvGmemToLdsLoader,
    ParityKvLdsToVgprLoader,
    ParityQLoader,
    ParitySoftmaxHelper,
    traits_cache_key,
    ParityStoreHelper,
    wire_ptr,
    wire_view,
)
from fmha_meta import FmhaInputMetadata
from .tuning_fwd import Gfx950Knobs, entry_plan
from .fwd_wide import (
    WideGemmHelper,
    WideKernelContext,
    WideKvLdsToVgprLoader,
    WideSoftmaxHelper,
    WideStoreHelper,
    make_wide_body,
)
from .standalone import dualwave

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.utils.arith import _to_raw as as_mlir_value

KERNEL_NAME = "flash_attn_func_gfx950_kernel"

# Pulled from the production module so the pipeline body below reads exactly as
# it does there. Imported, not re-implemented -- these are the scheduling
# primitives the hand-built pipeline is made of, and a second copy of any of
# them would be a second thing to keep in step.
from .dualwave_helpers import _anchor_v_o  # noqa: E402  (one-accumulator safe)
from .dualwave_helpers import exp2_wait_state  # noqa: E402  (issue 10)

_anchor_v_p = dualwave._anchor_v_p
_dualwave_sync_barrier = dualwave._dualwave_sync_barrier
_s_barrier = dualwave._s_barrier
_s_nop = dualwave._s_nop
_s_setprio = dualwave._s_setprio
_s_waitcnt = dualwave._s_waitcnt
_sched_barrier = dualwave._sched_barrier
_sched_barrier_exp_pairs = dualwave._sched_barrier_exp_pairs
_sched_barrier_pairs = dualwave._sched_barrier_pairs
_stagger_extra_barrier_if_one = dualwave._stagger_extra_barrier_if_one
_v_pair_to_vec32 = dualwave._v_pair_to_vec32
_v_vec32_to_pair = dualwave._v_vec32_to_pair
_waitcnt_vm_n = dualwave._waitcnt_vm_n


def _stagger_extra_barrier_if_zero(stagger_i32):
    """Emit `s_barrier;` only when stagger == 0.

    Local copy of `dualwave._stagger_extra_barrier_if_zero`, differing from it
    in one character class: the constraint string is `"s,~{scc}"`, not `"s"`.

    The asm body runs `s_cmp_eq_u32` and `s_cbranch_scc0`, so it destroys SCC.
    Without the clobber LLVM believes SCC survives the asm and will happily
    keep a compare live across it. It does exactly that here: the epilogue's
    `s_cmp_eq_u64 s[2:3], 0` -- the runtime "is the LSE pointer null?" test in
    `ParityStoreHelper._store_lse_row` -- got scheduled *above* this asm, and
    the `s_cbranch_scc1` that reads it, ~200 instructions later, branched on
    the asm's leftover `stagger == 0` instead. The LSE store was then skipped
    for every row, which is why fp16 + PADDED_HEAD returned an untouched L
    tensor at BLOCK_DMODEL 96/128/160/192 while bf16, whose schedule happened
    to land the two compares in the other order, was fine.

    Nothing about the defect is dtype- or head-dim-specific; those were just
    the schedules that exposed it. Any SCC-carrying compare that LLVM chooses
    to hoist over the asm hits the same wall.

    Delete this and go back to the alias once upstream carries the clobber.
    """
    llvm.inline_asm(
        ir.Type.parse("!llvm.void"),
        [stagger_i32],
        ("s_cmp_eq_u32 $0, 0\n\ts_cbranch_scc0 1f\n\ts_barrier\n\t1:"),
        "s,~{scc}",
        has_side_effects=True,
    )


class _KvTailCausalMaskMixin:
    """Make a causal tile also mask the columns past `seqlen_kv`.

    Upstream deletes the KV tail mask from every causal build. The reason is
    written out on `DualwaveSoftmaxHelper.causal_mask_pair_if_needed`:

        This replaces seq_pad_mask_if_needed: with delta = seqlen_kv -
        seqlen_q the largest key any row may attend to is seqlen_kv - 1, so
        every padding column is already strictly above the diagonal.

    True -- for the `delta` that sentence names. But a window build re-points
    `delta_i32` at the resolved `window_right` (`init_tile_bounds` in
    `dualwave_helpers.py`), and that is what makes the argument fail:
    top-left causal is `window_right == 0`, so the bound is `col <= row`, and
    for `row >= seqlen_kv` that admits columns the K buffer does not have.
    Those columns come back as zero from the out-of-range buffer load, which is
    a logit of 0 rather than -inf, so each one takes a real share of the
    softmax weight and contributes nothing to the numerator. Every row at or
    past `seqlen_kv` comes out too small, by a factor that grows with the
    overhang -- exactly the `seqlen_q > seqlen_k` failures, and only those.

    Both bounds have to move, not just the mask:

    - the mask itself: `seq_pad_mask_if_needed` is the existing one and is
      already a runtime-guarded no-op for a tile that ends at or before
      `seqlen_kv`, so it costs one scalar compare on the tiles that do not
      need it;
    - the *decision* to mask: the inherited test is "does some row's right
      bound land inside this tile", and a tile wholly past the diagonal end
      answers no while still holding padding columns. Running the tail mask
      before that test rather than inside it is what keeps the two
      independent.

    Composed as a mixin over both helpers because the two bodies reach the
    causal mask by different routes -- the dual-wave body through
    `causal_mask_split_prologue_if_needed` and three epilogue calls, the wide
    body (`fwd_wide.make_wide_body`, which this file does not vendor)
    through one call per tile -- and `causal_mask_prologue_if_needed` is the
    single point all of them pass through.

    Delete this and go back to the plain helpers once upstream masks the tail
    in window builds (issue 8).
    """

    def causal_mask_prologue_if_needed(self, v_s, tile_idx=None, kv_end_pos=None, **kwargs):
        if tile_idx is None:
            tile_idx = fx.Index(0)
        v_s = self.seq_pad_mask_if_needed(v_s, tile_idx)
        return super().causal_mask_prologue_if_needed(
            v_s, tile_idx=tile_idx, kv_end_pos=kv_end_pos, **kwargs
        )


class _Exp2WaitStateMixin:
    """`exp2` with the gfx950 `v_exp_f32` wait state; see `exp2_wait_state`.

    The body is `dualwave._exp2_score_slice` verbatim apart from the one call,
    which is why it is copied rather than wrapped: the results are consumed
    inside it (`Vec.from_elements` on the `start == 0` path), so there is no
    return value a wrapper could interpose on.

    The forward is exposed to the hazard the same way dK/dV is, and by the same
    consumer -- `cast_p` reaches `_bf16_trunc_pack_v8`. A scan of the 108
    compiled bf16 forward kernels found a zero-gap `v_exp_f32` site in seven,
    at `BLOCK_DMODEL` 96, 224 and 256, which is exactly the set of rungs whose
    head dims were failing: 96 serves head_dim 72/80/88/96 and 224 serves 216.

    This also supersedes the `_s_nop(1)` in `ParityGemmHelper.qk`, whose own
    comment records that it fixes head_dim 96 "not for the reason it looks
    like" and by perturbing register allocation. That reads as the same defect
    seen from the other end: a barrier that does not bind, working by accident.
    The `_s_nop` is left in place because it was measured to cost nothing and
    removing it is a separate change with its own bisection.
    """

    def exp2(self, v_s, start, length):
        if const_expr(start == 0):
            s_lo = [Vec(v_s[0])[r] for r in range_constexpr(16)]
            lo_partial = exp2_wait_state(
                [dualwave.rocdl.exp2(T.f32, as_mlir_value(s_lo[r])) for r in range_constexpr(16)]
            )
            return Vec.from_elements(lo_partial, fx.Float32).ir_value(), v_s[1]
        lo_partial = [Vec(v_s[0])[r] for r in range_constexpr(16)]
        hi_full = exp2_wait_state(
            [dualwave.rocdl.exp2(T.f32, as_mlir_value(Vec(v_s[1])[r])) for r in range_constexpr(16)]
        )
        return lo_partial, hi_full


class _ParitySoftmaxHelper(_Exp2WaitStateMixin, _KvTailCausalMaskMixin, ParitySoftmaxHelper):
    pass


class _WideSoftmaxHelper(_Exp2WaitStateMixin, _KvTailCausalMaskMixin, WideSoftmaxHelper):
    pass


class _FwdGemmHelper(ParityGemmHelper):
    """`qk` with `qk_scale` applied, which is what the *forward* wants.

    `ParityGemmHelper.qk` returns raw `Q·K^T` because `dq`
    shares it and folds `qk_scale` into its own LSE subtract. The dual-wave
    forward body calls `qk` from six sites across its software pipeline -- the
    prologue, two in the main loop, three in the epilogue -- and every one of
    them wants the scaled score, so overriding once here beats six wrapped call
    sites that a seventh site could silently join without.

    The wide body is not covered by this and does not want to be: it never
    calls `qk`, because its D axis is staged and only the last `qk_stage` has a
    complete sum to scale. `make_wide_body` calls `scale_scores` itself, right
    after that loop.
    """

    def qk(self, v_k, q_all_bf16, stage=0):
        return self.scale_scores(super().qk(v_k, q_all_bf16, stage))


class _FwdStoreHelper(ParityStoreHelper):
    """O staged through the K/V LDS buffers; see `ParityStoreHelper._store_final_o_row`.

    Safe in `_main_body` because every wave's last DS read of those buffers is
    the epilogue's final `load_v`, retired with `lgkmcnt(0)` ahead of the C12
    sync barrier, and no wave reaches `store_final_o` before passing one more
    barrier: the stagger-closing one, which pairs group A with group B's C12
    barrier (or, unstaggered, the plain `s_barrier` there). Every wave has
    therefore finished reading K/V from LDS before any wave writes O into it.
    """

    O_VIA_LDS = True


def _prologue_loads(traits, ctx, kv_gmem_to_lds, kv_lds_to_regs, q_loader, page_ids, stagger_i32, *, q_staged,
                    prefetch_k1=True):
    """The forward prologue up to the first QK: every load issued, K0 and Q in registers, stagger opened.

    Returns `(v_k, q_all_bf16, k1_retire_n)`. The caller must execute
    `_waitcnt_vm_n(k1_retire_n)` before its prologue-end barrier -- the one
    group A's loop cluster 0 (`load_k(1)`) follows.

    **Issue order and the vmcnt contract.** This wave issues, in order:

        K0 (NUM_DMA_K) -> Q (K_STEPS_QK) -> K1 (NUM_DMA_K, if prefetch_k1) -> V0 (NUM_DMA_V)

    and nothing else, so every wait counts from the back. Vector-memory loads
    retire in issue order, so `vmcnt(n)` means "all but the last n are done":

    - `vmcnt(Q + K1 + V0)` before the first barrier: K0 landed, in every wave;
    - `vmcnt(K1 + V0)`: Q landed (staged Q is this wave's own rows, so no
      barrier is needed before reading it from LDS);
    - `vmcnt(V0)` (`k1_retire_n`): K1 landed.

    On return, at most K1 and V0 are outstanding, in that order; for group B
    (and unstaggered builds) only V0. The caller's loop accounting must start
    from that. `prefetch_k1=False` drops K1 from both the issue and the counts
    for a body that never reads tile 1; `k1_retire_n` is then a no-op.

    **Why everything goes out before the first barrier.** A fresh workgroup has
    nothing in LDS to protect, so nothing orders K1/V0 behind K0's barrier,
    and Q is read only by the wave that loaded it. Issued after that barrier
    instead -- as this used to -- Q's latency and K1/V0's issue back-pressure
    sat between the barrier and the first MFMA, on the critical path of every
    workgroup.

    **Why K1 is not retired before the first QK.** The first QK reads K0 and Q
    only. K1 must still be landed by every wave before the barrier preceding
    the *other* group's first `load_k(1)`. Group B's extra (stagger-opening)
    barrier pairs with group A's prologue-end barrier, so B retires K1 before
    it; group A retires it before its prologue-end barrier (`k1_retire_n`),
    which B's own prologue-end barrier pairs with ahead of B's `load_k(1)`. B
    is parked at the extra barrier through A's first QK either way, so waiting
    there costs it nothing.
    """
    n_k1 = ctx.NUM_DMA_K if prefetch_k1 else 0
    after_q = n_k1 + ctx.NUM_DMA_V

    if const_expr(traits.PAGED):
        pro_pageid_0 = page_ids.async_load_split_page(0)
        kv_gmem_to_lds.load_k_split(0, 0, page_id=pro_pageid_0)
    else:
        kv_gmem_to_lds.load_k_split(0, 0)
    _sched_barrier(0)

    # Q is **not scaled** -- `qk_scale` is applied to the f32 scores by
    # `_FwdGemmHelper.qk`; see `ParityQLoader.scale_all`. Staged, this only
    # issues the DMAs; the packs are read out of LDS once they have landed.
    if const_expr(q_staged):
        q_loader.issue_to_lds()
        q_all_bf16 = None
    else:
        q_all_bf16 = q_loader.load_all()
    _sched_barrier(0)

    if const_expr(prefetch_k1):
        if const_expr(traits.PAGED):
            pro_pageid_1 = page_ids.async_load_split_page(1)
            kv_gmem_to_lds.load_k_split(1, 1, page_id=pro_pageid_1)
        else:
            kv_gmem_to_lds.load_k_split(1, 1)
    if const_expr(traits.PAGED):
        pro_v0_pid = page_ids.async_load_split_page(0)
        kv_gmem_to_lds.load_v_split(0, 0, page_id=pro_v0_pid)
    else:
        kv_gmem_to_lds.load_v_split(0, 0)
    _sched_barrier(0)
    _waitcnt_vm_n(traits.K_STEPS_QK + after_q)
    _sched_barrier(0)
    _s_barrier()

    v_k = kv_lds_to_regs.load_k(0)
    if const_expr(q_staged):
        _sched_barrier(0)
        _waitcnt_vm_n(after_q)
        q_all_bf16 = q_loader.load_all()
    _sched_barrier(0)
    _s_waitcnt(traits.LGKMCNT_0_ONLY)

    # OPEN the wave-group phase shift: one extra s_barrier on group B.
    if const_expr(traits.DUALWAVE_SWP_ENABLE_STAGGER):
        _waitcnt_vm_n(after_q)

        @flyc.jit
        def _retire_k1_and_extra_barrier_if_one():
            if fx.Int32(stagger_i32) != fx.Int32(0):
                _waitcnt_vm_n(ctx.NUM_DMA_V)
                _sched_barrier(0)
                _s_barrier()

        _retire_k1_and_extra_barrier_if_one()
    else:
        _waitcnt_vm_n(ctx.NUM_DMA_V)
        _sched_barrier(0)
        _s_barrier()
    return v_k, q_all_bf16, ctx.NUM_DMA_V


_COMPILED = {}


def build_flash_attn_func_gfx950_module_primary(meta, knobs):
    """Build the gfx950 parity kernel for a resolved (meta, knobs) pair.

    Takes the two objects rather than a long parameter list, split on *who
    decides*: `meta` is what the caller asked for, `knobs` is what the tuning
    policy answered. `knobs` must come from `Gfx950Knobs.resolve(meta)` --
    nothing here falls back to a policy, and `knobs.build_traits(meta)` is the arch
    configuration that call produced.
    """
    if meta.block_dmodel is None or knobs.hdim_qk_floor is None:
        raise ValueError("meta and knobs must come from `tuning_fwd.plan` or `resolve_knobs`")
    traits = knobs.build_traits(meta)

    BLOCK_DMODEL = meta.block_dmodel
    PADDED_HEAD = meta.padded_head
    # D columns at or below this are guaranteed real, so the K mask can skip
    # them. See `rung_below` and `ParityKvLdsToVgprLoader.load_k`.
    HDIM_QK_FLOOR = knobs.hdim_qk_floor
    STRIDES_CONSTEXPR = knobs.strides_constexpr

    # Which algorithm this build is. `D_STAGES > 1` is the discriminator rather
    # than a width threshold: staging is what the dual-wave schedule cannot
    # express, so anything staged is on the wide path by construction.
    WIDE = traits.D_STAGES > 1 or traits.VO_SHARDS > 1

    # Whether *every* tile needs the mask applied, or only the ones near the
    # causal diagonal. Plain causal is the latter: `max_num_tiles` already
    # stops the walk just past the diagonal, so the interior tiles below it are
    # fully live and the pipeline masks only `v_s_0`. A window is the former --
    # its left bound clips tiles anywhere in the range -- and so is
    # `CROSS_SEQLEN`, which is why that flag already gates the `v_s_1` site.
    MASK_ALL_TILES = traits.CROSS_SEQLEN or traits.WINDOW

    # Whether the KV walk ends at the last live tile rather than at the base
    # class's bound, which is rounded up to an even count of at least four --
    # what the dual-wave pipeline's prologue, two-tile loop and three-tile
    # epilogue used to require. The epilogue now drains any segment length
    # (see `_drain_three` and the shorter drains after it), so a block whose
    # KV range is one to three tiles, or any odd count, no longer computes
    # masked padding tiles. Split-K and paged keep the padded bound; see
    # `ParityKernelContext._drop_padding_tiles`. The wide body walks one tile
    # at a time and is left as it was.
    #
    # Opt-in, by the `exact_kv_tiles` knob (`traits.EXACT_KV_TILES`), which
    # `make_traits` refuses for the bodies above. It pays where the padding is
    # a large share of the walk (+69% at seqlen_k 64, +33% at 128) and costs
    # 1.5-2% at four tiles or more, where the four whole-body arms only add
    # code; off, the kernel is the single padded body it always was.
    EXACT_TILE_COUNT = traits.EXACT_KV_TILES

    # A scale baked into the build configuration. `None` means "derive it",
    # which `fmha.resolve_scale` then does from the *real* head dim rather than
    # the compiled tile -- the distinction that matters under a padded head.
    # Precedence is per-call `scale` > `meta.sm_scale` > derived.
    BUILD_SM_SCALE = meta.sm_scale

    # Everything the build depends on, so that two builds differing anywhere
    # cannot collide in the JIT disk cache. `traits_cache_key` is all 93 trait
    # fields rather than `traits.cache_tag`'s subset -- see there for what the
    # subset was silently returning the wrong binary for. The rest are builder
    # values with no trait of their own.
    _cache_tag = (
        traits_cache_key(traits),
        BLOCK_DMODEL,
        PADDED_HEAD,
        HDIM_QK_FLOOR,
        STRIDES_CONSTEXPR,
        BUILD_SM_SCALE,
    )

    _lds_elem_dtype = dualwave.dtype_to_elem_type(traits.DTYPE_STR)

    # Q staged through LDS so that its global loads are whole rows; see
    # `ParityQLoader.issue_to_lds`. A region of its own rather than a reused
    # one: every K/V buffer but V1 is being filled while Q is in flight. The
    # 8-wave families cannot hold a second workgroup per CU on VGPRs alone
    # (248 of 256 at head_dim 128), so the extra LDS costs no occupancy there;
    # the 4-wave ones could, so they keep the register load.
    _q_stage = None if WIDE or traits.NUM_WAVES != 8 else ParityQLoader.q_stage_layout(traits)
    if _q_stage is not None:
        _bt_bytes = traits.PAGED_BT_LDS_SIZE * 4 if traits.PAGED else 0
        if traits.LDS_KV_TOTAL_SIZE * traits.BF16_BYTES + _bt_bytes + _q_stage[1] > Gfx950Knobs.LDS_CAP_BYTES:
            _q_stage = None
    _Q_STAGE_ELEMS = _q_stage[1] // traits.BF16_BYTES if _q_stage is not None else 0

    if const_expr(traits.PAGED and _q_stage is not None):

        @fx.struct
        class SharedStorage:
            kv: fx.Array[_lds_elem_dtype, traits.LDS_KV_TOTAL_SIZE, 16]
            bt: fx.Array[fx.Int32, traits.PAGED_BT_LDS_SIZE, 16]
            q: fx.Array[_lds_elem_dtype, _Q_STAGE_ELEMS, 16]

    elif const_expr(traits.PAGED):

        @fx.struct
        class SharedStorage:
            kv: fx.Array[_lds_elem_dtype, traits.LDS_KV_TOTAL_SIZE, 16]
            bt: fx.Array[fx.Int32, traits.PAGED_BT_LDS_SIZE, 16]

    elif const_expr(_q_stage is not None):

        @fx.struct
        class SharedStorage:
            kv: fx.Array[_lds_elem_dtype, traits.LDS_KV_TOTAL_SIZE, 16]
            q: fx.Array[_lds_elem_dtype, _Q_STAGE_ELEMS, 16]

    else:

        @fx.struct
        class SharedStorage:
            kv: fx.Array[_lds_elem_dtype, traits.LDS_KV_TOTAL_SIZE, 16]

    # `Workspace`, `BlockTable` and `block_table_stride` are the three kernargs
    # gfx950 carries and gfx1201 does not -- split-K and paged, neither of which
    # exists over there. Every one of their consumers already sits behind a
    # compile-time guard (`const_expr(traits.SPLITK)` in `init_workspace` and
    # `init_workspace_io`, `const_expr(traits.PAGED)` in `GenericPageIdLoader`
    # and `init_descriptors`, `DUALWAVE_SWP_DEBUG_LAZY_COUNTS` for the lazy
    # counts), so in a build with those off nothing reads them.
    #
    # **Annotating them `Constexpr` in that case removes the kernarg entirely**,
    # rather than passing a null pointer in a slot that still exists:
    # `compiler/kernel_function.py` sorts constexpr-annotated parameters into
    # `constexpr_values` and never adds them to `kernel_arg_types`. That matters
    # because the slots sit *between* `LSE` and `seqinfo_q0`, so leaving them
    # occupied would shift every later argument two positions away from
    # gfx1201's layout -- a consumer that hardcodes the kernarg block reads the
    # wrong slots no matter what the pointers contain.
    #
    # With them gone, and the six tensor operands now pointers, the forward's
    # kernarg is 296 bytes: 292 of declared fields plus one 4-byte hole, where
    # `sm_scale` (f32, offset 172) meets the fifteen i64 strides that need
    # 8-byte alignment. No tensor descriptor remains: 536 -> 296 at a fixed
    # configuration, and **240 is exactly six 40-byte descriptors**, which is
    # what says the conversion recovered all of them and nothing else moved.
    #
    # It is **not** byte-identical to `flash_attn_func_aiw_kernel` and this
    # comment used to claim it was. That kernel models to 300 bytes and its
    # field order diverges from slot 3 on: it carries `batch_size` on the wire,
    # which this one recovers from the grid, and it puts `sm_scale` last rather
    # than before the strides. Matching them is an ABI decision with a caller on
    # the other end, not a cleanup; the sizes are recorded here so whoever makes
    # it starts from measurements.
    #
    # The annotation can be a per-build choice because this `def` executes on
    # every build and the module has no `from __future__ import annotations`,
    # so the expression is evaluated here rather than kept as a string.
    #
    # The stand-in value is `0` and not `None` because a constexpr value becomes
    # part of the JIT cache key through `Constexpr.value_signature`, which
    # accepts int/bool/float/str/tuple/lambda and raises on anything else. It is
    # never read; the guards above see to that.
    _WS_RUNTIME = bool(traits.SPLITK or traits.DUALWAVE_SWP_DEBUG_LAZY_COUNTS)
    _BT_RUNTIME = bool(traits.PAGED)
    _WS_ANN = fx.Tensor if _WS_RUNTIME else fx.Constexpr
    _BT_ANN = fx.Tensor if _BT_RUNTIME else fx.Constexpr
    _BTS_ANN = fx.Int32 if _BT_RUNTIME else fx.Constexpr

    @flyc.kernel(known_block_size=[traits.BLOCK_SIZE, 1, 1])
    def flash_attn_func_gfx950_kernel(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        B: fx.Pointer,
        O: fx.Pointer,  # noqa: E741
        LSE: fx.Pointer,
        Workspace: _WS_ANN,
        BlockTable: _BT_ANN,
        seqinfo_q0: fx.Pointer,
        seqinfo_q1: fx.Pointer,
        seqinfo_k0: fx.Pointer,
        seqinfo_k1: fx.Pointer,
        varlen_bits: fx.Int32,
        num_seqlens: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        philox_seed_ptr: fx.Pointer,
        philox_offset1: fx.Pointer,
        philox_offset2: fx.Int64,
        philox_seed_output: fx.Pointer,
        philox_offset_output: fx.Pointer,
        idropout_p: fx.Int32,
        dropout_scale: fx.Float32,
        num_head_q: fx.Int32,
        num_head_k: fx.Int32,
        hdim_qk: fx.Int32,
        hdim_vo: fx.Int32,
        sm_scale: fx.Float32,
        stride_q_batch: fx.Int64,
        stride_q_head: fx.Int64,
        stride_q_seq: fx.Int64,
        stride_k_batch: fx.Int64,
        stride_k_head: fx.Int64,
        stride_k_seq: fx.Int64,
        stride_v_batch: fx.Int64,
        stride_v_head: fx.Int64,
        stride_v_seq: fx.Int64,
        stride_o_batch: fx.Int64,
        stride_o_head: fx.Int64,
        stride_o_seq: fx.Int64,
        stride_b_batch: fx.Int64,
        stride_b_head: fx.Int64,
        stride_b_seq_q: fx.Int64,
        block_table_stride: _BTS_ANN,
    ):
        # The six tensor operands arrive as bare pointers -- see `wire_ptr` in
        # `dualwave_helpers.py` for why, and `wire_view` for why they are
        # wrapped back up here rather than at each use. Doing it once, at the
        # boundary, is what leaves the addressing code below untouched: every
        # extent it needs is a wire scalar already, so the only thing the
        # production helpers ever wanted from a tensor was `fx.get_iter`, and
        # that is exactly what survives the round trip.
        Q = wire_view(Q)
        K = wire_view(K)
        V = wire_view(V)
        B = wire_view(B)
        O = wire_view(O)  # noqa: E741
        LSE = wire_view(LSE)
        ctx = (WideKernelContext if WIDE else ParityKernelContext)(
            traits,
            strides=(
                stride_q_batch,
                stride_q_head,
                stride_q_seq,
                stride_k_batch,
                stride_k_head,
                stride_k_seq,
                stride_v_batch,
                stride_v_head,
                stride_v_seq,
            ),
            o_strides=(stride_o_batch, stride_o_head, stride_o_seq),
            sm_scale=sm_scale,
            num_head_q=num_head_q,
            num_head_k=num_head_k,
            hdim_qk=hdim_qk,
            hdim_vo=hdim_vo,
            padded_head=PADDED_HEAD,
            hdim_qk_floor=HDIM_QK_FLOOR,
            window_left=window_left,
            window_right=window_right,
            seqinfo=(seqinfo_q0, seqinfo_q1, seqinfo_k0, seqinfo_k1),
            varlen_bits=varlen_bits,
            num_seqlens=num_seqlens,
            Q=Q,
            K=K,
            V=V,
            O=O,
            DebugCounts=Workspace,
            BlockTable=BlockTable,
            Bias=B,
            bias_strides=(stride_b_batch, stride_b_head, stride_b_seq_q),
            philox=(philox_seed_ptr, philox_offset1, philox_offset2, philox_seed_output, philox_offset_output),
            idropout_p=idropout_p,
            dropout_scale=dropout_scale,
            seq_len=max_seqlen_q,
            seq_len_kv=max_seqlen_k,
            stride_q_n=stride_q_seq,
            stride_kv_n=stride_k_seq,
            head_dim_runtime=hdim_qk,
            block_table_stride=block_table_stride,
            LSE=LSE,
        )
        ctx.init_types_and_constants()
        ctx.init_runtime_indices()
        ctx.init_lds(SharedStorage)
        ctx.init_thread_mapping()
        ctx.init_sequence_lengths()
        ctx.init_descriptors()
        ctx.init_workspace()
        ctx.init_philox()
        ctx.init_atoms_and_lds_ptrs()
        ctx.init_dma_thread_offsets()
        ctx.init_tile_bounds(exact_tile_count=EXACT_TILE_COUNT)
        ctx.init_active_guard()
        ctx.init_lds_read_bases()
        ctx.init_dma_m0_tables()
        if const_expr(_q_stage is not None):
            ctx.lds_q_stage_idx = fx.Index(fx.ptrtoint(ctx.lds.q.ptr))

        active = ctx.active
        elem_dtype = ctx.elem_dtype
        stagger_i32 = ctx.stagger_i32
        l_row_init = ctx.c_zero_f
        split_t_end = ctx.split_t_end
        v_o_zero = ctx.c_zero_v16f32

        kv_gmem_to_lds = ParityKvGmemToLdsLoader(ctx)
        kv_lds_to_regs = (WideKvLdsToVgprLoader if WIDE else ParityKvLdsToVgprLoader)(ctx)
        output_store = (WideStoreHelper if WIDE else _FwdStoreHelper)(ctx)
        page_ids = dualwave.DualwavePageIdLoader(ctx)
        q_loader = ParityQLoader(ctx)
        gemm_helper = (WideGemmHelper if WIDE else _FwdGemmHelper)(ctx)
        softmax_helper = (_WideSoftmaxHelper if WIDE else _ParitySoftmaxHelper)(ctx)

        def _main_body(drain):
            """Prologue, loop and the epilogue `drain`; see `_exact_body`.

            `drain` is `_drain_three` or `_drain_two` for a segment of three
            tiles or more, `_drain_one` or `_drain_none` for two tiles or one.
            The last two have no loop iteration, and prefetch no K beyond what
            their segment holds.
            """
            pipelined = drain is _drain_three or drain is _drain_two
            # Paged: stage the block-table row into LDS before any page-id ds_read.
            if const_expr(traits.PAGED):
                page_ids.load_block_table_to_lds()
                _s_waitcnt(0)
                _sched_barrier(0)
                _s_barrier()

            v_k, q_all_bf16, k1_retire_n = _prologue_loads(traits,
                                                           ctx,
                                                           kv_gmem_to_lds,
                                                           kv_lds_to_regs,
                                                           q_loader,
                                                           page_ids,
                                                           stagger_i32,
                                                           q_staged=_q_stage is not None,
                                                           prefetch_k1=drain is not _drain_none)

            # Prologue scores + first softmax pass for KV tile 0
            if const_expr(traits.PAGED):
                pro_pageid_2_lds = page_ids.load_page_id_lds(page_ids.split_tile(2))
            v_s_0 = gemm_helper.qk(v_k, q_all_bf16)
            _sched_barrier(0)

            # `split_tile(0)`, not tile 0. The prologue loaded `load_k_split(0)`,
            # so the tile it must mask is the *base* of this workgroup's range,
            # and split-K was long the only thing that moved that base -- which
            # is why the non-split arms used to spell it `0`. A window moves it
            # too, and masking tile 0's diagonal against another tile's scores
            # is wrong in a way only a left bound exposes: the right bound is
            # so slack at tile 0 that it keeps everything either way.
            #
            # Identical to the old code wherever the base is zero, so builds
            # without split-K or a window are unchanged bit for bit.
            if const_expr(traits.CAUSAL):
                v_s_0 = softmax_helper.causal_mask_split_prologue_if_needed(v_s_0)
            else:
                v_s_0 = softmax_helper.seq_pad_mask_if_needed(v_s_0, softmax_helper.split_tile(0))
            m_row_pro = softmax_helper.reduce_max(v_s_0)
            if const_expr(traits.CAUSAL):
                m_row_pro = softmax_helper.floor_masked_max(m_row_pro)
            v_s_0 = softmax_helper.sub_m(v_s_0, m_row_pro)
            v_p_0 = softmax_helper.exp2(v_s_0, 0, 16)
            pro_pageid_2 = page_ids.finish_page_id(pro_pageid_2_lds) if const_expr(traits.PAGED) else fx.Index(0)
            # Group A's K1 retirement, deferred from before its first QK; see
            # `_prologue_loads`. A no-op for group B and for unstaggered
            # builds, which retired it already.
            _waitcnt_vm_n(k1_retire_n)
            _dualwave_sync_barrier()

            # Software-pipelined inner loop
            # `split_tile(3)`, for the same reason the prologue's mask uses
            # `split_tile(0)`: the loop resumes three tiles past this
            # workgroup's *base*, and split-K used to be the only thing that
            # moved it. An absolute 3 makes a window build re-walk the tiles it
            # deliberately skipped. Identical wherever the base is zero.
            loop_lb = ctx.split_tile(3)

            if const_expr(traits.PAGED):
                _init_v_pid_lds = page_ids.load_page_id_lds(loop_lb - fx.Index(2))
                kv_gmem_to_lds.load_k_split(2, 0, page_id=pro_pageid_2)
            elif const_expr(pipelined):
                kv_gmem_to_lds.load_k_split(2, 0)

            # ============================= Main loop =============================
            init_args = [m_row_pro, l_row_init]
            for _ in range_constexpr(traits.D_CHUNKS):
                init_args.append(v_o_zero)
            init_args.append(_v_pair_to_vec32(v_p_0))
            if const_expr(traits.PAGED):
                init_args.append(page_ids.finish_page_id(_init_v_pid_lds))
            loop_results = init_args
            v_pid_arg_idx = 3 + traits.D_CHUNKS
            # The one- and two-tile arms give the loop an empty range, which
            # folds away at compile time instead of being tested at run time.
            loop_ub = split_t_end - fx.Index(1) if const_expr(pipelined) else loop_lb
            for j, loop_args in range(
                loop_lb,
                loop_ub,
                fx.Index(2),
                init=init_args,
            ):
                m_row = loop_args[0]
                l_row = loop_args[1]
                v_o = [loop_args[2 + i] for i in range_constexpr(traits.D_CHUNKS)]
                v_p_0 = _v_vec32_to_pair(loop_args[2 + traits.D_CHUNKS])
                if const_expr(traits.PAGED):
                    cur_pageid = loop_args[v_pid_arg_idx]
                j_idx = j

                # Cluster 0
                _s_nop(7)
                _sched_barrier(0)
                if const_expr(traits.PAGED):
                    kv_gmem_to_lds.load_v_tile(j_idx - 2, 1, page_id=cur_pageid)
                else:
                    kv_gmem_to_lds.load_v_tile(j_idx - 2, 1)
                v_k = kv_lds_to_regs.load_k(1)
                _s_waitcnt(traits.LGKMCNT_0_ONLY)
                _waitcnt_vm_n(ctx.NUM_DMA_K + ctx.NUM_DMA_V)
                _dualwave_sync_barrier()

                # Cluster 1
                if const_expr(traits.PAGED):
                    c2_pageid_lds = page_ids.load_page_id_lds(j_idx)
                v_s_1 = gemm_helper.qk(v_k, q_all_bf16)
                v_p_0 = softmax_helper.exp2(v_p_0, 16, 16)
                l_row = softmax_helper.reduce_sum(l_row, v_p_0)
                v_p_0 = softmax_helper.cast_p(v_p_0, j_idx - fx.Index(3))
                v_p_0 = _anchor_v_p(traits, v_p_0, elem_dtype=elem_dtype)
                _sched_barrier_exp_pairs(traits, 6, 3, 1)
                _sched_barrier_pairs(traits, 10, 5, 1)
                c2_pageid = page_ids.finish_page_id(c2_pageid_lds) if const_expr(traits.PAGED) else fx.Index(0)
                _dualwave_sync_barrier()

                # Cluster 2
                _s_nop(7)
                _sched_barrier(0)
                if const_expr(traits.PAGED):
                    kv_gmem_to_lds.load_k_tile(j_idx, 1, page_id=c2_pageid)
                else:
                    kv_gmem_to_lds.load_k_tile(j_idx, 1)
                v_v = kv_lds_to_regs.load_v(0)
                _s_waitcnt(traits.LGKMCNT_0_ONLY)
                _waitcnt_vm_n(ctx.NUM_DMA_K + ctx.NUM_DMA_V)
                _dualwave_sync_barrier()

                # Cluster 3
                if const_expr(traits.PAGED):
                    c4_pageid_lds = page_ids.load_page_id_lds(j_idx - 1)
                if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                    _s_setprio(1)
                v_o = gemm_helper.pv_step_k(0, v_p_0, v_v, v_o)
                if const_expr(traits.CAUSAL and MASK_ALL_TILES):
                    v_s_1 = softmax_helper.causal_mask_prologue_if_needed(
                        v_s_1,
                        j_idx - 2,
                        kv_end_tile=j_idx - 1,
                    )
                else:
                    v_s_1 = softmax_helper.bias_to_lists(v_s_1, j_idx - 2)
                m_tile_max_a = softmax_helper.reduce_max(v_s_1)
                _sched_barrier_pairs(traits, 4, 6, 2)
                if const_expr(traits.DUALWAVE_SWP_LAZY_RESCALE):
                    v_o, m_row, l_row, v_p_0 = softmax_helper.lazy_rescale_o(v_o, m_row, l_row, m_tile_max_a, v_p_0)
                else:
                    v_o, m_row, l_row, v_p_0 = softmax_helper.rescale_o(v_o, m_row, l_row, m_tile_max_a, v_p_0)
                v_o = gemm_helper.pv_step_k(1, v_p_0, v_v, v_o)
                v_o = gemm_helper.pv_step_k(2, v_p_0, v_v, v_o)
                v_o = gemm_helper.pv_step_k(3, v_p_0, v_v, v_o)
                v_s_1 = softmax_helper.sub_m(v_s_1, m_row)
                v_p_1 = softmax_helper.exp2(v_s_1, 0, 16)

                _sched_barrier_pairs(traits, 6, 6, 2)
                _sched_barrier_exp_pairs(traits, 6, 3, 2)
                if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                    _s_setprio(0)
                c4_pageid = page_ids.finish_page_id(c4_pageid_lds) if const_expr(traits.PAGED) else fx.Index(0)
                _dualwave_sync_barrier()

                # Cluster 4
                _s_nop(7)
                _sched_barrier(0)
                if const_expr(traits.PAGED):
                    kv_gmem_to_lds.load_v_tile(j_idx - 1, 0, page_id=c4_pageid)
                else:
                    kv_gmem_to_lds.load_v_tile(j_idx - 1, 0)
                v_k = kv_lds_to_regs.load_k(0)
                _s_waitcnt(traits.LGKMCNT_0_ONLY)
                _waitcnt_vm_n(ctx.NUM_DMA_K + ctx.NUM_DMA_V)
                _dualwave_sync_barrier()

                # Cluster 5
                if const_expr(traits.PAGED):
                    _c6_kpid_lds = page_ids.load_page_id_lds(j_idx + 1)
                v_s_0 = gemm_helper.qk(v_k, q_all_bf16)
                v_p_1 = softmax_helper.exp2(v_p_1, 16, 16)
                l_row = softmax_helper.reduce_sum(l_row, v_p_1)
                v_p_1 = softmax_helper.cast_p(v_p_1, j_idx - fx.Index(2))
                v_p_1 = _anchor_v_p(traits, v_p_1, elem_dtype=elem_dtype)
                _sched_barrier_exp_pairs(traits, 6, 3, 3)
                _sched_barrier_pairs(traits, 10, 5, 3)
                _c6_kpid = page_ids.finish_page_id(_c6_kpid_lds) if const_expr(traits.PAGED) else fx.Index(0)
                _dualwave_sync_barrier()

                # Cluster 6
                _s_nop(7)
                _sched_barrier(0)
                if const_expr(traits.PAGED):
                    kv_gmem_to_lds.load_k_tile(j_idx + 1, 0, page_id=_c6_kpid)
                else:
                    kv_gmem_to_lds.load_k_tile(j_idx + 1, 0)
                v_v = kv_lds_to_regs.load_v(1)
                if const_expr(traits.CAUSAL):
                    v_s_0 = softmax_helper.causal_mask_prologue_if_needed(
                        v_s_0,
                        j_idx - 1,
                        kv_end_tile=j_idx,
                    )
                else:
                    v_s_0 = softmax_helper.bias_to_lists(v_s_0, j_idx - 1)
                _s_waitcnt(traits.LGKMCNT_0_ONLY)
                _waitcnt_vm_n(ctx.NUM_DMA_K + ctx.NUM_DMA_V)
                _dualwave_sync_barrier()

                # Cluster 7
                if const_expr(traits.PAGED):
                    next_pageid_lds = page_ids.load_page_id_lds(j_idx)
                if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                    _s_setprio(1)
                v_o = gemm_helper.pv_step_k(0, v_p_1, v_v, v_o)
                m_tile_max_b = softmax_helper.reduce_max(v_s_0)
                _sched_barrier_pairs(traits, 4, 6, 4)
                if const_expr(traits.DUALWAVE_SWP_LAZY_RESCALE):
                    v_o, m_row, l_row, v_p_1 = softmax_helper.lazy_rescale_o(v_o, m_row, l_row, m_tile_max_b, v_p_1)
                else:
                    v_o, m_row, l_row, v_p_1 = softmax_helper.rescale_o(v_o, m_row, l_row, m_tile_max_b, v_p_1)
                v_o = gemm_helper.pv_step_k(1, v_p_1, v_v, v_o)
                v_o = gemm_helper.pv_step_k(2, v_p_1, v_v, v_o)
                v_o = gemm_helper.pv_step_k(3, v_p_1, v_v, v_o)
                v_s_0 = softmax_helper.sub_m(v_s_0, m_row)
                v_p_0 = softmax_helper.exp2(v_s_0, 0, 16)
                _sched_barrier_pairs(traits, 6, 5, 4)
                _sched_barrier_exp_pairs(traits, 6, 3, 4)
                if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                    _s_setprio(0)
                if const_expr(traits.PAGED):
                    next_pageid = page_ids.finish_page_id(next_pageid_lds)
                _dualwave_sync_barrier()

                yield_args = [m_row, l_row] + v_o + [_v_pair_to_vec32(v_p_0)]
                if const_expr(traits.PAGED):
                    yield_args.append(next_pageid)
                loop_results = yield yield_args

            # Epilogue drains the final in-flight tiles without further prefetch-ahead.
            m_row = loop_results[0]
            l_row = loop_results[1]
            v_o = [loop_results[2 + i] for i in range_constexpr(traits.D_CHUNKS)]
            v_p_0 = _v_vec32_to_pair(loop_results[2 + traits.D_CHUNKS])
            ec0_v_pid = loop_results[v_pid_arg_idx] if const_expr(traits.PAGED) else None
            drain_state = (m_row, l_row, v_o, v_p_0, None, ec0_v_pid)

            _store_o(*drain(drain_state, q_all_bf16))

        def _exact_body():
            """Pick the arm for this workgroup's segment length.

            A runtime choice, uniform across the workgroup: the segment length
            depends only on the block's position and the sequence lengths.

            The branch sits *above* the prologue, so each arm is a whole body
            with its own prologue and loop. Branching after the loop instead,
            between drains that share one prologue and loop, is the obvious
            shape and was measured first: any such branch, even to the
            one-tile drain, pushed this 248-VGPR kernel into scratch spills
            (108 to 588 bytes) and cost the long sequences a few percent. With
            the branch at the top no value is live across it but the context,
            and the arms each store their own O, so none has to be merged
            either.
            """
            n_tiles = split_t_end - ctx.split_t0
            if n_tiles >= fx.Index(3):
                if (n_tiles // fx.Index(2)) * fx.Index(2) == n_tiles:
                    _main_body(_drain_three)
                else:
                    _main_body(_drain_two)
            elif n_tiles == fx.Index(2):
                _main_body(_drain_one)
            else:
                _main_body(_drain_none)

        def _store_o(m_row, l_row, v_o):
            l_inv = softmax_helper.safe_l_inv(l_row)
            softmax_helper.scale_o(v_o, l_inv)

            if const_expr(traits.DUALWAVE_SWP_ENABLE_STAGGER):
                _stagger_extra_barrier_if_zero(stagger_i32)
            else:
                _s_barrier()

            if const_expr(not traits.SPLITK):
                output_store.store_final_o(v_o, ctx.q_row, m_row, l_row)
            else:
                output_store.store_splitk_partial_o(v_o, m_row, l_row, ctx.q_row)

        # -- Epilogue drains ------------------------------------------------
        #
        # The pipeline's state at the top of a drain is: tile `t`'s scores are
        # reduced and half exponentiated (`v_p`), its V is in flight to V
        # buffer `par`, and K of tile `t + 1` sits in K buffer `1 - par` (with
        # K of `t + 2`, when that tile exists, in flight to buffer `par`). The
        # prologue leaves exactly that state with `t = split_t0, par = 0`, and
        # so does every loop iteration, two tiles further on.
        #
        # From there `_drain_step` retires `t` and advances to `t + 1` with the
        # parity flipped, `_drain_tail` retires the last two tiles and
        # `_drain_none` the last one. The three-tile drain is the epilogue this
        # kernel always had -- C0..C3, C4..C7, C8..C13 -- and the shorter ones
        # are its suffixes. That is what keeps their LDS hazards the ones the
        # epilogue already gets right: every buffer is read at least two
        # dual-wave barriers after its `vmcnt` wait, and overwritten at least
        # two clusters after its last read, so the lagging wave group of the
        # stagger is never a cluster short.
        #
        # A drain state is `(m_row, l_row, v_o, v_p, l_rescale, v_pid)`:
        # `l_rescale` is the factor the previous step applied to O and not yet
        # to `l_row` (None when nothing is pending), `v_pid` the page of V
        # `t + 1` (paged builds only).

        def _drain_step(st, t, par, q_all_bf16, *, k_prefetch, sched_ids):
            """Retire tile `t`, leave tile `t + 1` half exponentiated. C0..C3.

            K of `t + 2` is in flight on entry, which is what the first `vmcnt`
            wait allows for; `k_prefetch` issues K of `t + 3` as well.
            """
            m_row, l_row, v_o, v_p, l_rescale, v_pid = st
            # Load V(t+1) into the other V buffer; read K(t+1).
            _s_nop(7)
            _sched_barrier(0)
            if const_expr(traits.PAGED):
                kv_gmem_to_lds.load_v_tile(t + 1, 1 - par, page_id=v_pid)
            else:
                kv_gmem_to_lds.load_v_tile(t + 1, 1 - par)
            v_k = kv_lds_to_regs.load_k(1 - par)
            _s_waitcnt(traits.LGKMCNT_0_ONLY)
            _waitcnt_vm_n(ctx.NUM_DMA_K + ctx.NUM_DMA_V)
            _dualwave_sync_barrier()

            # S(t+1); finish P(t).
            if const_expr(traits.PAGED and k_prefetch):
                k_pid_lds = page_ids.load_page_id_lds(t + 3)
            v_s = gemm_helper.qk(v_k, q_all_bf16)
            if const_expr(l_rescale is not None):
                l_row = softmax_helper.apply_l_rescale(l_row, l_rescale)
            v_p = softmax_helper.exp2(v_p, 16, 16)
            l_row = softmax_helper.reduce_sum(l_row, v_p)
            v_p = softmax_helper.cast_p(v_p, t)
            v_p = _anchor_v_p(traits, v_p, elem_dtype=elem_dtype)
            _sched_barrier_exp_pairs(traits, 6, 3, sched_ids[0])
            _sched_barrier_pairs(traits, 10, 5, sched_ids[0])
            if const_expr(k_prefetch):
                k_pid = page_ids.finish_page_id(k_pid_lds) if const_expr(traits.PAGED) else fx.Index(0)
            _dualwave_sync_barrier()

            # Prefetch K(t+3) when that tile exists; read V(t); mask S(t+1).
            if const_expr(k_prefetch):
                _s_nop(7)
                _sched_barrier(0)
                if const_expr(traits.PAGED):
                    kv_gmem_to_lds.load_k_tile(t + 3, 1 - par, page_id=k_pid)
                else:
                    kv_gmem_to_lds.load_k_tile(t + 3, 1 - par)
            v_v = kv_lds_to_regs.load_v(par)
            if const_expr(traits.CAUSAL):
                v_s = softmax_helper.causal_mask_prologue_if_needed(
                    v_s,
                    t + 1,
                    kv_end_tile=t + 2,
                )
            else:
                v_s = softmax_helper.seq_pad_mask_if_needed(v_s, t + 1)
            _s_waitcnt(traits.LGKMCNT_0_ONLY)
            _waitcnt_vm_n(ctx.NUM_DMA_K + ctx.NUM_DMA_V if k_prefetch else ctx.NUM_DMA_V)
            _dualwave_sync_barrier()

            # O += P(t) V(t); start P(t+1).
            if const_expr(traits.PAGED):
                v_pid_lds = page_ids.load_page_id_lds(t + 2)
            if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                _s_setprio(1)
            v_o = gemm_helper.pv(v_p, v_v, v_o)
            m_tile_max = softmax_helper.reduce_max(v_s)
            row_max, rescale = softmax_helper.rescale_from_tile_max(m_row, m_tile_max)
            m_row = row_max
            v_s = softmax_helper.sub_m(v_s, row_max)
            v_p = softmax_helper.exp2(v_s, 0, 16)
            _sched_barrier_pairs(traits, 10, 5, sched_ids[1])
            _sched_barrier_exp_pairs(traits, 6, 3, sched_ids[1])
            _sched_barrier(0)
            softmax_helper.scale_o(v_o, rescale)
            v_o = _anchor_v_o(traits, v_o)
            if const_expr(traits.DUALWAVE_SWP_SETPRIO):
                _s_setprio(0)
            v_pid = page_ids.finish_page_id(v_pid_lds) if const_expr(traits.PAGED) else None
            _dualwave_sync_barrier()
            return (m_row, l_row, v_o, v_p, rescale, v_pid)

        def _drain_tail(st, t, par, q_all_bf16):
            """Retire tiles `t` and `t + 1`, the last two. C8..C13."""
            m_row, l_row, v_o, v_p, l_rescale, v_pid = st
            # Load V(t+1); read K(t+1). No K is in flight any more.
            _s_nop(7)
            _sched_barrier(0)
            if const_expr(traits.PAGED):
                kv_gmem_to_lds.load_v_tile(t + 1, 1 - par, page_id=v_pid)
            else:
                kv_gmem_to_lds.load_v_tile(t + 1, 1 - par)
            v_k = kv_lds_to_regs.load_k(1 - par)
            _s_waitcnt(traits.LGKMCNT_0_ONLY)
            _waitcnt_vm_n(ctx.NUM_DMA_V)
            _dualwave_sync_barrier()

            # S(t+1); finish P(t).
            v_s = gemm_helper.qk(v_k, q_all_bf16)
            if const_expr(l_rescale is not None):
                l_row = softmax_helper.apply_l_rescale(l_row, l_rescale)
            v_p = softmax_helper.exp2(v_p, 16, 16)
            l_row = softmax_helper.reduce_sum(l_row, v_p)
            v_p = softmax_helper.cast_p(v_p, t)
            v_p = _anchor_v_p(traits, v_p, elem_dtype=elem_dtype)
            _sched_barrier_exp_pairs(traits, 6, 3, 9)
            _sched_barrier_pairs(traits, 10, 5, 9)
            _dualwave_sync_barrier()

            # Read V(t); mask S(t+1).
            v_v = kv_lds_to_regs.load_v(par)
            if const_expr(traits.CAUSAL):
                v_s = softmax_helper.causal_mask_prologue_if_needed(
                    v_s,
                    t + 1,
                    kv_end_tile=t + 2,
                )
            else:
                v_s = softmax_helper.seq_pad_mask_if_needed(v_s, t + 1)
            _s_waitcnt(traits.LGKMCNT_0_ONLY)
            _waitcnt_vm_n(0)
            _dualwave_sync_barrier()

            # O += P(t) V(t); all of P(t+1).
            v_o = gemm_helper.pv(v_p, v_v, v_o)
            m_tile_max = softmax_helper.reduce_max(v_s)
            row_max, rescale = softmax_helper.rescale_from_tile_max(m_row, m_tile_max)
            m_row = row_max
            v_s = softmax_helper.sub_m(v_s, row_max)
            v_p = softmax_helper.exp2(v_s, 0, 16)
            _sched_barrier_pairs(traits, 9, 6, 10)
            _sched_barrier_exp_pairs(traits, 7, 3, 10)
            _sched_barrier(0)
            v_p = softmax_helper.exp2(v_p, 16, 16)
            l_row = softmax_helper.apply_l_rescale(l_row, rescale)
            l_row = softmax_helper.reduce_sum(l_row, v_p)
            v_p = softmax_helper.cast_p(v_p, t + 1)
            v_p = _anchor_v_p(traits, v_p, elem_dtype=elem_dtype)
            _sched_barrier(0)
            softmax_helper.scale_o(v_o, rescale)
            v_o = _anchor_v_o(traits, v_o)
            _s_barrier()
            _sched_barrier(0)

            # Read V(t+1).
            v_v = kv_lds_to_regs.load_v(1 - par)
            _s_waitcnt(traits.LGKMCNT_0_ONLY)
            _dualwave_sync_barrier()

            # O += P(t+1) V(t+1).
            v_o = gemm_helper.pv(v_p, v_v, v_o)
            return m_row, l_row, v_o

        def _drain_three(st, q_all_bf16):
            """An even segment of four tiles or more: the original epilogue."""
            t = split_t_end - 4
            st = _drain_step(st, t, 0, q_all_bf16, k_prefetch=True, sched_ids=(5, 6))
            st = _drain_step(st, t + 1, 1, q_all_bf16, k_prefetch=False, sched_ids=(7, 8))
            return _drain_tail(st, t + 2, 0, q_all_bf16)

        def _drain_two(st, q_all_bf16):
            """An odd segment of three tiles or more.

            The loop stopped one tile later than for the even segment one
            longer, so `t` is one tile on and still at parity 0, and K of the
            last tile is the one in flight. This is the three-tile drain from
            its C4, with the buffers swapped.
            """
            t = split_t_end - 3
            st = _drain_step(st, t, 0, q_all_bf16, k_prefetch=False, sched_ids=(7, 8))
            return _drain_tail(st, t + 1, 1, q_all_bf16)

        def _drain_one(st, q_all_bf16):
            """Two tiles: the prologue's and one more. The three-tile drain from its C8."""
            return _drain_tail(st, split_t_end - 2, 0, q_all_bf16)

        def _drain_none(st, _q_all_bf16):
            """One tile, the prologue's: finish P(t), then O += P(t) V(t).

            The barriers are the tail's last two clusters: V(t) is waited on,
            then two barriers pass before any wave reads it, which is what the
            lagging wave group of the stagger needs.
            """
            m_row, l_row, v_o, v_p, _l_rescale, _v_pid = st
            v_p = softmax_helper.exp2(v_p, 16, 16)
            l_row = softmax_helper.reduce_sum(l_row, v_p)
            v_p = softmax_helper.cast_p(v_p, split_t_end - 1)
            v_p = _anchor_v_p(traits, v_p, elem_dtype=elem_dtype)
            _s_waitcnt(traits.LGKMCNT_0_ONLY)
            _waitcnt_vm_n(0)
            _dualwave_sync_barrier()
            _s_barrier()
            _sched_barrier(0)
            v_v = kv_lds_to_regs.load_v(0)
            _s_waitcnt(traits.LGKMCNT_0_ONLY)
            _dualwave_sync_barrier()
            v_o = gemm_helper.pv(v_p, v_v, v_o)
            return m_row, l_row, v_o

        # head_dim > 256 runs a different algorithm, in its own file: no
        # dual-wave pipeline, D staged through LDS and sharded across waves.
        # See `fwd_wide.py` for why none of the schedule above ports.
        if const_expr(WIDE):
            _body = make_wide_body(
                ctx,
                traits,
                q_loader=q_loader,
                kv_gmem_to_lds=kv_gmem_to_lds,
                kv_lds_to_regs=kv_lds_to_regs,
                gemm_helper=gemm_helper,
                softmax_helper=softmax_helper,
                output_store=output_store,
            )
        elif const_expr(EXACT_TILE_COUNT):
            _body = _exact_body
        else:

            def _body():
                _main_body(_drain_three)

        if const_expr(traits.CAUSAL and traits.CROSS_SEQLEN and not traits.SPLITK):
            output_store.zero_o_block_if_needed()

        if active is None:
            _body()
        else:

            @flyc.jit
            def _run_body_if_active():
                if active:
                    _body()

            _run_body_if_active()

        if const_expr(traits.SPLITK):
            output_store.store_empty_split()

    # LOCAL DIVERGENCE FROM UPSTREAM: upstream defines a second `@flyc.kernel`
    # here, `flash_attn_splitk_combine_kernel`, and launches it below. Both are
    # deleted -- AOTriton locates a module's kernel by uniqueness, and this one
    # is dead here because the descriptions pin `num_kv_splits=1`. Restore both
    # blocks together when AOTriton builds a split-K forward.

    def _resolve_window_args(window):
        """`(window_left, window_right)` for the wire, as signed i32.

        Always a pair, even for a build that ignores it -- the non-windowed
        arms forward sentinels so every build shares one ABI and stays directly
        comparable, which is the same reason the strides are passed even under
        `strides_constexpr`. `fmha.window_args` makes the same call.

        A *sentinel* rather than a bound is what goes on the wire for the fixed
        alignments: the kernel resolves it against each sequence's own lengths,
        which is the only correct thing to do once varlen means there is more
        than one pair of lengths to resolve against.
        """
        if not traits.CAUSAL:
            if window is not None:
                # Dropping it silently would return dense attention: right
                # shape, finite, wrong. A window is only ever passed by a
                # caller who believes it is being applied.
                raise ValueError("window= requires a causal build; this one has causal=False")
            return 0, 0
        if not traits.WINDOW:
            if window is not None:
                raise ValueError(
                    "this build is not compiled for windows; pass causal_type=3 in FmhaInputMetadata "
                    "to get generalized sliding-window attention"
                )
            # Bottom-right causal, which is what `delta = seqlen_kv - seqlen_q`
            # already means on this kernel.
            return fmha.WINDOW_BOTRIGHT, fmha.WINDOW_BOTRIGHT
        if window is None:
            raise ValueError(
                "a window build requires window=(left, right); pass "
                "(fmha.WINDOW_TOPLEFT, fmha.WINDOW_TOPLEFT) for top-left causal or "
                "(fmha.WINDOW_BOTRIGHT, fmha.WINDOW_BOTRIGHT) for bottom-right"
            )
        wl, wr = window
        return int(wl), int(wr)

    @flyc.jit
    def launch_flash_attn_func_gfx950(
        Q: fx.Pointer,
        K: fx.Pointer,
        V: fx.Pointer,
        O: fx.Pointer,  # noqa: E741
        LSE: fx.Pointer,
        Bias: fx.Pointer,
        Workspace: _WS_ANN,
        BlockTable: _BT_ANN,
        batch_size: fx.Int32,
        seqinfo_q0: fx.Pointer,
        seqinfo_q1: fx.Pointer,
        seqinfo_k0: fx.Pointer,
        seqinfo_k1: fx.Pointer,
        varlen_bits: fx.Int32,
        num_seqlens: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        philox_seed_ptr: fx.Pointer,
        philox_offset1: fx.Pointer,
        philox_offset2: fx.Int64,
        philox_seed_output: fx.Pointer,
        philox_offset_output: fx.Pointer,
        idropout_p: fx.Int32,
        dropout_scale: fx.Float32,
        num_head_q: fx.Int32,
        num_head_k: fx.Int32,
        hdim_qk: fx.Int32,
        hdim_vo: fx.Int32,
        sm_scale: fx.Float32,
        stride_q_batch: fx.Int64,
        stride_q_head: fx.Int64,
        stride_q_seq: fx.Int64,
        stride_k_batch: fx.Int64,
        stride_k_head: fx.Int64,
        stride_k_seq: fx.Int64,
        stride_v_batch: fx.Int64,
        stride_v_head: fx.Int64,
        stride_v_seq: fx.Int64,
        stride_o_batch: fx.Int64,
        stride_o_head: fx.Int64,
        stride_o_seq: fx.Int64,
        stride_b_batch: fx.Int64,
        stride_b_head: fx.Int64,
        stride_b_seq_q: fx.Int64,
        block_table_stride: _BTS_ANN,
        stream: fx.Stream = fx.Stream(None),
    ):
        # Make the build configuration visible to the JIT cache key.
        _ = _cache_tag
        # The grid's z extent counts *sequences*, which is `num_seqlens` when a
        # packed tensor holds several in one batch slot and `batch_size`
        # otherwise. The two are genuinely different numbers -- a packed
        # (1, H, T, D) call is `batch_size=1, num_seqlens=N` -- and using the
        # batch extent there would launch one program for N sequences.
        bs_idx = fx.Index(num_seqlens if num_seqlens != fx.Int32(0) else batch_size)
        sl_idx = fx.Index(max_seqlen_q)
        num_q_blocks = (sl_idx + traits.BLOCK_M - 1) // traits.BLOCK_M
        if const_expr(traits.SPLITK):
            grid_z = bs_idx * traits.NUM_KV_SPLITS
        else:
            grid_z = bs_idx
        # GQA head packing: one workgroup per `Q_HEADS_PER_WG` q heads. See
        # `ParityKernelContext._pack_q_heads`.
        if const_expr(traits.Q_HEADS_PER_WG > 1):
            grid_x = num_head_q // traits.Q_HEADS_PER_WG
        else:
            grid_x = num_head_q

        # Grid axis order is the production one: head fastest. Not a free
        # choice -- on gfx950 the q_block-fastest arrangement was measured to
        # cost 7% under causal (see `_init_dualwave_thread_mapping`), the
        # opposite sign from gfx1201, because MI355X's 8 XCDs make this an
        # L2-locality lever rather than a duration-spreading one.
        flash_attn_func_gfx950_kernel(
            Q,
            K,
            V,
            Bias,
            O,
            LSE,
            Workspace,
            BlockTable,
            seqinfo_q0,
            seqinfo_q1,
            seqinfo_k0,
            seqinfo_k1,
            varlen_bits,
            num_seqlens,
            max_seqlen_q,
            max_seqlen_k,
            window_left,
            window_right,
            philox_seed_ptr,
            philox_offset1,
            philox_offset2,
            philox_seed_output,
            philox_offset_output,
            idropout_p,
            dropout_scale,
            num_head_q,
            num_head_k,
            hdim_qk,
            hdim_vo,
            sm_scale,
            stride_q_batch,
            stride_q_head,
            stride_q_seq,
            stride_k_batch,
            stride_k_head,
            stride_k_seq,
            stride_v_batch,
            stride_v_head,
            stride_v_seq,
            stride_o_batch,
            stride_o_head,
            stride_o_seq,
            stride_b_batch,
            stride_b_head,
            stride_b_seq_q,
            block_table_stride,
            value_attrs=fmha.kernel_attrs(traits.BLOCK_SIZE, traits.WAVES_PER_EU),
        ).launch(
            grid=(grid_x, num_q_blocks, grid_z),
            block=(traits.BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    # On the JitFunction, not around the call, so AOT compiles get them too.
    # The post-RA scheduler buys 2-8% (hd64 +7.6%, hd128 +4%, hd512 +3-6%,
    # hd256 0); 0.3.1's AOT builds never had it off.
    launch_flash_attn_func_gfx950.compile_hints = fmha.compile_hints(post_misched=True)

    def _args(
        Q,
        K,
        V,
        O,  # noqa: E741
        batch_size,
        seqlen_q,
        seqlen_k=None,
        scale=None,
        lse=None,
        workspace=None,
        block_table=None,
        block_table_stride=None,
        window=None,
        varlen=None,
        num_seqlens=0,
        bias=None,
        dropout_p=None,
        philox_seed=None,
        philox_offset1=None,
        philox_offset2=0,
        philox_seed_out=None,
        philox_offset_out=None,
        stream=None,
    ):
        """Every kernel argument but the stream, in launch order.

        One place that turns tensors into the wire format, so `_launch` and
        `_compile` cannot drift apart -- which is the bug this shape exists to
        prevent, and the reason the gfx1201 kernel grew the same helper.

        The diagonal a *numeric* `window` bound is measured from is VarlenBits
        bit 18 (`principal_diagonal_choice`), and `varlen["bits"]` is its only
        source. A dense call carries it with
        `varlen=fmha.varlen_dense(principal_diagonal=...)`; `varlen=None` is
        word 0, i.e. top-left. The bit means the same thing dense or varlen
        (`d` is merely uniform).
        """
        seqlen_k = seqlen_q if seqlen_k is None else seqlen_k
        ptrs, shape_meta, st = fmha.prep_tensors(
            [("Q", Q), ("K", K), ("V", V), ("O", O)],
            q_heads=("O",),
        )
        # `prep_tensors` is used for the checks and the strides, not its
        # pointers: those are `fx.Uint8`, and a byte pointer carries alignment
        # 1. `wire_ptr` below types each operand from its own tensor instead,
        # which is what keeps the alignment identical to the `fx.Tensor` path
        # this replaced.
        del ptrs
        num_head_q, num_head_k, hdim_qk, hdim_vo = shape_meta
        # GQA head packing serves `Q_HEADS_PER_WG` consecutive q heads of one
        # group per workgroup, all reading the same K/V. A group it does not
        # divide would put two kv heads' q heads in one workgroup, which reads
        # the wrong K/V for one of them -- finite and wrong, so it is refused.
        if traits.Q_HEADS_PER_WG > 1 and (num_head_q // num_head_k) % traits.Q_HEADS_PER_WG:
            raise ValueError(
                f"this build packs {traits.Q_HEADS_PER_WG} q heads per workgroup, which needs a GQA group "
                f"size divisible by it; got {num_head_q} q heads over {num_head_k} kv heads"
            )

        # **The 8-element D pitch is the alignment contract.** Loads and stores
        # are 8 columns wide, so the kernel touches `ceil8(hdim)` columns of
        # every row. An 8-aligned pitch puts that inside the row's own slack;
        # a tightly-packed odd width -- contiguous (B, H, S, 100), pitch 100 --
        # has no slack, and the chunk at column 96 runs into the *next row*.
        #
        # The grant is the D axis and only the D axis: `t[b, h, s, :]` is
        # `ceil8(hdim)` contiguous elements for every `(b, h, s)` including the
        # last, and nothing is assumed contiguous on any other axis -- a guard
        # page between two rows, two heads or two batches is the caller's right.
        # See the module docstring; that is why this checks the strides for
        # *slack* rather than for a packed layout, and why no bound anywhere in
        # these kernels rounds an axis other than D.
        #
        # Every head_dim that is a multiple of 8 satisfies this contiguously,
        # which is what makes 8xD the natural input contract even though the
        # compiled tiles are 32xD. Only the odd widths need a padded view, and
        # only those are checked -- an exact tile width touches nothing past
        # its own columns whatever the pitch.
        #
        # gfx1201 checks the same thing in its interface layer; this kernel is
        # dispatched directly, so the check belongs here. Raising beats the
        # alternative, which is silently corrupting the next row.
        if PADDED_HEAD:
            for _name, _t in (("Q", Q), ("K", K), ("V", V), ("O", O)):
                _d = _t.shape[3]
                _need = (_d + 7) // 8 * 8
                if _need == _d:
                    continue  # 8xD: the row is a whole number of chunks
                # Two separate requirements, and `stride(2) % 8` -- what
                # gfx1201's interface checks -- is only the first of them.
                #
                # *Alignment*: a row starts at `sum(index * stride)`, so every
                # non-D stride must be a multiple of 8 for the 16-byte access
                # to land aligned.
                #
                # *Slack*: the tail chunk needs `ceil8(D) - D` unused elements
                # after the row. The gap to whatever comes next in memory is
                # the smallest non-D stride, which for a BHSD tensor is the D
                # pitch but for a BSHD one is `D` itself -- consecutive heads
                # of the same token are adjacent, so there is no slack at all
                # and the pitch check would wave it through while the store
                # corrupted the next head.
                _outer = [_t.stride(i) for i in range(3) if _t.shape[i] > 1]
                _aligned = _t.stride(3) == 1 and all(s % 8 == 0 for s in _outer)
                _slack = min(_outer, default=_need)
                if not _aligned or _slack < _need:
                    raise ValueError(
                        f"{_name} has shape {tuple(_t.shape)} strides {tuple(_t.stride())}, which "
                        f"cannot hold a head_dim of {_d}. {_d} is not a multiple of 8, so the "
                        f"kernel reads and writes {_need} columns per row and needs the D axis "
                        f"innermost, every other stride a multiple of 8, and {_need - _d} unused "
                        f"element(s) after each row. Allocate the last dimension as {_need} and "
                        f"pass a [..., :{_d}] view, as PyTorch's SDPA shim does -- or use a "
                        f"head_dim that is a multiple of 8, which needs no padding at all."
                    )

        # The kernel skips masking the D columns at or below the floor, so a
        # narrower call would silently reduce over the caller's padding. Cheap
        # host-side check; the alternative is a plausible wrong answer.
        if HDIM_QK_FLOOR and hdim_qk <= HDIM_QK_FLOOR:
            raise ValueError(
                f"this build serves hdim_qk in ({HDIM_QK_FLOOR}, {BLOCK_DMODEL}], got {hdim_qk}; "
                f"build for the narrower head_dim, or pin hdim_qk_floor=0 to mask every column"
            )
        # **Split-K's combine kernel is not stride-general.** It is shared
        # production code and it addresses O the way the dual-wave kernel's
        # BSHD-flattened world does, from just `(batch_size, seq_len,
        # stride_o_seq)`:
        #
        #     o_global = seq * stride_o_seq + head * HEAD_DIM + col
        #
        # and a batch origin of `seq_len * stride_o_seq`. Two things follow,
        # and neither is implied by the other:
        #
        # - heads must be adjacent, `stride_o_head == HEAD_DIM`. On a
        #   BHSD-contiguous O the head term is `head * HEAD_DIM` while the
        #   sequence term is `seq * HEAD_DIM` too, so heads alias tokens.
        # - batches must be packed, `stride_o_batch == seqlen * stride_o_seq`.
        #   A gapped batch stride -- an over-allocated allocation sliced back --
        #   satisfies the first condition and still lands in the wrong batch.
        #
        # Both were measured: error 3.5 and 4.0 against a 2e-2 tolerance,
        # finite, deterministic, no fault. The whole point of this port is that
        # these three strides are free, and the combine path never received
        # that change of variables -- see `dualwave`'s docstring
        # for the change every other helper did get.
        #
        # Strict on purpose. Split-K is off by default and has no caller here,
        # so a rejected layout costs nothing while a silent wrong answer is the
        # failure mode this whole guard exists to prevent.
        if traits.SPLITK:
            _packed_batch = seqlen_q * O.stride(2)
            if O.stride(1) != BLOCK_DMODEL or O.stride(0) != _packed_batch:
                raise ValueError(
                    f"num_kv_splits > 1 needs an O the split-K combine kernel can address: head "
                    f"stride {BLOCK_DMODEL} (heads adjacent) and batch stride {_packed_batch} "
                    f"(batches packed), got strides {tuple(O.stride())} for shape "
                    f"{tuple(O.shape)}. The combine is shared with the production dual-wave kernel "
                    f"and is not stride-general; it would return plausible wrong numbers. Pass a "
                    f"tightly-allocated (B, S, H, D) tensor transposed to (B, H, S, D), or build "
                    f"with num_kv_splits=1."
                )

        if traits.RETURN_LSE and lse is None:
            raise ValueError("this build has return_lse=True and requires an fp32 `lse` tensor")
        if traits.SPLITK and workspace is None:
            raise ValueError("num_kv_splits > 1 requires an fp32 workspace")
        # Placeholders for the tensors a given build does not read. Every one
        # is behind a `const_expr` gate in the kernel, so the slot must be a
        # valid tensor but its contents are never touched.
        # A bias build must be handed one, and a build without bias must not
        # be: silently ignoring a bias tensor returns dense attention that is
        # the right shape and the wrong answer, and it is only ever passed by
        # a caller who believes it is being applied.
        if traits.BIAS_TYPE and bias is None:
            raise ValueError("this build has bias=True and requires a (batch, num_heads, seqlen_q, seqlen_k) tensor")
        if bias is not None and not traits.BIAS_TYPE:
            raise ValueError("this build was not compiled for bias; pass bias=True in FmhaInputMetadata")
        if bias is not None:
            if bias.shape[3] != K.shape[2] or bias.shape[2] != Q.shape[2]:
                raise ValueError(
                    f"bias must be (batch, num_heads, seqlen_q, seqlen_k); got {tuple(bias.shape)} "
                    f"against seqlen_q={Q.shape[2]}, seqlen_k={K.shape[2]}"
                )
            if bias.stride(3) != 1:
                raise ValueError(f"bias needs a contiguous seqlen_k axis; strides are {tuple(bias.stride())}")
        bias_t = bias if bias is not None else O
        bias_st = tuple(int(x) for x in bias.stride()[:3]) if bias is not None else (0, 0, 0)

        # `fmha.dropout_args` is shared with gfx1201: it turns the probability into the
        # i32 threshold the raw random is compared against and the `1/(1-p)`
        # survivor scale, both once per call rather than per element, and
        # keeps the counter as the (pointer, immediate) pair torch splits it
        # into so a captured graph can re-read the pointer half.
        *_dp, _dp_keepalive = fmha.dropout_args(
            bool(traits.ENABLE_DROPOUT),
            dropout_p,
            philox_seed,
            philox_offset1,
            philox_offset2,
            device=Q.device,
            stream=stream,
        )
        if dropout_p is not None and not traits.ENABLE_DROPOUT:
            raise ValueError("this build was not compiled for dropout; pass dropout=True in FmhaInputMetadata")
        _dp_out = (
            fmha.ptr_arg(philox_seed_out) if philox_seed_out is not None else fmha.NULL_PTR,
            fmha.ptr_arg(philox_offset_out) if philox_offset_out is not None else fmha.NULL_PTR,
        )

        lse_out = lse if lse is not None else O
        # When the build folds these away (see `_WS_ANN` at the kernel), the
        # slot still has to be *filled* -- the launcher's parameter is still
        # positional, it is just constexpr, so the value is consumed at trace
        # time instead of becoming a kernarg. `0` rather than the old `O`
        # stand-in, so the JIT cache key does not vary with an unused tensor.
        ws = (workspace if workspace is not None else O) if _WS_RUNTIME else 0
        bt = (block_table if block_table is not None else O) if _BT_RUNTIME else 0

        # `fmha.varlen_args` is shared with gfx1201: it encodes the same
        # wire format, and it is where the two host-side checks live that no
        # kernel can make -- `batch_size` must be the tensor's batch extent
        # whatever the layout, and a packed `num_seqlens` must agree with the
        # length array. Passing the sequence count where the batch extent
        # belongs launches N programs over a 1-batch tensor and every one of
        # them addresses a plausible row.
        _vl = fmha.varlen_args(STRIDES_CONSTEXPR, varlen, seqlen_q, seqlen_k, Q, batch_size, num_seqlens)

        return (
            # Six pointers, not six tensors. `wire_ptr` types each one from the
            # tensor it came from so the alignment the JIT assumes is the one
            # the `fx.Tensor` path assumed; LSE is pinned to f32 rather than
            # read off `lse_out`, because that name falls back to `O` when the
            # build returns no LSE and the slot's type should not depend on it.
            wire_ptr(Q),
            wire_ptr(K),
            wire_ptr(V),
            wire_ptr(O),
            wire_ptr(lse_out, fx.Float32),
            wire_ptr(bias_t),
            ws,
            bt,
            batch_size,
            _vl[1],
            _vl[2],
            _vl[3],
            _vl[4],
            _vl[0],
            int(num_seqlens),
            *_vl[5:],  # max_seqlen_q, max_seqlen_k -- the decode's MAX fallback
            *_resolve_window_args(window),
            _dp[0],
            _dp[1],
            _dp[2],
            _dp_out[0],
            _dp_out[1],
            _dp[3],
            _dp[4],
            num_head_q,
            num_head_k,
            hdim_qk,
            hdim_vo,
            fmha.resolve_scale(
                Q,
                scale if scale is not None else BUILD_SM_SCALE,
                PADDED_HEAD,
                1.0 / (BLOCK_DMODEL**0.5),
            ),
            *st,
            *bias_st,
            0 if block_table_stride is None else block_table_stride,
        ), stream

    def _launch(*args, **kwargs):
        packed, stream = _args(*args, **kwargs)
        return fmha.run_compiled(
            _COMPILED,
            launch_flash_attn_func_gfx950,
            *packed,
            stream if stream is not None else fx.Stream(None),
        )

    def _compile(*args, **kwargs):
        packed, stream = _args(*args, **kwargs)
        return flyc.compile(launch_flash_attn_func_gfx950, *packed, fx.Stream(stream))

    _launch.compile = _compile
    _launch.traits = traits
    _launch.knobs = knobs
    _launch.varlen_dense = fmha.varlen_dense
    return _launch


def build_flash_attn_func_gfx950_module(**kwargs):
    """Keyword front end: name a problem, get the policy's schedule.

    `FmhaInputMetadata`'s fields name the problem; any other keyword pins a
    knob. The real head dims are rounded up the ladder, as the dispatchers'
    `plan` does.
    """
    from dataclasses import fields

    meta_fields = {f.name for f in fields(FmhaInputMetadata)}
    # block_dmodel is required but nullable: None lets `plan` pick the tile.
    meta = FmhaInputMetadata(**{"block_dmodel": None, **{k: v for k, v in kwargs.items() if k in meta_fields}})
    meta, knobs = entry_plan(meta, overrides={k: v for k, v in kwargs.items() if k not in meta_fields})
    return build_flash_attn_func_gfx950_module_primary(meta, knobs)
