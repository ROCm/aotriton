# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI description of the flash bwd dQ/dB FlyDSL backend (gfx1201, gfx950 and gfx942).

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
from ._flyc_common import FLYC_HEAD_DIM_OPTIONS, FlycBwdHints, flyc_bwd_disabled, flyc_input_metadata


@ati.start
@ati.disable(when=flyc_bwd_disabled,
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
# (fmha_utils.bias_args: "B and DB are separate tensors and may be laid
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
# this backend's compiled ladders differ from the operator's BLOCK_DMODEL axis).
# Kept out of the ordered kernarg block above -- BLOCK_DMODEL/PADDED_HEAD are
# not part of `bwd_dq_kernel`'s signature at all, they are build-time
# Python values.
#
# `options=` is this backend's own BLOCK_DMODEL axis, the union of the per-arch
# ladders; the disable predicate narrows it per arch.
@ati.scalar('BLOCK_DMODEL',
            options=FLYC_HEAD_DIM_OPTIONS,
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
    from tuning_dq import resolve_knobs

    meta = flyc_input_metadata(choices)
    knobs = resolve_knobs(arch, meta)

    def build():
        """Deferred: imports flydsl, so only `aotriton.flyc_compile` calls it."""
        from tuning_dq import builder
        return builder(arch, meta, knobs)

    return build, asdict(knobs)
