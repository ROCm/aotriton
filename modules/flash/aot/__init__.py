# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Flash family build block (the ATI registry for the `flash` family).

PASSIVE: this module only DECLARES the family — it imports the kernel/affine
descriptions, defines the two `@ati.metro_kernel` plans inline, and the two
`@ati.operator` defs, then exposes `operators` (the build roots). NO build calls,
NO `kernels`/`affine_kernels` lists, NO registry writes — the code generator's
parser+linker (`aotriton.codegen.parser` / `.linker`) compiles these passive
descriptions into the final IR tree.

Backends are referenced BY the in-file def/object (a metro function carrying
`__ati_metro__`, a kernel def carrying `__ati__`, an affine def carrying
`__ati_affine__`); the linker keys on the declared backend NAME. `default_kdesc`
and `struct_cfields` are NOT declared — the linker derives both from the backend
tree. Family is inferred from the `modules/<family>/aot` path.
"""

import aotriton.template_instantiation as ati

from .attn_fwd import attn_fwd
from .bwd_kernel_dk_dv import bwd_kernel_dk_dv
from .bwd_kernel_dq import bwd_kernel_dq
from .bwd_kernel_fuse import bwd_kernel_fuse
from .bwd_preprocess import bwd_preprocess
from .debug_simulate_encoded_softmax import debug_simulate_encoded_softmax
from .aiter_fwd import aiter_fmha_v3_fwd
from .aiter_bwd import aiter_fmha_v3_bwd
from .flyc_attn_fwd import flyc_attn_fwd
from .flyc_bwd_dkdv import flyc_bwd_dkdv
from .flyc_bwd_dq import flyc_bwd_dq


# --- triton metro backends (transpiled, never executed) -------------------

@ati.start
@ati.metro_kernel
def metro_fwd(params):
    attn_fwd(params)
    if params.encoded_softmax.data_ptr() != 0:
        debug_simulate_encoded_softmax(params)


# The same shape as metro_fwd with the FlyDSL kernel in place of the triton one,
# and the SAME triton debug kernel after it. That mix is the point rather than a
# convenience: one metro launcher, two DSLs, two hsacos, one stream. It is also
# the only thing in the tree that exercises a cross-DSL metro, so it is what
# would catch a launcher that quietly assumes every step is triton.
#
# debug_simulate_encoded_softmax stays triton deliberately. FlyDSL has no
# equivalent, the kernel is a debugging aid rather than a performance path, and
# porting it would remove exactly the cross-DSL property this metro exists to
# test. Its launch_condition already carries the encoded_softmax != nullptr
# guard, so the conditional step needs nothing flyc-specific.
@ati.start
@ati.metro_kernel
def metro_fwd_flyc(params):
    flyc_attn_fwd(params)
    if params.encoded_softmax.data_ptr() != 0:
        debug_simulate_encoded_softmax(params)


# union_precedence: the KEY kernels (dk_dv, dq) own the canonical operand bindings;
# bwd_preprocess names some shared strides differently (dO's 4th stride is
# `stride_don` there vs `stride_dok` on the key kernels). When bwd_kernel_dq @ati.cites
# the whole metro, the gap donor must be a key kernel — this priority order (key first)
# steers both the cite gap-fill and the operator params-struct union.
@ati.start
@ati.hints.union_precedence([bwd_kernel_dk_dv, bwd_kernel_dq, bwd_preprocess])
@ati.metro_kernel
def metro_bwd(params):
    bwd_preprocess(params)
    bwd_kernel_dk_dv(params)
    bwd_kernel_dq(params)


# The same shape as metro_bwd with the FlyDSL kernels in place of the two key
# Triton ones, and the SAME Triton preprocess step in front. As with
# metro_fwd_flyc, the mix is the point rather than a convenience.
#
# bwd_preprocess stays Triton because FlyDSL has no equivalent: the flyc dK/dV
# and dQ kernels both READ `Delta = rowsum(dO * O)` and neither produces it
# (FlyDSL's own interfaces compute it in torch, on the host). bwd_preprocess
# computes exactly that quantity in fp32, so it feeds them directly with no
# adapter -- for every layout, since the varlen_bits port merged
# bwd_preprocess_varlen into bwd_preprocess and the merged kernel decodes the
# layout itself. This metro therefore calls it unconditionally, exactly as
# metro_bwd does.
#
# No @ati.hints.union_precedence: a flyc kdesc contributes no func_cfields
# (ir/flyc/kdesc.py), so this metro adds nothing to the operator's params-struct
# union -- every operand it touches already arrives via metro_bwd.
@ati.start
@ati.metro_kernel
def metro_bwd_flyc(params):
    bwd_preprocess(params)
    flyc_bwd_dkdv(params)
    flyc_bwd_dq(params)


# --- operators (declarative @ati.operator form) ---------------------------
#
# Stacked-@: @ati.start (top) ends the stack; @ati.operator (bottom, next to def)
# starts it. Backends are referenced by their in-file def; the linker derives the
# params struct (union over backends) and the functional-axes owner (default kernel).

@ati.start
# Operator-level partial tuning, declared EXPLICITLY (not inherited from a kernel).
@ati.tune.fallback(PADDED_HEAD=False)
@ati.tune.binning(Max_seqlen_q=ati.tune.binning.le,
                  Max_seqlen_k=ati.tune.binning.le)
# flyc, index 2 -- the metro, not the bare kernel, so the encoded_softmax debug
# step runs on this backend too. The linker builds flyc kdescs before the
# operators and binds their functional space afterwards (ir/ops/infer.py), because
# a flyc kernel both IS reachable as a backend and BORROWS this operator's
# functional space.
@ati.backend(2, metro_fwd_flyc, 'flyc')
@ati.backend(1, aiter_fmha_v3_fwd, 'aiter')
@ati.backend(0, metro_fwd, 'triton')
@ati.operator(call_options_name='attn_options')
def op_attn_fwd():
    pass


@ati.start
@ati.tune.binning(max_seqlen_q=ati.tune.binning.le,
                  max_seqlen_k=ati.tune.binning.le)
# flyc, index 3 -- the metro, so the Delta the flyc kernels read is produced.
# Same linker ordering as the forward's flyc backend: flyc kdescs are built
# before the operators and have their functional space bound afterwards
# (ir/ops/infer.py).
@ati.backend(3, metro_bwd_flyc, 'flyc')
@ati.backend(2, aiter_fmha_v3_bwd, 'aiter')
@ati.backend(1, bwd_kernel_fuse, 'triton_fuse')
@ati.backend(0, metro_bwd, 'triton_split')
@ati.operator(call_options_name='attn_options')
def op_attn_bwd():
    pass


operators = [op_attn_fwd, op_attn_bwd]
__all__ = ['operators']
