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
from .bwd_preprocess_varlen import bwd_preprocess_varlen
from .debug_simulate_encoded_softmax import debug_simulate_encoded_softmax
from .aiter_fwd import aiter_fmha_v3_fwd
from .aiter_bwd import aiter_fmha_v3_bwd


# --- triton metro backends (transpiled, never executed) -------------------

@ati.metro_kernel
def metro_fwd(params):
    attn_fwd(params)
    if params.encoded_softmax.data_ptr() != 0:
        debug_simulate_encoded_softmax(params)


# union_precedence: the KEY kernels (dk_dv, dq) own the canonical operand bindings;
# the preprocess kernels name some shared strides differently (dO's 4th stride is
# `stride_don` there vs `stride_dok` on the key kernels). When bwd_kernel_dq @ati.cites
# the whole metro, the gap donor must be a key kernel — this priority order (key first)
# steers both the cite gap-fill and the operator params-struct union.
@ati.hints.union_precedence([bwd_kernel_dk_dv, bwd_kernel_dq,
                             bwd_preprocess_varlen, bwd_preprocess])
@ati.metro_kernel
def metro_bwd(params):
    if params.num_seqlens > 0:
        bwd_preprocess_varlen(params)
    else:
        bwd_preprocess(params)
    bwd_kernel_dk_dv(params)
    bwd_kernel_dq(params)


# --- operators (declarative @ati.operator form) ---------------------------
#
# Stacked-@: @ati.kernel (top) ends the stack; @ati.operator (bottom, next to def)
# starts it. Backends are referenced by their in-file def; the linker derives the
# params struct (union over backends) and the functional-axes owner (default kernel).

@ati.kernel
# Operator-level partial tuning, declared EXPLICITLY (not inherited from a kernel).
@ati.tune.fallback(PADDED_HEAD=False)
@ati.tune.binning(Max_seqlen_q=ati.tune.binning.le,
                  Max_seqlen_k=ati.tune.binning.le)
@ati.backend(1, aiter_fmha_v3_fwd, 'aiter')
@ati.backend(0, metro_fwd, 'triton')
@ati.operator(call_options_name='attn_options')
def op_attn_fwd():
    pass


@ati.kernel
@ati.tune.binning(max_seqlen_q=ati.tune.binning.le,
                  max_seqlen_k=ati.tune.binning.le)
@ati.backend(2, aiter_fmha_v3_bwd, 'aiter')
@ati.backend(1, bwd_kernel_fuse, 'bwd_kernel_fuse')
@ati.backend(0, metro_bwd, 'triton_split')
@ati.operator(call_options_name='attn_options')
def op_attn_bwd():
    pass


operators = [op_attn_fwd, op_attn_bwd]
__all__ = ['operators']
