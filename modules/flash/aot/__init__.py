# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Flash family build block (the ATI registry for the `flash` family).

Enumerating `modules/<family>/aot` packages drives the per-family build (interim
protocol; `--module_dir` deferred). This module imports the `aot/` kernel
descriptions (each finalizes on import via `@ati.kernel`/`@ati.source`), builds the
AtiKernelDescription for each kernel, assembles the metros (from the
`@ati.metro_kernel` descriptions via `lower_plan`) and the operators, runs
SHARED_IFACE inference + operator registration, and exposes the family's
`kernels` / `operators` / `affine_kernels` for the generator (`aotriton.rules`).

ATI-only: the legacy env-switched per-kernel triton classes are gone; the
generated code is the full-ATI output (the re-baselined golden).

DESIGN NOTE (deferred to next phase): this module should ultimately expose only
`operators`; each operator should refer to its kernels (triton metros + affine
backends), and the code generator (generate.py / the ATI package) should do all the
heavy lifting of assembling metros and kdescs. The `_build_metro_*` / per-kernel
`build_kernel_description` wiring here is interim scaffolding for this phase and is
to be pushed down into the operator/generator layers next phase.
"""

from pathlib import Path

import aotriton.template_instantiation as ati
from aotriton.op import MetroKernel, ConditionalKernel
from aotriton.template_instantiation.compat import build_kernel_description
from aotriton.template_instantiation.metro import lower_plan
from aotriton.template_instantiation.ops.infer import infer_shared_iface
from aotriton.template_instantiation import registry as _ati_registry
from aotriton.template_instantiation.compat.operator_adapter import (
    build_merged_struct_cfields)

from .ops import OpAttnFwd, OpAttnBwd
from . import (
    attn_fwd as _d_attn_fwd,
    bwd_kernel_dk_dv as _d_dk_dv,
    bwd_kernel_dq as _d_dq,
    debug_simulate_encoded_softmax as _d_debug,
    bwd_preprocess as _d_pp,
    bwd_preprocess_varlen as _d_ppv,
    bwd_kernel_fuse as _d_fuse,
    metro_fwd as _m_fwd,
    metro_bwd as _m_bwd,
)
from .aiter_fwd import aiter_fmha_v3_fwd
from .aiter_bwd import aiter_fmha_v3_bwd

# Each kernel's own Triton source file under ../kernel/ (per-kernel source_path —
# no flash.py aggregator). Relative to this package's parent (modules/flash/).
_KERNEL_DIR = Path(__file__).resolve().parent.parent / 'kernel'


def _build(desc_module, kernel_attr, src_filename):
    """build_kernel_description for one aot/ description (which has already finalized
    its kernel object on import), with the kernel's own source file."""
    kernel = getattr(desc_module, kernel_attr)
    return build_kernel_description(
        kernel, family='flash',
        source_path=str(_KERNEL_DIR / src_filename),
        triton_kernel_name=kernel_attr)


# Key kernels first (the cite-based aux kernels resolve against them via the flat
# kernel registry, so build order is key-before-aux).
__attn_fwd = _build(_d_attn_fwd, 'attn_fwd', 'fwd_kernel.py')
__bwd_kernel_dk_dv = _build(_d_dk_dv, 'bwd_kernel_dk_dv', 'bwd_kernel_dk_dv.py')
__bwd_kernel_dq = _build(_d_dq, 'bwd_kernel_dq', 'bwd_kernel_dq.py')
# Aux kernels (cite the key kernels).
__debug_simulate_encoded_softmax = _build(
    _d_debug, 'debug_simulate_encoded_softmax', 'dropout_rng.py')
__bwd_preprocess = _build(_d_pp, 'bwd_preprocess', 'bwd_preprocess.py')
__bwd_preprocess_varlen = _build(_d_ppv, 'bwd_preprocess_varlen', 'bwd_preprocess.py')
__bwd_kernel_fuse = _build(_d_fuse, 'bwd_kernel_fuse', 'bwd_kernel_fuse.py')

# Affine (asm) backends.
__fwd_aiter = aiter_fmha_v3_fwd()
__bwd_aiter = aiter_fmha_v3_bwd()


kernels = [
    __bwd_preprocess,
    __bwd_preprocess_varlen,
    __attn_fwd,
    __bwd_kernel_dk_dv,
    __bwd_kernel_dq,
    __bwd_kernel_fuse,
    __debug_simulate_encoded_softmax,
]

affine_kernels = [
    __fwd_aiter,
    __bwd_aiter,
]


class MetroFwdKernel(MetroKernel):
    FAMILY = OpAttnFwd.FAMILY
    SHARED_IFACE = OpAttnFwd
    ARGUMENTS = OpAttnFwd.ARGUMENTS


class MetroBwdKernel(MetroKernel):
    FAMILY = OpAttnBwd.FAMILY
    SHARED_IFACE = OpAttnBwd
    ARGUMENTS = OpAttnBwd.ARGUMENTS


def _build_metro_fwd():
    kernel_map = {
        'attn_fwd': __attn_fwd,
        'debug_simulate_encoded_softmax': __debug_simulate_encoded_softmax,
    }
    return lower_plan(_m_fwd.metro_fwd.__ati_metro__, kernel_map,
                      lambda steps: MetroFwdKernel('triton', steps),
                      ConditionalKernel)


def _build_metro_bwd():
    kernel_map = {
        'bwd_preprocess': __bwd_preprocess,
        'bwd_preprocess_varlen': __bwd_preprocess_varlen,
        'bwd_kernel_dk_dv': __bwd_kernel_dk_dv,
        'bwd_kernel_dq': __bwd_kernel_dq,
    }
    return lower_plan(_m_bwd.metro_bwd.__ati_metro__, kernel_map,
                      lambda steps: MetroBwdKernel('triton_split', steps),
                      ConditionalKernel)


# --- operators (declarative @ati.operator form) ---------------------------
#
# Built metros (the triton backends). Affine backends are the already-built legacy
# objects (__fwd_aiter / __bwd_aiter); porting their internals to ATI is deferred.
_metro_fwd = _build_metro_fwd()
_metro_bwd = _build_metro_bwd()

# The bwd params struct has no single owning kernel: it is the union of the metro
# sub-kernels' fields (KEY-first: dk_dv, dq, then the preprocess kernels that
# contribute Out — this order reproduces the legacy struct order), plus DQ_ACC
# (which lives only on the deferred affine backend) injected after DB. The fwd
# struct is just attn_fwd's (the feature superset), so no struct_cfields there.
_bwd_struct = build_merged_struct_cfields(
    [__bwd_kernel_dk_dv, __bwd_kernel_dq,
     __bwd_preprocess_varlen, __bwd_preprocess],
    inject={'name': 'DQ_ACC', 'after': 'DB',
            'ctype': 'LazyTensorInternal<4>*', 'nbits': 0})


# Stacked-@ operator form: @ati.kernel (top) ends the stack; @ati.operator (bottom,
# next to def) STARTS the description (like @ati.source). Decorators apply bottom-up,
# so @ati.operator runs first and seeds the pending list.
@ati.kernel
# Operator-level partial tuning, declared EXPLICITLY (not inherited from a kernel):
# matches the legacy OpAttnFwd.PARTIALLY_TUNED_FUNCTIONALS. op_attn_bwd declares none.
@ati.tune.fallback(PADDED_HEAD=False)
@ati.tune.binning(Max_seqlen_q=ati.tune.binning.le,
                  Max_seqlen_k=ati.tune.binning.le)
@ati.backend(1, __fwd_aiter, 'aiter')
@ati.backend(0, _metro_fwd, 'triton')
@ati.operator(family='flash', call_options_name='attn_options',
              default_kdesc=__attn_fwd)
def op_attn_fwd():
    pass


@ati.kernel
@ati.tune.binning(max_seqlen_q=ati.tune.binning.le,
                  max_seqlen_k=ati.tune.binning.le)
@ati.backend(2, __bwd_aiter, 'aiter')
@ati.backend(1, __bwd_kernel_fuse, 'bwd_kernel_fuse')
@ati.backend(0, _metro_bwd, 'triton_split')
@ati.operator(family='flash', call_options_name='attn_options',
              default_kdesc=__bwd_kernel_dk_dv, struct_cfields=_bwd_struct)
def op_attn_bwd():
    pass


operators = [op_attn_fwd, op_attn_bwd]

# Infer SHARED_IFACE (operator -> metro -> kernel), then register operators so
# @ati.cite("<op>.<metro>[.<kernel>]") resolves.
infer_shared_iface(operators)
for _op in operators:
    if getattr(_op, 'FAMILY', None) is not None:
        _ati_registry.register_op(_op)

