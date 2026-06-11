# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from v3python.op import (
    MetroKernel,
    ConditionalKernel,
)
from .ops import OpAttnFwd, OpAttnBwd
from .attn_fwd import attn_fwd
from .bwd_preprocess import (
    bwd_preprocess,
    bwd_preprocess_varlen,
)
from .bwd_postprocess import bwd_postprocess
from .bwd_kernel_dk_dv import bwd_kernel_dk_dv
from .bwd_kernel_dq import bwd_kernel_dq
from .bwd_kernel_fuse import bwd_kernel_fuse
# from .debug_fill_dropout_rng import debug_fill_dropout_rng, debug_fill_dropout_rng_tensor
from .debug_simulate_encoded_softmax import debug_simulate_encoded_softmax
from .aiter_fwd import aiter_fmha_v3_fwd
from .aiter_bwd import aiter_fmha_v3_bwd

SOURCE_FILE = 'tritonsrc/flash.py'

__bwd_preprocess = bwd_preprocess('bwd_preprocess', SOURCE_FILE)
__bwd_preprocess_varlen = bwd_preprocess_varlen('bwd_preprocess_varlen', SOURCE_FILE)
__attn_fwd = attn_fwd('attn_fwd', SOURCE_FILE)


def _ati_enabled(kernel_name: str) -> bool:
    import os
    sel = os.getenv('AOTRITON_ATI_KERNELS', default='')
    return kernel_name in sel.replace(',', ' ').split()


if _ati_enabled('attn_fwd'):
    # Route attn_fwd through the ATI adapter (executive plan Step 4.2.5). The
    # description lives in modules/flash/attn_fwd_ati.py (Mode B; the Triton
    # source is untouched). The adapter is byte-for-byte equivalent to the legacy
    # __attn_fwd, so it drops into both `kernels` and the metro operator below.
    import sys as _sys
    import importlib.util as _ilu
    from pathlib import Path as _Path
    _ROOT = _Path(__file__).resolve().parents[3]
    _TRITONSRC = _ROOT / 'tritonsrc'
    if str(_TRITONSRC) not in _sys.path:
        _sys.path.insert(0, str(_TRITONSRC))
    from fwd_kernel import attn_fwd as _ati_attn_fwd_jit
    # Load modules/flash/attn_fwd_ati.py by path — 'flash' as a top-level package
    # name collides with this very module (v3python.rules.flash), so import it
    # under a private module name instead.
    _spec = _ilu.spec_from_file_location(
        '_ati_modules_flash_attn_fwd', _ROOT / 'modules' / 'flash' / 'attn_fwd_ati.py')
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    from v3python.template_instantiation.compat import build_kernel_description
    _mod.describe_attn_fwd(_ati_attn_fwd_jit)
    __attn_fwd = build_kernel_description(_ati_attn_fwd_jit, family='flash',
                                         source_path=SOURCE_FILE,
                                         triton_kernel_name='attn_fwd')
    # SHARED_IFACE (which operator's param struct this kernel borrows) is NOT set
    # here — it is inferred from the operator -> metro -> kernel relationship by
    # infer_shared_iface() after `operators` is built below.
__bwd_kernel_dk_dv = bwd_kernel_dk_dv('bwd_kernel_dk_dv', SOURCE_FILE)
__bwd_kernel_dq = bwd_kernel_dq('bwd_kernel_dq', SOURCE_FILE)
__bwd_kernel_fuse = bwd_kernel_fuse('bwd_kernel_fuse', SOURCE_FILE)
__fwd_aiter = aiter_fmha_v3_fwd()
__bwd_aiter = aiter_fmha_v3_bwd()
# # TODO: Re-implement this as part of kernel(?)
__debug_simulate_encoded_softmax = debug_simulate_encoded_softmax('debug_simulate_encoded_softmax', SOURCE_FILE)

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

class MetroBwdKernel(MetroKernel):
    FAMILY = OpAttnBwd.FAMILY
    SHARED_IFACE = OpAttnBwd
    ARGUMENTS = OpAttnBwd.ARGUMENTS

class MetroFwdKernel(MetroKernel):
    FAMILY = OpAttnFwd.FAMILY
    SHARED_IFACE = OpAttnFwd
    ARGUMENTS = OpAttnFwd.ARGUMENTS

def _build_op_attn_fwd():
    backends = [
        MetroFwdKernel('triton',
                       [__attn_fwd,
                        ConditionalKernel('encoded_softmax', '->data_ptr() != nullptr', __debug_simulate_encoded_softmax)]),
        __fwd_aiter,  # No need to provide encoded_softmax because no dropout support
    ]
    if _ati_enabled('op_attn_fwd'):
        # ATI operator: dispatch among backends; the param struct is the DEFAULT
        # backend's (attn_fwd, the feature superset). Reuses the ATI attn_fwd
        # adapter for the functional/struct surface. (union_params over the metro
        # sub-kernels — projecting the debug kernel's private args — lands with the
        # metro transpiler, Step 5.5; today the default backend's struct is used.)
        from v3python.template_instantiation.compat.operator_adapter import AtiOperator
        from v3python.template_instantiation.tune.binning import binning as _bn
        return AtiOperator('op_attn_fwd', family='flash', default_kdesc=__attn_fwd,
                           backends=backends,
                           optune_keys={'Max_seqlen_q': _bn.le, 'Max_seqlen_k': _bn.le},
                           call_options_name='attn_options')
    return OpAttnFwd(backends)


operators = [
    _build_op_attn_fwd(),
    OpAttnBwd([
        MetroBwdKernel('triton_split',
                       [ConditionalKernel('num_seqlens', '> 0', __bwd_preprocess_varlen, __bwd_preprocess),  # padded varlen (num_seqlens < 0) should call bwd_preprocess
                        __bwd_kernel_dk_dv,
                        __bwd_kernel_dq]),
        __bwd_kernel_fuse,
        __bwd_aiter,
    ]),
]

# Infer SHARED_IFACE (the param struct a kernel borrows) from the
# operator -> metro -> kernel relationship. Legacy kernels declare it on their
# class; ATI adapter kernels leave it None and have it filled in here. No-op for
# the all-legacy build; replaces the Step-4 hand-set scaffolding.
from v3python.template_instantiation.operator.infer import infer_shared_iface
infer_shared_iface(operators)

