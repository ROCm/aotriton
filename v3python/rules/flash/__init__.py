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

if _ati_enabled('debug_simulate_encoded_softmax'):
    # Route debug through the ATI adapter (executive plan Step 11). It cites the
    # forward metro's key kernel (attn_fwd) for the shared operands' practices, so
    # __attn_fwd must already be ATI-built+registered above. The cite resolves via
    # the flat kernel registry (op_attn_fwd is built later, below).
    assert _ati_enabled('attn_fwd'), (
        'AOTRITON_ATI_KERNELS=debug_simulate_encoded_softmax requires attn_fwd too '
        '(debug cites op_attn_fwd.triton.attn_fwd)')
    import importlib.util as _ilu2
    from pathlib import Path as _Path2
    from tritonsrc.dropout_rng import debug_simulate_encoded_softmax as _ati_debug_jit
    _ROOT2 = _Path2(__file__).resolve().parents[3]
    _spec2 = _ilu2.spec_from_file_location(
        '_ati_modules_flash_debug', _ROOT2 / 'modules' / 'flash'
        / 'debug_simulate_encoded_softmax_ati.py')
    _mod2 = _ilu2.module_from_spec(_spec2)
    _spec2.loader.exec_module(_mod2)
    from v3python.template_instantiation.compat import build_kernel_description
    _mod2.describe_debug_simulate_encoded_softmax(_ati_debug_jit)
    __debug_simulate_encoded_softmax = build_kernel_description(
        _ati_debug_jit, family='flash', source_path=SOURCE_FILE,
        triton_kernel_name='debug_simulate_encoded_softmax')

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

def _build_metro_fwd():
    """The forward metro. Hand-built by default; when AOTRITON_ATI_KERNELS includes
    'metro_fwd', built from the @ati.metro_kernel transpiler (Step 9) via lower_plan
    — which must lower to the SAME MetroKernel/ConditionalKernel IR."""
    if not _ati_enabled('metro_fwd'):
        return MetroFwdKernel('triton',
                              [__attn_fwd,
                               ConditionalKernel('encoded_softmax', '->data_ptr() != nullptr',
                                                 __debug_simulate_encoded_softmax)])
    import importlib.util as _ilu
    from pathlib import Path as _Path
    from v3python.template_instantiation.metro import lower_plan
    _ROOT = _Path(__file__).resolve().parents[3]
    _spec = _ilu.spec_from_file_location(
        '_ati_modules_flash_metro_fwd', _ROOT / 'modules' / 'flash' / 'metro_fwd_ati.py')
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    kernel_map = {
        'attn_fwd': __attn_fwd,
        'debug_simulate_encoded_softmax': __debug_simulate_encoded_softmax,
    }
    return lower_plan(_mod.metro_fwd.__ati_metro__, kernel_map,
                      lambda steps: MetroFwdKernel('triton', steps),
                      ConditionalKernel)


def _build_op_attn_fwd():
    backends = [
        _build_metro_fwd(),
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

# Register operators in the ops registry so @ati.cite("<op>.<metro>[.<kernel>]")
# can resolve metros/sub-kernels (executive plan Step 8). Each op exposes
# FAMILY/NAME/get_backend; metros expose get_kernel/iter_subkernels.
from v3python.template_instantiation import registry as _ati_registry
for _op in operators:
    if getattr(_op, 'FAMILY', None) is not None:
        _ati_registry.register_op(_op)

