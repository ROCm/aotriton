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


def _load_ati_module(module_filename: str):
    """Import a modules/flash/*_ati.py description module by path (the 'flash'
    top-level package name collides with this v3python.rules.flash module, so we
    load under a private name)."""
    import sys as _sys
    import importlib.util as _ilu
    from pathlib import Path as _Path
    _root = _Path(__file__).resolve().parents[3]
    _tritonsrc = _root / 'tritonsrc'
    if str(_tritonsrc) not in _sys.path:
        _sys.path.insert(0, str(_tritonsrc))
    # Make modules/flash importable so description modules can share helpers
    # (e.g. `from _common_ati import flash_disabled`).
    _modflash = _root / 'modules' / 'flash'
    if str(_modflash) not in _sys.path:
        _sys.path.insert(0, str(_modflash))
    spec = _ilu.spec_from_file_location(
        '_ati_modules_flash_' + module_filename.replace('.py', ''),
        _root / 'modules' / 'flash' / module_filename)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if _ati_enabled('attn_fwd'):
    # Route attn_fwd through the ATI adapter (executive plan Step 4.2.5). The
    # description lives in modules/flash/attn_fwd_ati.py (Mode B; the Triton
    # source is untouched). The adapter is byte-for-byte equivalent to the legacy
    # __attn_fwd, so it drops into both `kernels` and the metro operator below.
    _mod = _load_ati_module('attn_fwd_ati.py')
    from fwd_kernel import attn_fwd as _ati_attn_fwd_jit
    from v3python.template_instantiation.compat import build_kernel_description
    _mod.describe_attn_fwd(_ati_attn_fwd_jit)
    __attn_fwd = build_kernel_description(_ati_attn_fwd_jit, family='flash',
                                         source_path=SOURCE_FILE,
                                         triton_kernel_name='attn_fwd')
    # SHARED_IFACE (which operator's param struct this kernel borrows) is NOT set
    # here — it is inferred from the operator -> metro -> kernel relationship by
    # infer_shared_iface() after `operators` is built below.
__bwd_kernel_dk_dv = bwd_kernel_dk_dv('bwd_kernel_dk_dv', SOURCE_FILE)
if _ati_enabled('bwd_kernel_dk_dv'):
    # Key bwd kernel via the ATI adapter (executive plan Step 12). Standalone full
    # description (no cite); SHARED_IFACE (OpAttnBwd) is inferred after `operators`.
    from v3python.template_instantiation.compat import build_kernel_description as _bkd
    _m_dkdv = _load_ati_module('bwd_kernel_dk_dv_ati.py')
    from bwd_kernel_dk_dv import bwd_kernel_dk_dv as _ati_dkdv_jit
    _m_dkdv.describe_bwd_kernel_dk_dv(_ati_dkdv_jit)
    __bwd_kernel_dk_dv = _bkd(_ati_dkdv_jit, family='flash', source_path=SOURCE_FILE,
                             triton_kernel_name='bwd_kernel_dk_dv')
__bwd_kernel_dq = bwd_kernel_dq('bwd_kernel_dq', SOURCE_FILE)
if _ati_enabled('bwd_kernel_dq'):
    from v3python.template_instantiation.compat import build_kernel_description as _bkd
    _m_dq = _load_ati_module('bwd_kernel_dq_ati.py')
    from bwd_kernel_dq import bwd_kernel_dq as _ati_dq_jit
    _m_dq.describe_bwd_kernel_dq(_ati_dq_jit)
    __bwd_kernel_dq = _bkd(_ati_dq_jit, family='flash', source_path=SOURCE_FILE,
                          triton_kernel_name='bwd_kernel_dq')
if _ati_enabled('bwd_preprocess'):
    # Aux bwd kernels; cite a bwd key kernel (built+registered above) for shared
    # operands. Built here (after dk_dv/dq) so the flat-registry cite resolves.
    assert _ati_enabled('bwd_kernel_dk_dv'), (
        'AOTRITON_ATI_KERNELS=bwd_preprocess requires bwd_kernel_dk_dv too '
        '(bwd_preprocess cites op_attn_bwd.triton_split.bwd_kernel_dk_dv)')
    from v3python.template_instantiation.compat import build_kernel_description as _bkd
    _m_pp = _load_ati_module('bwd_preprocess_ati.py')
    from bwd_preprocess import (bwd_preprocess as _ati_pp_jit,
                                bwd_preprocess_varlen as _ati_ppv_jit)
    _m_pp.describe_bwd_preprocess(_ati_pp_jit)
    __bwd_preprocess = _bkd(_ati_pp_jit, family='flash', source_path=SOURCE_FILE,
                           triton_kernel_name='bwd_preprocess')
    _m_pp.describe_bwd_preprocess_varlen(_ati_ppv_jit)
    __bwd_preprocess_varlen = _bkd(_ati_ppv_jit, family='flash', source_path=SOURCE_FILE,
                                  triton_kernel_name='bwd_preprocess_varlen')
__bwd_kernel_fuse = bwd_kernel_fuse('bwd_kernel_fuse', SOURCE_FILE)
if _ati_enabled('bwd_kernel_fuse'):
    # Alternative fused bwd backend; cites the bwd metro's sub-kernels (built above)
    # for the merged operand vocabulary, overriding BLOCK_DMODEL (<=256) and its
    # own perf. Built after dk_dv/dq/preprocess so the flat-registry cites resolve.
    assert _ati_enabled('bwd_kernel_dk_dv') and _ati_enabled('bwd_preprocess'), (
        'AOTRITON_ATI_KERNELS=bwd_kernel_fuse requires bwd_kernel_dk_dv, '
        'bwd_kernel_dq and bwd_preprocess (it cites all three)')
    from v3python.template_instantiation.compat import build_kernel_description as _bkd
    _m_fuse = _load_ati_module('bwd_kernel_fuse_ati.py')
    from bwd_kernel_fuse import bwd_kernel_fuse as _ati_fuse_jit
    _m_fuse.describe_bwd_kernel_fuse(_ati_fuse_jit)
    __bwd_kernel_fuse = _bkd(_ati_fuse_jit, family='flash', source_path=SOURCE_FILE,
                            triton_kernel_name='bwd_kernel_fuse')
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
    from dropout_rng import debug_simulate_encoded_softmax as _ati_debug_jit
    from v3python.template_instantiation.compat import build_kernel_description
    _mod2 = _load_ati_module('debug_simulate_encoded_softmax_ati.py')
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
    from v3python.template_instantiation.metro import lower_plan
    _mod = _load_ati_module('metro_fwd_ati.py')
    kernel_map = {
        'attn_fwd': __attn_fwd,
        'debug_simulate_encoded_softmax': __debug_simulate_encoded_softmax,
    }
    return lower_plan(_mod.metro_fwd.__ati_metro__, kernel_map,
                      lambda steps: MetroFwdKernel('triton', steps),
                      ConditionalKernel)


def _build_metro_bwd():
    """The backward metro (the `triton_split` backend). Hand-built by default; when
    AOTRITON_ATI_KERNELS includes 'metro_bwd', built from the @ati.metro_kernel
    transpiler via lower_plan — same MetroKernel/ConditionalKernel IR."""
    if not _ati_enabled('metro_bwd'):
        return MetroBwdKernel('triton_split',
                              [ConditionalKernel('num_seqlens', '> 0',
                                                 __bwd_preprocess_varlen, __bwd_preprocess),
                               __bwd_kernel_dk_dv,
                               __bwd_kernel_dq])
    from v3python.template_instantiation.metro import lower_plan
    _mod = _load_ati_module('metro_bwd_ati.py')
    kernel_map = {
        'bwd_preprocess': __bwd_preprocess,
        'bwd_preprocess_varlen': __bwd_preprocess_varlen,
        'bwd_kernel_dk_dv': __bwd_kernel_dk_dv,
        'bwd_kernel_dq': __bwd_kernel_dq,
    }
    return lower_plan(_mod.metro_bwd.__ati_metro__, kernel_map,
                      lambda steps: MetroBwdKernel('triton_split', steps),
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


def _build_op_attn_bwd():
    # op_attn_bwd stays the LEGACY operator: its params struct is the hand-coded
    # union of the metro's sub-kernels, and no single bwd key kernel is the feature
    # superset (unlike fwd's attn_fwd), so the AtiOperator default-backend-reuse
    # shortcut does not apply. The backends' metro carries the ATI dk_dv/dq
    # sub-kernels (they borrow OpAttnBwdParams via SHARED_IFACE inference). The ATI
    # op_attn_bwd — building the struct from union_params — is the deferred Step 10.
    return OpAttnBwd([
        _build_metro_bwd(),
        __bwd_kernel_fuse,
        __bwd_aiter,
    ])


operators = [
    _build_op_attn_fwd(),
    _build_op_attn_bwd(),
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

