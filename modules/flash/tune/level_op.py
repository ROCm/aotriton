# Copyright © 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Op-level tuning strategy for flash (modular-tune.md §4.1/§4.3): selects among
backend implementations of a whole operator (attn_fwd / attn_bwd), using only
`force_backend_index` -- no KernelControl / kernel_fine_control. Needs only
the plain *testing* pyaotriton library build (REQUIRES_TUNING_LIB = False),
as opposed to the tuning-instrumented build used by level_kernel.py.

IMPORTANT: like level_kernel.py, this module must stay torch/pyaotriton-free
AT MODULE SCOPE -- see level_kernel.py's docstring for why. All torch imports
are deferred into get_impl() / enumerate_variants() / impl_desc().

Highest-risk area #1 (modular-tune.md): list_impls() below returns BARE
interface names ('attn_fwd', 'attn_bwd'), never the `op.`-prefixed DSL surface
syntax used by ImplSelector.as_text()/parse_text() -- that prefix is added/
stripped by ImplSelector itself, one level up.
"""

from aotriton.tune.tdesc import TuningLevel

_cached_arch = None

def _gpu_arch() -> str:
    global _cached_arch
    if _cached_arch is None:
        import torch
        _cached_arch = torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]
    return _cached_arch


_OP_DICT_CACHE = None


def _build_op_dict():
    """Lazily compose the testing-lib op-backend classes (force_backend_index
    layered onto the plain SdpaCalls direct_call implementations). Only
    called once, from FlashOpLevel.get_impl(); cached at module scope."""
    from aotriton.tune.kftdesc import BackendForTuneDescription
    from .calls import SdpaCalls, attn_fwd as _attn_fwd, bwd_kernel_dk_dv as _bwd_kernel_dk_dv

    class AttnOptionsWrapperOp:
        """
        Wraps attn_options from the *testing* version of pyaotriton (installed/test/).
        The testing library has no KernelControl / kernel_fine_control; only
        force_backend_index is used.
        """

        def __init__(self):
            from pyaotriton.v3.flash import attn_options as _attn_options
            self._c = _attn_options()

        @classmethod
        def for_op_backend(cls, backend_index: int) -> 'AttnOptionsWrapperOp':
            obj = cls()
            obj._backend = backend_index
            obj._c.force_backend_index = backend_index
            return obj

        @property
        def backend_index(self) -> int:
            return self._backend

        @property
        def c_object(self):
            return self._c

        def disable_probing(self):
            """Stub — op tuning has no probing phase."""
            pass

    class SdpaOpCommon(BackendForTuneDescription, SdpaCalls):
        EXT_CLASS = AttnOptionsWrapperOp
        BACKEND_COUNT = None  # must define in subclass

        def create_extargs(self, *, which_impl=None, probe=False):
            backend_index = which_impl.impl_index if which_impl is not None else 0
            return self.EXT_CLASS.for_op_backend(backend_index)

    class attn_fwd(SdpaOpCommon, _attn_fwd):
        # kMetro_Triton=0, kSlimAffine_AiterFmhaV3Fwd=1 (gfx942/gfx950 only)

        @property
        def BACKEND_COUNT(self):
            return 2 if _gpu_arch() in ('gfx942', 'gfx950') else 1

    class attn_bwd(SdpaOpCommon, _bwd_kernel_dk_dv):
        # kMetro_TritonSplit=0, kShim_BwdKernelFuse=1, kSlimAffine_AiterFmhaV3Bwd=2 (gfx942/gfx950 only)

        OUTPUT_TNAMES = ["dk", "dv", "dq", "db"]

        @property
        def BACKEND_COUNT(self):
            return 3 if _gpu_arch() in ('gfx942', 'gfx950') else 2

        def direct_call(self, direct_inputs, extargs):
            im, view, devm = direct_inputs
            import torch
            from aotriton.tune.gpu_utils import zero_devm
            if extargs.backend_index == 2:  # kSlimAffine_AiterFmhaV3Bwd accumulates into dq_acc; clear before each call.
                zero_devm(devm.dq_acc)
            err = self._direct_call(direct_inputs, extargs)
            return (devm.dk, devm.dv, devm.dq, devm.db), err

        def prepare_directs(self, im, inputs):
            im, view, devm = super().prepare_directs(im, inputs)
            import torch
            from pyaotriton import lazy_tensor
            from aotriton.tune.gpu_utils import mk_aotensor
            # FIXME: only allocate when backend == 2 (AITER); other backends don't
            # need dq_acc. Current interface does not support this — `args` would
            # need to be threaded into prepare_directs to know the backend index.
            devm.dq_acc = torch.zeros(*devm.q.size(), dtype=torch.float32, device=devm.q.device)
            dq_acc_view, _ = mk_aotensor(devm.dq_acc)
            view.dq_acc = lazy_tensor.eager_null_dq_acc(dq_acc_view)
            return im, view, devm

    return {
        'attn_fwd': attn_fwd(),
        'attn_bwd': attn_bwd(),
    }


class FlashOpLevel(TuningLevel):
    NAME = 'op'
    REQUIRES_TUNING_LIB = False

    def list_impls(self, entry, arch: str | None = None) -> list[str]:
        return ['attn_fwd', 'attn_bwd']

    def get_impl(self, name: str):
        global _OP_DICT_CACHE
        if _OP_DICT_CACHE is None:
            _OP_DICT_CACHE = _build_op_dict()
        return _OP_DICT_CACHE[name]

    def enumerate_variants(self, entry, im, which_impl: str, pt) -> list[dict]:
        kernel = self.get_impl(which_impl)
        return [{'backend_index': i} for i in range(kernel.BACKEND_COUNT)]

    def impl_desc(self, kernel, args) -> dict:
        return {'backend_index': args.backend_index}
