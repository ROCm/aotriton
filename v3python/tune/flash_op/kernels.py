# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from ..flash.kernel_calls import (
    SdpaCalls,
    attn_fwd,
    bwd_kernel_dk_dv,
)
import torch

_cached_arch = None

def _gpu_arch() -> str:
    global _cached_arch
    if _cached_arch is None:
        _cached_arch = torch.cuda.get_device_properties(0).gcnArchName.split(':')[0]
    return _cached_arch

class AttnOptionsWrapperOp:
    """
    Wraps attn_options from the *testing* version of pyaotriton (installed/test/).
    The testing library has no KernelControl / kernel_fine_control; only
    force_backend_index is used.  All imports are lazy so that importing this
    module outside a GPU container does not fail.
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


class SdpaOpCommon(SdpaCalls):
    EXT_CLASS = AttnOptionsWrapperOp
    BACKEND_COUNT = None  # must define in subclass

    def create_extargs(self, *, which_impl=None, probe=False):
        backend_index = which_impl.backend_index if which_impl is not None else 0
        return self.EXT_CLASS.for_op_backend(backend_index)


class attn_fwd_op(SdpaOpCommon, attn_fwd):
    # kMetro_Triton=0, kSlimAffine_AiterFmhaV3Fwd=1 (gfx942/gfx950 only)

    @property
    def BACKEND_COUNT(self):
        return 2 if _gpu_arch() in ('gfx942', 'gfx950') else 1


class attn_bwd_op(SdpaOpCommon, bwd_kernel_dk_dv):
    # kMetro_TritonSplit=0, kShim_BwdKernelFuse=1, kSlimAffine_AiterFmhaV3Bwd=2 (gfx942/gfx950 only)

    @property
    def BACKEND_COUNT(self):
        return 3 if _gpu_arch() in ('gfx942', 'gfx950') else 2

    def direct_call(self, direct_inputs, extargs):
        im, view, devm = direct_inputs
        from ..gpu_utils import zero_devm
        if extargs.backend_index == 2:  # kSlimAffine_AiterFmhaV3Bwd accumulates into dq_acc; clear before each call.
            zero_devm(devm.dq_acc)
        return super().direct_call(direct_inputs, extargs)

    def prepare_directs(self, im, inputs):
        im, view, devm = super().prepare_directs(im, inputs)
        from pyaotriton import lazy_tensor
        from ..gpu_utils import mk_aotensor
        devm.dq_acc = torch.zeros(*devm.q.size(), dtype=torch.float32, device=devm.q.device)
        dq_acc_view, _ = mk_aotensor(devm.dq_acc)
        view.dq_acc = lazy_tensor.eager_null_dq_acc(dq_acc_view)
        return im, view, devm
