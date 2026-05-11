# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from ..flash.kernel_calls import (
    SdpaCalls,
    attn_fwd,
    bwd_kernel_dk_dv,
)


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
    # kMetro_Triton=0, kSlimAffine_AiterFmhaV3Fwd=1
    BACKEND_COUNT = 2


class attn_bwd_op(SdpaOpCommon, bwd_kernel_dk_dv):
    # kMetro_TritonSplit=0, kShim_BwdKernelFuse=1, kSlimAffine_AiterFmhaV3Bwd=2
    BACKEND_COUNT = 3

    def prepare_directs(self, im, inputs):
        view = super().prepare_directs(im, inputs)
        # Replace null dq_acc with a real lazy accumulator for non-Triton backends
        from pyaotriton import lazy_tensor, T4
        from ..gpu_utils import cast_dtype
        dq = view.dq
        dq_view = T4(dq.data_ptr(), tuple(dq.size()), dq.stride(), cast_dtype(dq.dtype))
        view.dq_acc = lazy_tensor.dq_acc(dq_view, dq.device.index)
        return view
