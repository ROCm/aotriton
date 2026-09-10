# Copyright © 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Op-level impl provider for flash (modular-tune.md Revision note 3): plain
module-level functions (NOT a TuningLevel/strategy-object subclass -- that
intermediate layer was removed) selecting among backend implementations of a
whole operator (attn_fwd / attn_bwd), using only `force_backend_index` -- no
KernelControl / kernel_fine_control. Needs only the plain *testing* pyaotriton
library build, as opposed to the tuning-instrumented build used by
level_kernel.py.

`FlashTune` (modules/flash/tune/desc.py) is the only caller of this module: it
lazily imports `level_op` from its `get_impl()`/`_do_probe_all_impls()`
methods whenever a DSL name carries the `op.` prefix, and otherwise never
touches it -- see desc.py's docstring for the exact dispatch and F3's
lazy-import boundary.

IMPORTANT: like level_kernel.py, this module must stay torch/pyaotriton-free
AT MODULE SCOPE -- see level_kernel.py's docstring for why. All torch imports
are deferred into get_impl() / enumerate_variants() / impl_desc().

Highest-risk area #1 (modular-tune.md): list_impls() below returns BARE
interface names ('attn_fwd', 'attn_bwd'), never the `op.`-prefixed DSL surface
syntax -- FlashTune.list_impls() (the caller) applies the `op.` prefix, and
FlashTune.get_impl()/_do_probe_all_impls() strip it back off (via
ImplSelector.split_dsl_name()) before calling into this module.
"""

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
    called once, from get_impl(); cached at module scope."""
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
        # The index vocabulary is pyaotriton.v3.flash.OpAttnFwdBackend, generated
        # from the same list that assigns BackendEnum. Read it rather than the
        # comment that used to sit here.
        #
        # BACKEND_COUNT is NOT OpAttnFwdBackend.Max, and the difference is the
        # point: Max is how many backends the LIBRARY was generated with (the
        # enum is the same on every arch), while this is how many the tuner
        # sweeps here. aiter is gfx942/gfx950-only, so a tuner driven by Max
        # would probe a backend that cannot run there.
        #
        # It is NOT "how many are available on the arch in hand" either, and the
        # gap is deliberate: flyc took fwd index 2 and bwd index 3, and these
        # counts do not include it, so `enumerate_variants` never probes flyc on
        # gfx1201 or gfx950 -- the two arches that DO have flyc kernels -- and
        # produces no tuning-DB rows for it. Raising the counts is not the fix
        # on its own: whether a library carries flyc IMAGES is a build-time
        # choice independent of the arch, and the tuner has no probe for it, so
        # a raised count would fail every sweep on a flyc-less build of the same
        # arch. flyc stays reachable through an explicit `force_backend_index`
        # until that probe exists.

        @property
        def BACKEND_COUNT(self):
            return 2 if _gpu_arch() in ('gfx942', 'gfx950') else 1

    class attn_bwd(SdpaOpCommon, _bwd_kernel_dk_dv):
        # See attn_fwd above on OpAttnBwdBackend and why BACKEND_COUNT is not Max.

        OUTPUT_TNAMES = ["dk", "dv", "dq", "db"]

        @property
        def BACKEND_COUNT(self):
            return 3 if _gpu_arch() in ('gfx942', 'gfx950') else 2

        def direct_call(self, direct_inputs, extargs):
            im, view, devm = direct_inputs
            import torch
            from aotriton.tune.gpu_utils import zero_devm
            # The aiter backend accumulates into dq_acc; clear before
            # each call. Named, not 2: the literal was correct only as long as
            # nobody inserted a backend ahead of it, which is exactly what
            # happened to attn_fwd when flyc took index 2.
            from pyaotriton.v3.flash import OpAttnBwdBackend
            if extargs.backend_index == OpAttnBwdBackend.kAiter:
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


def list_impls(entry, arch: str | None = None) -> list[str]:
    """Bare iface names for the op level -- FlashTune.list_impls() applies the
    'op.' DSL prefix to these before returning them."""
    return ['attn_fwd', 'attn_bwd']


def get_impl(name: str):
    global _OP_DICT_CACHE
    if _OP_DICT_CACHE is None:
        _OP_DICT_CACHE = _build_op_dict()
    return _OP_DICT_CACHE[name]


def enumerate_variants(entry, im, which_impl: str, pt) -> list[dict]:
    kernel = get_impl(which_impl)
    return [{'backend_index': i} for i in range(kernel.BACKEND_COUNT)]


def impl_desc(kernel, args) -> dict:
    return {'backend_index': args.backend_index}
