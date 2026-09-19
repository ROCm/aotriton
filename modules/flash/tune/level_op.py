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


# Arch restrictions, keyed by the name @ati.backend declared. Absent = runs
# everywhere. Mirrors the description layer, which decides this at codegen time
# against a Functional while the tuner needs it at run time:
#   aiter : @ati.affine.arch in aot/aiter_fwd.py:31, aiter_bwd.py:33
#   flyc  : the f.arch disable predicates in aot/flyc_*.py
_BACKEND_ARCHS = {
    'aiter': frozenset({'gfx942', 'gfx950'}),
    'flyc': frozenset({'gfx950', 'gfx1201'}),
}


def _backend_table(enum_struct_name: str):
    """(names in library-index order, {name: index}), from <Struct>.by_index.

    by_index is built from the generator's X-macro (bindings/v3.cc:118-121) and
    keyed by the names @ati.backend declared, so reordering a backend in the
    description needs no edit here.
    """
    from pyaotriton.v3 import flash
    struct = getattr(flash, enum_struct_name)
    by_index = dict(struct.by_index)
    names = [by_index[i] for i in range(struct.Max)]
    return names, {name: i for i, name in enumerate(names)}


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
        def for_op_backend(cls, backend_index: int, backend_name: str) -> 'AttnOptionsWrapperOp':
            obj = cls()
            obj._backend = backend_index
            obj._backend_name = backend_name
            obj._c.force_backend_index = backend_index
            return obj

        @property
        def backend_index(self) -> int:
            return self._backend

        @property
        def backend_name(self) -> str:
            """The name `@ati.backend` declared -- 'triton', 'aiter', 'flyc',
            'triton_split', 'triton_fuse'. Carried alongside the index because
            the index alone is not an identity: it is positional in the
            operator's backend list, so it means different things for fwd and
            bwd and would silently change meaning if a backend were inserted."""
            return self._backend_name

        @property
        def c_object(self):
            return self._c

        def disable_probing(self):
            """Stub — op tuning has no probing phase."""
            pass

    class SdpaOpCommon(BackendForTuneDescription, SdpaCalls):
        EXT_CLASS = AttnOptionsWrapperOp
        BACKEND_ENUM = None  # 'OpAttnFwdBackend' / 'OpAttnBwdBackend'; set in subclass

        def available_backends(self, dtype: str) -> list[str]:
            """Declared names of the backends that can run this arch and dtype,
            in library index order. This list defines the op-level variant
            space: its LENGTH is how many variants get benchmarked and its
            ORDER fixes which impl_index means which backend."""
            names, _ = _backend_table(self.BACKEND_ENUM)
            arch = _gpu_arch()
            out = []
            for n in names:
                if arch not in _BACKEND_ARCHS.get(n, frozenset({arch})):
                    continue
                if n == 'flyc' and dtype == 'float32':
                    continue
                out.append(n)
            return out

        def backend_index_of(self, name: str) -> int:
            _, index_of = _backend_table(self.BACKEND_ENUM)
            return index_of[name]

        def create_extargs(self, *, which_impl=None, probe=False, dtype=None):
            # impl_index is a POSITION in enumerate_variants()'s list, not a
            # backend index; on gfx1201 position 1 is flyc=2, not aiter=1.
            # `dtype` must match what enumerate_variants saw or this indexes a
            # differently-filtered list.
            names = self.available_backends(dtype)
            position = which_impl.impl_index if which_impl is not None else 0
            name = names[position]
            return self.EXT_CLASS.for_op_backend(self.backend_index_of(name), name)

    class attn_fwd(SdpaOpCommon, _attn_fwd):
        # The vocabulary is pyaotriton.v3.flash.OpAttnFwdBackend, generated
        # from the same list that assigns BackendEnum. triton=0, aiter=1, flyc=2.
        #
        # Caveat: whether a library carries flyc IMAGES is a build-time
        # property, not an arch one, and there is still no probe for it. A
        # NOIMAGE or otherwise flyc-less build of gfx950/gfx1201 will fail the
        # flyc entries of the sweep rather than skip them.
        BACKEND_ENUM = 'OpAttnFwdBackend'

    class attn_bwd(SdpaOpCommon, _bwd_kernel_dk_dv):
        # triton_split=0, triton_fuse=1, aiter=2, flyc=3.
        BACKEND_ENUM = 'OpAttnBwdBackend'

        OUTPUT_TNAMES = ["dk", "dv", "dq", "db"]

        def direct_call(self, direct_inputs, extargs):
            im, view, devm = direct_inputs
            import torch
            from aotriton.tune.gpu_utils import zero_devm
            # The aiter backend accumulates into dq_acc; clear before each
            # call. By name, not index: this was once a bare 2 that went wrong
            # when flyc was inserted ahead of it.
            if extargs.backend_name == 'aiter':
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
    """One dict per backend runnable for this entry's arch and dtype, in
    library index order.

    Carries a NAME because the consumer keeps only each dict's position, and
    position != backend index wherever the runnable set is not a contiguous
    prefix (gfx1201: triton=0, flyc=2).
    """
    kernel = get_impl(which_impl)
    return [{'backend_name': name} for name in kernel.available_backends(entry.dtype)]


def impl_desc(kernel, args) -> dict:
    """Both spellings: the name as identity, the index because the optune LUT
    stores it and export_best_results.py reads it back from here (impl_index
    is only a position)."""
    return {'backend_name': args.backend_name,
            'backend_index': args.backend_index}
