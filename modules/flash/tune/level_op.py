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


# Which arches each backend can actually run on, keyed by the name
# `@ati.backend` declared for it in modules/flash/aot/__init__.py.
#
# A backend absent from this table runs everywhere; only the restricted ones
# are listed. The restrictions are not this module's to invent -- each mirrors
# the description layer, and the citation is the point, because these will move:
#
#   aiter : @ati.affine.arch(['gfx942', 'gfx950']) in modules/flash/aot/
#           aiter_fwd.py:31 and aiter_bwd.py:33.
#   flyc  : the `f.arch not in <ladder>` disable predicates in
#           modules/flash/aot/flyc_attn_fwd.py:120-140, flyc_bwd_dkdv.py:45-48
#           and flyc_bwd_dq.py:47-50, all keyed on {gfx950, gfx1201}.
#
# This is a duplicate of knowledge that lives elsewhere, and duplicating it is
# a compromise rather than a design: the authoritative predicates are evaluated
# at CODEGEN time against a Functional, while the tuner needs the answer at RUN
# time from a GPU and an installed library. There is no probe for "does this
# library carry images for backend B on this arch" -- see the note in attn_fwd
# below.
_BACKEND_ARCHS = {
    'aiter': frozenset({'gfx942', 'gfx950'}),
    'flyc': frozenset({'gfx950', 'gfx1201'}),
}


def _backend_table(enum_struct_name: str):
    """(names in library-index order, {name: index}) read from pyaotriton.

    Both come from `<Struct>.by_index`, the dict the binding builds out of the
    generator's X-macro (modules/flash/bindings/v3.cc:118-121), whose values
    are verbatim the strings `@ati.backend` declared. So this module never
    spells a backend index, and never has to reproduce the generator's
    kCamelCase constant-naming rule to find one: the declared name IS the key.
    A backend added, removed or reordered in the description shows up here
    without an edit.
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

        def available_backends(self) -> list[str]:
            """Declared names of the backends this arch can run, in library
            index order. This list defines the op-level variant space: its
            LENGTH is how many variants get benchmarked and its ORDER fixes
            which impl_index means which backend."""
            names, _ = _backend_table(self.BACKEND_ENUM)
            arch = _gpu_arch()
            return [n for n in names
                    if arch in _BACKEND_ARCHS.get(n, frozenset({arch}))]

        def backend_index_of(self, name: str) -> int:
            _, index_of = _backend_table(self.BACKEND_ENUM)
            return index_of[name]

        def create_extargs(self, *, which_impl=None, probe=False):
            # which_impl.impl_index is a POSITION in enumerate_variants()'s
            # list, not a backend index, and the two are only equal while the
            # available backends happen to form a contiguous prefix of the
            # library's list. They do not on gfx1201, which has flyc (index 2
            # for fwd) but no aiter (index 1): position 1 there is flyc, and
            # feeding 1 to force_backend_index would run aiter -- a backend
            # with no images on that arch. Resolve through the name instead,
            # against the same ordered list enumerate_variants() published.
            names = self.available_backends()
            position = which_impl.impl_index if which_impl is not None else 0
            name = names[position]
            return self.EXT_CLASS.for_op_backend(self.backend_index_of(name), name)

    class attn_fwd(SdpaOpCommon, _attn_fwd):
        # The vocabulary is pyaotriton.v3.flash.OpAttnFwdBackend, generated
        # from the same list that assigns BackendEnum -- read it rather than
        # any comment here. Today: triton=0, aiter=1, flyc=2.
        #
        # The swept set is neither `Max` nor a count. Max is how many backends
        # the LIBRARY was generated with, and the enum is identical on every
        # arch, so sweeping Max would probe backends that cannot run here. A
        # count cannot express the set either: on gfx1201 the runnable
        # backends are triton and flyc, indices 0 and 2, which is not
        # `range(n)` of anything. Hence `available_backends()` returning names.
        #
        # ONE CAVEAT SURVIVES from when flyc was excluded outright: whether a
        # library carries flyc IMAGES is, strictly, a build-time property
        # rather than an arch property, and the tuner still has no probe for
        # it. Enabling flyc here therefore assumes a library built normally for
        # gfx950/gfx1201 -- which is now the only kind that configures at all,
        # since CMakeLists.txt's flydsl-llvm tripwire fails any image-mode
        # build without a FlyDSL wheel. A NOIMAGE or otherwise flyc-less build
        # of those two arches will fail the flyc entries of the sweep rather
        # than skip them.
        BACKEND_ENUM = 'OpAttnFwdBackend'

    class attn_bwd(SdpaOpCommon, _bwd_kernel_dk_dv):
        # See attn_fwd above. Today: triton_split=0, triton_fuse=1, aiter=2,
        # flyc=3.
        BACKEND_ENUM = 'OpAttnBwdBackend'

        OUTPUT_TNAMES = ["dk", "dv", "dq", "db"]

        def direct_call(self, direct_inputs, extargs):
            im, view, devm = direct_inputs
            import torch
            from aotriton.tune.gpu_utils import zero_devm
            # The aiter backend accumulates into dq_acc; clear before each
            # call. Compared by declared name rather than by index: the index
            # was already once a bare 2 that silently became wrong when flyc
            # was inserted, and OpAttnBwdBackend.kAiter only fixed half of
            # that -- it is still an integer whose meaning is positional. The
            # name is the backend's identity and cannot be shifted by a
            # neighbour.
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
    """One dict per runnable backend, in library index order.

    Carries `backend_name`, not `backend_index`. The consumer
    (localq/handlers.py `_build_fanout`) reads only each dict's POSITION, which
    becomes the impl_index on the wire and in the database -- so whatever is
    inside has to be something that can be resolved back to a backend without
    assuming position and index agree. On gfx1201 they do not: the runnable set
    is {triton=0, flyc=2}, so position 1 is flyc. A name survives that; an
    index copied from a position does not.
    """
    kernel = get_impl(which_impl)
    return [{'backend_name': name} for name in kernel.available_backends()]


def impl_desc(kernel, args) -> dict:
    """Both spellings, deliberately.

    `backend_name` is the identity -- stable across builds, and the thing worth
    reading in a report. `backend_index` is what was actually written to
    force_backend_index for this run, and it is what the optune LUT stores, so
    it has to be recoverable from the record rather than re-derived later from
    an impl_index that is only a position (see
    pq/export_best_results.py's op path, which reads it from here).
    """
    return {'backend_name': args.backend_name,
            'backend_index': args.backend_index}
