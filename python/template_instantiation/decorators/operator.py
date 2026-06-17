# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Operator description surface: `ati.operator` + `ati.backend`.

An operator dispatches among interchangeable BACKENDS (a triton metro, a fused triton
kernel, an affine asm kernel, ...). It is described declaratively with the stacked-@
form, mirroring kernels but finalizing into an Operator rather than a KernelSpec:

    @ati.kernel                                       # ends the stack (finalizes)
    @ati.backend(1, fwd_aiter, 'aiter')               # explicit dispatch index
    @ati.backend(0, metro_fwd, 'triton')
    @ati.tune.binning(Max_seqlen_q=ati.tune.binning.le,   # -> operator OPTUNE_KEYS
                      Max_seqlen_k=ati.tune.binning.le)
    @ati.operator(call_options_name='attn_options')   # innermost, next to def
    def op_attn_fwd():
        pass

Operator tuning is EXPLICIT: an operator never inherits a kernel's @ati.tune.*
(kernel-level perf/fallback would fight the operator's backend-selection tuning).
@ati.tune.binning on an operator means "which backend" (OPTUNE_KEYS); a
@ati.tune.fallback means the operator's own PARTIALLY_TUNED (default {});
@ati.tune.configs is accepted but ignored with a warning (a kernel-only concept).

TODO: rename the terminal @ati.kernel to a generic @ati.start facade that finalizes
any stacked description (kernel OR operator); @ati.operator/@ati.source mark the
start, @ati.start ends the stack.
"""


class BackendSpec:
    """One @ati.backend(index, ref, name): a dispatchable operator backend.

    `index` is the explicit dispatch / enum / tuning-database order (load-bearing:
    the op tuning rows store this integer). `ref` is the in-file object used to
    IDENTIFY the backend dependency (a metro function carrying `__ati_metro__`, a
    triton kdesc, or an affine kernel) — the linker keys its symbol table on the
    target's declared NAME (`ref_name`), NOT on the object identity. `name` is the
    backend NAME used to form the C++ enum (e.g. 'triton_split' -> kMetro_TritonSplit).

    `obj` is retained for the interim eager build path (it passes the BUILT backend
    object directly); once the codegen linker owns the build (exec0 Step 3) the
    reference is resolved purely by `ref_name` against the per-family symbol table."""

    __slots__ = ('index', 'obj', 'name', 'ref_name')

    def __init__(self, index, obj, name):
        assert isinstance(index, int), \
            f'@ati.backend index must be an int, got {index!r}'
        assert isinstance(name, str) and name, \
            f'@ati.backend name must be a non-empty string, got {name!r}'
        self.index = index
        self.obj = obj
        self.name = name
        # Name-based reference the linker will key on: the target's declared NAME when
        # the ref already carries one (a built metro/kdesc/affine in the eager path),
        # else the backend enum name as a stand-in (passive metro functions have no
        # NAME until the linker builds them under this same `name`).
        self.ref_name = getattr(obj, 'NAME', None) or name

    def __call__(self, kernel):
        from ..describe import accumulate_spec
        return accumulate_spec(self, kernel)

    def __repr__(self):
        return f'BackendSpec({self.index}, {self.name!r}, ref={self.ref_name!r})'


def backend(index, obj, name):
    """Declare one operator backend at an explicit dispatch index (see BackendSpec)."""
    return BackendSpec(index, obj, name)


class OperatorSpec:
    """The @ati.operator marker (innermost, next to the def): records the operator's
    declared parameters that are not themselves decorators.

    `default_kdesc` (the functional-axes owner) and `struct_cfields` (the params
    struct) are NO LONGER declared here — the linker DERIVES both from the backend
    tree (the union over backends for the struct; the default backend's first tunable
    sub-kernel for the axes). Family is inferred from the modules/<family>/aot path,
    so it is not declared either."""

    __slots__ = ('name', 'call_options_name')

    def __init__(self, name=None, *, call_options_name):
        self.name = name
        self.call_options_name = call_options_name

    def __call__(self, placeholder):
        # Innermost decorator: like @ati.source it runs first and seeds the pending
        # list with itself, so the specs above accumulate onto the same object.
        from ..describe import accumulate_spec
        if self.name is None:
            self.name = placeholder.__name__
        return accumulate_spec(self, placeholder)

    def __repr__(self):
        return f'OperatorSpec({self.name!r})'


def operator(name=None, *, call_options_name):
    """Innermost stacked-@ marker declaring the def to be an operator description
    (the operator analogue of @ati.source). The @ati.backend / @ati.tune.* specs
    ABOVE it accumulate onto the operator; @ati.kernel finalizes the stack into a
    PASSIVE @ati.operator def (fn.__ati_operator__) the linker builds."""
    return OperatorSpec(name, call_options_name=call_options_name)
