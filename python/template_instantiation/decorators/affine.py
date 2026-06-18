# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The `ati.affine.*` decorator surface (slim AITER-ASM affine kernels).

A SLIM affine kernel is a thin C++ shim: it does not generate a GPU kernel or own a
functional space. It translates an operator's params struct into a 3rd-party AITER API
call (via a COOKIE class) and packages pre-built `.co` files. Its ATI description is
metadata + a reference to the operator whose params struct it consumes, declared with
the stacked-@ form:

    @ati.kernel                                          # terminal (ends the stack)
    @ati.disable(when=_fwd_disabled)
    @ati.affine.aiter_asm(name='aiter_fmha_v3_fwd')      # innermost marker
    @ati.affine.shared_operator('op_attn_fwd')           # SHARED_IFACE (by op name)
    @ati.affine.arch(['gfx942', 'gfx950'])               # SUPPORTED_ARCH
    @ati.affine.limitations(Q=lambda d: 'fp16' in d or 'bf16' in d)  # CHOICE_FILTERS
    @ati.affine.structures(cookie='aiter::mha_fwd_args')            # COOKIE_CLASS
    @ati.affine.directories(co_dir='fmha_v3_fwd',
                            headers=['aotriton/_internal/flash/aiter.h'])
    def aiter_fmha_v3_fwd():
        pass

The bwd kernel additionally SUPPLIES the extra operand it contributes to the
operator's params struct (DQ_ACC) via `@ati.affine.supplies(ati.tensor(...))`; the
operator union (build_merged_struct_cfields) picks it up so the struct is a pure
union over all backends with no hand-injection.

These produce passive spec records; specs/finalize.py collects them into an AffineDecl
and the linker builds the ir.affine.AffineKernel.
"""


# --- spec records (callable -> accumulate onto the placeholder def) ----------

class _AffineSpec:
    """Base: a stacked-@ affine spec record. Calling it accumulates onto the def."""
    def __call__(self, placeholder):
        from ..describe import accumulate_spec
        return accumulate_spec(self, placeholder)


class AffineMarkerSpec(_AffineSpec):
    """@ati.affine.aiter_asm(name=...): the innermost marker that makes the def an
    affine-kernel description (the affine analogue of @ati.operator / @ati.source).
    `name` is the AOTriton shim name (defaults to the def name)."""
    __slots__ = ('name',)

    def __init__(self, name=None):
        self.name = name

    def __call__(self, placeholder):
        from ..describe import accumulate_spec
        if self.name is None:
            self.name = placeholder.__name__
        return accumulate_spec(self, placeholder)

    def __repr__(self):
        return f'AffineMarkerSpec({self.name!r})'


class SharedOperatorSpec(_AffineSpec):
    """@ati.affine.shared_operator('<op_name>'): the operator whose params struct
    this affine kernel consumes (SHARED_IFACE), referenced by NAME and resolved
    against the ops registry at finalize time (string avoids import cycles)."""
    __slots__ = ('op_name',)

    def __init__(self, op_name):
        assert isinstance(op_name, str) and op_name, \
            f'@ati.affine.shared_operator needs an operator name, got {op_name!r}'
        self.op_name = op_name

    def __repr__(self):
        return f'SharedOperatorSpec({self.op_name!r})'


class ArchSpec(_AffineSpec):
    """@ati.affine.arch([...]): SUPPORTED_ARCH."""
    __slots__ = ('arches',)

    def __init__(self, arches):
        self.arches = list(arches)


class LimitationsSpec(_AffineSpec):
    """@ati.affine.limitations(key=predicate, ...): CHOICE_FILTERS — per-argument
    predicates restricting which choices the ASM kernel supports."""
    __slots__ = ('filters',)

    def __init__(self, **filters):
        for k, fn in filters.items():
            assert callable(fn), \
                f'@ati.affine.limitations({k}=...) expects a callable, got {fn!r}'
        self.filters = dict(filters)


class StructuresSpec(_AffineSpec):
    """@ati.affine.structures(cookie='...'): the 3rd-party COOKIE_CLASS the shim
    fills in and hands to the AITER API."""
    __slots__ = ('cookie',)

    def __init__(self, cookie):
        self.cookie = cookie


class DirectoriesSpec(_AffineSpec):
    """@ati.affine.directories(co_dir='...', headers=[...]): CO_DIR (the kernel's
    name in the affine .co repository) + extra C++ headers the shim includes."""
    __slots__ = ('co_dir', 'headers')

    def __init__(self, co_dir, headers=None):
        self.co_dir = co_dir
        self.headers = list(headers) if headers else []


class SuppliesSpec(_AffineSpec):
    """@ati.affine.supplies(ati.tensor(...), ..., after=..., before=...): operands
    this affine backend contributes to the operator's params-struct union (e.g. the
    bwd DQ_ACC, which lives only on the affine backend). Each positional is an
    ati.tensor/ati.scalar spec. `after`/`before` name the neighbor operands the
    supplied ones must sit between in the merged struct order (e.g. DQ_ACC between
    DB and L), so union_params places them at the right index."""
    __slots__ = ('specs', 'after', 'before')

    def __init__(self, *specs, after=None, before=None):
        self.specs = list(specs)
        self.after = after
        self.before = before


# --- public decorator namespace (ati.affine.*) ------------------------------

def aiter_asm(name=None):
    return AffineMarkerSpec(name)


def shared_operator(op_name):
    return SharedOperatorSpec(op_name)


def arch(arches):
    return ArchSpec(arches)


def limitations(**filters):
    return LimitationsSpec(**filters)


def structures(cookie):
    return StructuresSpec(cookie)


def directories(co_dir, headers=None):
    return DirectoriesSpec(co_dir, headers)


def supplies(*specs, after=None, before=None):
    return SuppliesSpec(*specs, after=after, before=before)
