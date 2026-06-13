# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI affine-kernel surface (`ati.affine.*`).

A SLIM affine kernel (the AITER ASM backend) is a thin C++ shim: it does not
generate a GPU kernel or own a functional space. It translates an operator's params
struct into a 3rd-party AITER API call (via a COOKIE class) and packages pre-built
`.co` files. So its ATI description is just metadata + a reference to the operator
whose params struct it consumes, declared with the stacked-@ form:

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
"""

from pathlib import Path

from aotriton.base.cfield import cfield
from aotriton.affine import SlimAffineKernelDescription


# --- spec records (callable -> accumulate onto the placeholder def) ----------

class _AffineSpec:
    """Base: a stacked-@ affine spec record. Calling it accumulates onto the def."""
    def __call__(self, placeholder):
        from .describe import accumulate_spec
        return accumulate_spec(self, placeholder)


class AffineMarkerSpec(_AffineSpec):
    """@ati.affine.aiter_asm(name=...): the innermost marker that makes the def an
    affine-kernel description (the affine analogue of @ati.operator / @ati.source).
    `name` is the AOTriton shim name (defaults to the def name)."""
    __slots__ = ('name',)

    def __init__(self, name=None):
        self.name = name

    def __call__(self, placeholder):
        from .describe import accumulate_spec
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


# --- the built object -------------------------------------------------------

# DQ_ACC-style supplied operands declare a LazyTensor:*fp32:16 rank-N tensor; map
# the spec to the C struct field type the slim shim's params struct expects.
def _supplied_cfield(spec, index):
    """Build a struct cfield for one @ati.affine.supplies operand. Supports the
    LazyTensor tensors the affine backend adds (e.g. DQ_ACC)."""
    from .decorators import TensorSpec
    assert isinstance(spec, TensorSpec), \
        f'@ati.affine.supplies currently supports ati.tensor specs only, got {spec!r}'
    aname = spec.arg_names[0]
    dtype = spec.dtype
    rank = spec.rank if spec.rank is not None else 4
    if isinstance(dtype, str) and dtype.startswith('LazyTensor:'):
        ctype = f'LazyTensorInternal<{rank}>*'
    else:
        ctype = f'const TensorView<{rank}>*'
    return cfield(ctype=ctype, aname=aname, index=index, nbits=0)


class AtiAffineKernel(SlimAffineKernelDescription):
    """A slim affine kernel built from the @ati.affine.* stacked form. Exposes the
    surface the SlimAffineGenerator reads (NAME / FAMILY / CO_DIR / COOKIE_CLASS /
    SUPPORTED_ARCH / AFFINE_KERNEL_ROOT / co_gen / SHARED_IFACE / is_functional_disabled)
    plus `supplied_operands` (the operand specs it contributes to the operator union).

    SHARED_IFACE is filled in by infer_shared_iface (operator -> backend) like the
    triton kernels; the @ati.affine.shared_operator name is kept for reference.
    Interface's functional machinery is intentionally bypassed (a slim affine kernel
    has a vendored dispatcher: gen_functionals is empty, func_cfields unused)."""

    FAMILY = 'flash'
    AFFINE_KERNEL_ROOT = Path('aiter/hsa')
    SHARED_IFACE = None      # set by infer_shared_iface

    def __init__(self, *, name, family, co_dir, cookie, headers, supported_arch,
                 choice_filters, shared_operator_name, supplied_specs, disable,
                 supplies_after=None, supplies_before=None):
        # NOTE: deliberately do NOT call Interface.__init__ — that builds a functional
        # space from SHARED_IFACE's TYPE/FEAT_CHOICES, which a slim affine kernel does
        # not need (vendored dispatcher; slim codegen reads only the metadata below).
        self.NAME = name
        self.FAMILY = family
        self.CO_DIR = co_dir
        self.COOKIE_CLASS = cookie
        self.HEADER_EXTRA_INCLUDES = list(headers)
        self.SUPPORTED_ARCH = list(supported_arch)
        self.CHOICE_FILTERS = dict(choice_filters)
        self.shared_operator_name = shared_operator_name
        self._supplied_specs = list(supplied_specs)
        self._supplies_after = supplies_after
        self._supplies_before = supplies_before
        self._disable = disable      # DisableSpec | None
        # Precompute the cfields this backend supplies to the operator union (DQ_ACC).
        self._supplied_cfields = [_supplied_cfield(s, i)
                                  for i, s in enumerate(self._supplied_specs)]

    # NAME is a per-INSTANCE attribute (one AtiAffineKernel class, many kernels),
    # so the Interface classmethod _class_name_base (which reads the CLASS NAME)
    # can't see it — derive the name base from the instance here.
    @property
    def class_name_base(self):
        return ''.join(x.capitalize() for x in self.NAME.lower().split('_'))

    @property
    def enum_name(self):
        return f'kSlimAffine_{self.class_name_base}'

    @property
    def param_class_name(self):
        # The params struct is the shared operator's. Read it directly off
        # SHARED_IFACE (the AtiOperator exposes param_class_name), rather than
        # reconstructing via the legacy classmethod _class_name_base.
        return self.SHARED_IFACE.param_class_name

    @property
    def context_class_name(self):
        return self.class_name_base + 'Context'

    def co_dir_path(self, build_dir: Path, functional):
        return build_dir / self.AFFINE_KERNEL_ROOT / functional.arch / self.CO_DIR

    # --- operator-union contribution -------------------------------------
    #
    # The operator's params struct (build_merged_struct_cfields) is the union over
    # all backends' func_cfields. A slim affine kernel contributes only its SUPPLIED
    # operands (DQ_ACC); `func_cfields` exposes their cfields, and `union_order`
    # gives the ordering chain (after-anchor, supplied..., before-anchor) so the
    # merge places them at the right struct index.

    @property
    def func_cfields(self):
        return list(self._supplied_cfields)

    @property
    def union_order(self):
        names = [cf.aname for cf in self._supplied_cfields]
        if not names:
            return []
        chain = []
        if self._supplies_after is not None:
            chain.append(self._supplies_after)
        chain += names
        if self._supplies_before is not None:
            chain.append(self._supplies_before)
        return chain

    @property
    def supplied_operands(self):
        """The TensorSpecs this affine backend contributes to the operator union."""
        return list(self._supplied_specs)

    def is_functional_disabled(self, functional):
        if self._disable is None:
            return False
        return self._disable.when(functional)
