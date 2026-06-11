# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Builder: KernelSpec -> (axes, overrides) IR (executive plan Step 2.4;
agent-plans/ati+newbinds_rev1.md §3-§4).

This closes the loop from the decorator surface (Step 2.1-2.3) to the enumeration
core (Step 1.2-1.4): it groups the collected specs by choice variable, builds
Choice lists, computes each axis's signature anchor, resolves per-argument tensor
shape (rank + contiguous strides), and emits Axis + Override objects ready for
enumerate_functionals.

Stride parameters are hidden axes: each is its own single-choice u64:8 axis (or a
constexpr 1 for the contiguous stride). They live in the resolved arg table and
the params/signature machinery but are not godel digits.
"""

from .decorators import TensorSpec, ScalarSpec, ChoiceVar
from .introspect import ParamSpec
from .ir import Choice, Axis
from ..base.typed_choice import ELEMENTAL_TYPE_MAP


class AtiDescriptionError(Exception):
    """A diagnostic from the ATI front-end. Like the Triton compiler frontend it
    partially mimics, it names the kernel and parameter at fault and says how to
    fix it."""


def _is_ati_type_string(s: str) -> bool:
    """True if `s` is a type string Choice.parse accepts: a tensor pointer like
    '*fp16:16', or an elemental type like 'i32' / 'fp32' / 'u64'."""
    if not s:
        return False
    return (s.startswith('*') and s[1:] in ELEMENTAL_TYPE_MAP) or s in ELEMENTAL_TYPE_MAP

# Scalar type fallback (agent-plans/ati_rev1.md §3.2, fb Q5).
#
# An introspected annotation (introspect.py keeps it RAW) is one of:
#   * a string ('*u64', 'i32', ...) -> handed directly to Choice.parse, which
#     already parses the ATI type vocabulary (typed_choice.ELEMENTAL_TYPE_MAP).
#   * a triton type OBJECT (tl.float32, tl.int32, tl.uint64, ...) -> looked up in
#     the map below BY OBJECT IDENTITY. We never call str() on a triton dtype
#     (its str is 'int32'/'uint64', not our 'i32'/'u64'); keying by the public
#     type object is the stable contract.
#
# _build_annotation_type_map() is lazy so the module imports without triton (the
# generation venv has none). When triton is absent, only string annotations are
# resolvable here — which is the case the inspect fallback / tests exercise.
#
# A `tl.constexpr` annotation is intentionally unresolvable: such a parameter is
# a compile-time enumerated dimension and must be declared via
# @ati.scalar(..., options=...).
_ANNOTATION_TYPE = None        # built lazily; {tl.dtype object -> ATI type str}


def _annotation_type_map():
    global _ANNOTATION_TYPE
    if _ANNOTATION_TYPE is None:
        m = {}
        try:
            import triton.language as tl
            m = {
                tl.float32: 'fp32',
                tl.float16: 'fp16',
                tl.bfloat16: 'bf16',
                tl.int8: 'i8',
                tl.int16: 'i16',
                tl.int32: 'i32',
                tl.int64: 'i64',
                tl.uint8: 'u8',
                tl.uint16: 'u16',
                tl.uint32: 'u32',
                tl.uint64: 'u64',
                tl.int1: 'i1',
            }
        except Exception:
            pass
        _ANNOTATION_TYPE = m
    return _ANNOTATION_TYPE


_STRIDE_TYPE = 'u64:8'        # hidden stride dtype (matches the v2 stride_a8)
_STRIDE_ONE = 1              # contiguous stride -> constexpr 1


class BuiltKernel:
    """The lowered IR for one kernel: everything enumerate_functionals needs.

    Overrides (@ati.derives) split into two channels by target:
      * `overrides`      — target is a kernel ARGUMENT; applied in
        enumerate_functionals, lands in resolved[] (the compiled signature).
      * `perf_overrides` — target is a PERF-SCHEMA field; applied only in the perf
        layer (translate_*), never in resolved[]."""
    __slots__ = ('name', 'axes', 'overrides', 'perf_overrides', 'arguments',
                 'tune', 'disables')

    def __init__(self, name, axes, overrides, arguments, tune=None,
                 perf_overrides=None, disables=None):
        self.name = name
        self.axes = axes              # list[Axis] (incl. trivial + hidden strides)
        self.overrides = overrides    # functional-arg overrides
        self.perf_overrides = perf_overrides or []   # perf-field overrides
        self.arguments = arguments    # full signature order (list[str])
        self.tune = tune              # TuneSpec | None (perf schema, configs, ...)
        self.disables = disables or []  # list[DisableSpec] (functional-disable)

    def __repr__(self):
        nmulti = sum(1 for a in self.axes if not a.is_trivial)
        return (f'BuiltKernel({self.name!r}, {len(self.axes)} axes '
                f'({nmulti} multi-choice), {len(self.overrides)} overrides)')


def _scalar_type(spec: ScalarSpec, ann_by_name: dict, kernel_name: str):
    """Resolve a plain (non-options, non-shared) scalar's type, returning a value
    Choice.parse accepts: explicit type wins; else a string annotation (validated
    against the ATI type vocabulary); else a triton type OBJECT mapped by identity.

    Emits a kernel+parameter-named AtiDescriptionError on failure, in the spirit
    of the Triton compiler frontend this generator partially reimplements."""
    if spec.type_ is not None:
        return spec.type_
    ann = ann_by_name.get(spec.arg_name, ParamSpec.EMPTY)
    if isinstance(ann, str) and ann:
        if not _is_ati_type_string(ann):
            raise AtiDescriptionError(
                f"kernel {kernel_name!r}: parameter {spec.arg_name!r} is annotated "
                f"{ann!r}, which is not a recognized ATI type. Give it an explicit "
                f"type via ati.scalar({spec.arg_name!r}, '<type>'), or fix the "
                f"annotation (e.g. 'i32', 'fp32', '*u64').")
        return ann                      # Choice.parse / TypedChoice handle it
    mapped = _annotation_type_map().get(ann)
    if mapped is not None:
        return mapped
    if ann is ParamSpec.EMPTY:
        raise AtiDescriptionError(
            f"kernel {kernel_name!r}: parameter {spec.arg_name!r} has no type. "
            f"It is declared with ati.scalar({spec.arg_name!r}) but the kernel "
            f"gives it no annotation to infer from; specify it explicitly via "
            f"ati.scalar({spec.arg_name!r}, '<type>').")
    raise AtiDescriptionError(
        f"kernel {kernel_name!r}: parameter {spec.arg_name!r} is annotated with "
        f"{ann!r}, which ATI cannot map to a type. If it is a triton dtype, extend "
        f"_annotation_type_map; otherwise give an explicit type via "
        f"ati.scalar({spec.arg_name!r}, '<type>').")


def _choices_from(values) -> list:
    return [Choice.parse(v) for v in values]


def _resolve_signature_name(dtype, arg_names, param_index, kernel_name):
    """The argument that records an axis in persisted artifacts — the compact
    signature, the aks2/zip entry name, and the tuning-database row key.

    A MULTI-CHOICE shared ChoiceVar (ati.tensor_dtype/ati.choice_set bound by
    several specs) MUST declare signature_name explicitly when it spans several
    arguments — it is baked into stored artifacts, so it cannot be silently
    derived from spec order. A single-choice (radix 1) variable is trivial: it
    never appears in the compact signature / aks2 entry name, so signature_name is
    irrelevant and need not be given. A list-grouped spec
    (ati.scalar([names], ...)) or a single argument uses its first listed
    argument, which the author already wrote in the order they chose."""
    if isinstance(dtype, ChoiceVar) and dtype.signature_name is not None:
        assert dtype.signature_name in arg_names, (
            f"kernel {kernel_name!r}: choice variable signature_name "
            f"{dtype.signature_name!r} is not one of its arguments {arg_names}")
        return dtype.signature_name
    # A multi-choice shared ChoiceVar spanning multiple args needs an explicit
    # signature_name (single-choice variables are trivial and excluded from the
    # aks2 entry name, so they are exempt).
    if (isinstance(dtype, ChoiceVar) and len(arg_names) > 1
            and len(dtype.choices) > 1):
        raise AtiDescriptionError(
            f"kernel {kernel_name!r}: multi-choice variable {dtype.name!r} spans "
            f"{arg_names} and must declare an explicit signature_name (it is baked "
            f"into the aks2 entry name / DB row key); pass signature_name= to "
            f"ati.tensor_dtype/ati.choice_set")
    # List-grouped, single-argument, or single-choice: the first listed argument
    # represents it.
    return arg_names[0]


def build_kernel(kernel_spec) -> BuiltKernel:
    """Lower a KernelSpec (from describe()) into Axis + Override IR."""
    params = kernel_spec.params
    param_index = {p.name: i for i, p in enumerate(params)}
    ann_by_name = {p.name: p.annotation for p in params}
    name = getattr(kernel_spec.kernel, '__name__', 'kernel')

    axes = []

    # --- tensor axes (grouped by choice variable) ---
    # A shared ChoiceVar groups several tensors into one axis; a literal dtype is
    # an anonymous single-tensor axis.
    tensor_groups = {}     # var_name -> list[TensorSpec]
    for t in kernel_spec.tensors:
        tensor_groups.setdefault(t.var_name, []).append(t)

    for var_name, group in tensor_groups.items():
        first = group[0]
        if isinstance(first.dtype, ChoiceVar):
            choices = _choices_from(first.dtype.choices)
            # all members of a shared var must name the same variable -> same choices
        else:
            choices = _choices_from([first.dtype])     # literal: single choice
        arg_names = [a for t in group for a in t.arg_names]
        ranks = {}
        for t in group:
            r = t.resolve_rank(kernel_spec.param_names)
            for a in t.arg_names:
                ranks[a] = r
        contiguous = {}
        for t in group:
            cstride = t.resolve_contiguous(kernel_spec.param_names)
            if cstride is not None:
                contiguous[t.arg_name] = cstride
        anchor = min(param_index[a] for a in arg_names)
        signature_name = _resolve_signature_name(first.dtype, arg_names, param_index, name)
        axes.append(Axis(var_name, arg_names, choices, anchor,
                         ranks=ranks, contiguous=contiguous, kind='tensor',
                         signature_name=signature_name))
        # hidden stride axes for every matched stride of every tensor in the group
        for t in group:
            matched = t.match_strides(kernel_spec.param_names)
            for dim, sname in enumerate(matched):
                is_unit = (sname in contiguous.values())
                stype = _STRIDE_ONE if is_unit else _STRIDE_TYPE
                axes.append(Axis(sname, (sname,), _choices_from([stype]),
                                 param_index[sname],
                                 kind='stride_unit' if is_unit else 'stride',
                                 stride_of=(t.arg_name, dim)))

    # --- scalar axes (grouped by choice variable) ---
    scalar_groups = {}
    for s in kernel_spec.scalars:
        scalar_groups.setdefault(s.var_name, []).append(s)

    for var_name, group in scalar_groups.items():
        first = group[0]
        if first.dtype is not None:                 # shared ChoiceVar
            choices = _choices_from(first.dtype.choices)
        elif first.options is not None:             # enumerated (former feature)
            choices = _choices_from(first.options)
        else:                                       # plain runtime scalar
            choices = _choices_from([_scalar_type(first, ann_by_name, name)])
        arg_names = [a for s in group for a in s.arg_names]
        anchor = min(param_index[a] for a in arg_names)
        signature_name = _resolve_signature_name(first.dtype, arg_names, param_index, name)
        axes.append(Axis(var_name, arg_names, choices, anchor, kind='scalar',
                         signature_name=signature_name))

    axes.sort(key=lambda a: a.anchor)
    arguments = [p.name for p in params]
    # Split overrides by target: perf-schema fields -> perf channel, the rest ->
    # functional channel (applied in enumerate_functionals / resolved[]).
    perf_names = set()
    if kernel_spec.tune is not None and kernel_spec.tune.schema is not None:
        perf_names = {pp.name for pp in kernel_spec.tune.schema.params}
    func_ovs, perf_ovs = [], []
    for ov in kernel_spec.overrides:
        if any(t in perf_names for t in ov.targets):
            assert all(t in perf_names for t in ov.targets), (
                f'@ati.derives target mixes perf and non-perf names: {ov.targets}')
            perf_ovs.append(ov)
        else:
            func_ovs.append(ov)
    return BuiltKernel(name, axes, func_ovs, arguments,
                       tune=kernel_spec.tune, perf_overrides=perf_ovs,
                       disables=list(kernel_spec.disables))
