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
from .ir.typed_choice import ELEMENTAL_TYPE_MAP


class DescriptionError(Exception):
    """A diagnostic from the ATI front-end. Like the Triton compiler frontend it
    partially mimics, it names the kernel and parameter at fault and says how to
    fix it."""


def _is_ati_type_string(s: str) -> bool:
    """True if `s` is a type string Choice.parse accepts: a tensor pointer like
    '*fp16:16', an elemental type like 'i32' / 'fp32' / 'u64', or a lazy tensor
    'LazyTensor:*fp32:16'."""
    if not s:
        return False
    if s.startswith('LazyTensor:'):
        s = s[len('LazyTensor:'):]
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
                 'tune', 'disables', 'wiring')

    def __init__(self, name, axes, overrides, arguments, tune=None,
                 perf_overrides=None, disables=None, wiring=None):
        self.name = name
        self.axes = axes              # list[Axis] (incl. trivial + hidden strides)
        self.overrides = overrides    # functional-arg overrides
        self.perf_overrides = perf_overrides or []   # perf-field overrides
        self.arguments = arguments    # full signature order (list[str])
        self.tune = tune              # TuneSpec | None (perf schema, configs, ...)
        self.disables = disables or []  # list[DisableSpec] (functional-disable)
        # real->apparel arg wiring (rev0 §4.3), {real_arg: apparel_operand}, from
        # wires_to= on tensor/scalar specs. Empty when nothing is wired.
        self.wiring = wiring or {}

    def __repr__(self):
        nmulti = sum(1 for a in self.axes if not a.is_trivial)
        return (f'BuiltKernel({self.name!r}, {len(self.axes)} axes '
                f'({nmulti} multi-choice), {len(self.overrides)} overrides)')


def _scalar_type(spec: ScalarSpec, ann_by_name: dict, kernel_name: str):
    """Resolve a plain (non-options, non-shared) scalar's type, returning a value
    Choice.parse accepts: explicit type wins; else a string annotation (validated
    against the ATI type vocabulary); else a triton type OBJECT mapped by identity.

    Emits a kernel+parameter-named DescriptionError on failure, in the spirit
    of the Triton compiler frontend this generator partially reimplements."""
    if spec.type_ is not None:
        return spec.type_
    ann = ann_by_name.get(spec.arg_name, ParamSpec.EMPTY)
    if isinstance(ann, str) and ann:
        if not _is_ati_type_string(ann):
            raise DescriptionError(
                f"kernel {kernel_name!r}: parameter {spec.arg_name!r} is annotated "
                f"{ann!r}, which is not a recognized ATI type. Give it an explicit "
                f"type via ati.scalar({spec.arg_name!r}, '<type>'), or fix the "
                f"annotation (e.g. 'i32', 'fp32', '*u64').")
        return ann                      # Choice.parse / TypedChoice handle it
    mapped = _annotation_type_map().get(ann)
    if mapped is not None:
        return mapped
    if ann is ParamSpec.EMPTY:
        raise DescriptionError(
            f"kernel {kernel_name!r}: parameter {spec.arg_name!r} has no type. "
            f"It is declared with ati.scalar({spec.arg_name!r}) but the kernel "
            f"gives it no annotation to infer from; specify it explicitly via "
            f"ati.scalar({spec.arg_name!r}, '<type>').")
    raise DescriptionError(
        f"kernel {kernel_name!r}: parameter {spec.arg_name!r} is annotated with "
        f"{ann!r}, which ATI cannot map to a type. If it is a triton dtype, extend "
        f"_annotation_type_map; otherwise give an explicit type via "
        f"ati.scalar({spec.arg_name!r}, '<type>').")


def _choices_from(values) -> list:
    return [Choice.parse(v) for v in values]


def _resolve_signature_name(dtype, arg_names, param_index, kernel_name):
    """The LABEL recording an axis in persisted artifacts — the compact signature,
    the aks2/zip entry name, and the tuning-database row key.

    This is purely a label: it is NOT used to locate a kernel argument (the axis's
    representative real argument, `Axis.repr_arg`, does that). An author may set it
    to any string (e.g. 'dtype'); the generated code works regardless — only the
    persisted entry names / human-readable signatures change. When explicit it is
    used verbatim; it need not be one of the axis's arguments.

    A MULTI-CHOICE shared ChoiceVar spanning several arguments MUST declare it
    explicitly (the default — first argument — would be an arbitrary pick baked into
    stored artifacts). A single-choice (trivial) variable never appears in the
    compact signature, so it is exempt. A list-grouped or single-argument spec uses
    its first listed argument as the default label."""
    if isinstance(dtype, ChoiceVar) and dtype.signature_name is not None:
        return dtype.signature_name
    # A multi-choice shared ChoiceVar spanning multiple args needs an explicit
    # signature_name (single-choice variables are trivial and excluded from the
    # aks2 entry name, so they are exempt).
    if (isinstance(dtype, ChoiceVar) and len(arg_names) > 1
            and len(dtype.choices) > 1):
        raise DescriptionError(
            f"kernel {kernel_name!r}: multi-choice variable {dtype.name!r} spans "
            f"{arg_names} and must declare an explicit signature_name (it is baked "
            f"into the aks2 entry name / DB row key); pass signature_name= to "
            f"ati.tensor_dtype/ati.choice_set")
    # List-grouped, single-argument, or single-choice: the first listed argument
    # represents it.
    return arg_names[0]


def _resolve_named_dtypes(kernel_spec, name):
    """Resolve string `dtype` references on tensor/scalar specs against the kernel's
    named dtype-variable table (rev0 §4.2).

    A `dtype` slot holding a `str` is ambiguous: it is EITHER the name of an
    `@ati.tensor_dtype`/`ati.choice_set` variable declared on this kernel, OR a
    literal ATI type string (`'*fp16:16'`, `'i32'`). Resolution order:
      (1) a same-kernel dtype-var of that name  -> rewrite the slot to the ChoiceVar;
      (2) a literal ATI type string             -> leave as-is (handled downstream);
      (3) (Step 4) a dtype-var reachable through @ati.cite;
      (4) else DescriptionError.
    A `dtype` already given by object (ChoiceVar) or a literal already validated is
    left untouched. Mutates the spec dtype slots in place."""
    by_name = {dv.name: dv for dv in getattr(kernel_spec, 'dtype_vars', [])}
    # Scalars use `.dtype` only for a shared ChoiceVar; a literal type rides on
    # `.type_`. A scalar that named a dtype-var by string lands in `.type_`
    # (ScalarSpec puts any str second-arg there), so we must check both slots.
    for t in kernel_spec.tensors:
        d = t.dtype
        if isinstance(d, str) and d in by_name:
            t.dtype = by_name[d]
        elif isinstance(d, str) and not _is_ati_type_string(d):
            raise DescriptionError(
                f"kernel {name!r}: tensor {t.arg_name!r} names dtype variable "
                f"{d!r}, which is neither an @ati.tensor_dtype on this kernel "
                f"{sorted(by_name)} nor a literal ATI type. Declare it with "
                f"@ati.tensor_dtype({d!r}, dtype=[...]) or fix the type string.")
    for s in kernel_spec.scalars:
        d = s.type_
        if isinstance(d, str) and d in by_name:
            s.type_ = None
            s.dtype = by_name[d]
        elif isinstance(d, str) and not _is_ati_type_string(d):
            raise DescriptionError(
                f"kernel {name!r}: scalar {s.arg_name!r} names dtype variable "
                f"{d!r}, which is neither an @ati.tensor_dtype on this kernel "
                f"{sorted(by_name)} nor a literal ATI type. Declare it with "
                f"@ati.tensor_dtype({d!r}, dtype=[...]) or fix the type string.")


def build_kernel(kernel_spec) -> BuiltKernel:
    """Lower a KernelSpec (from describe()) into Axis + Override IR."""
    params = kernel_spec.params
    param_index = {p.name: i for i, p in enumerate(params)}
    ann_by_name = {p.name: p.annotation for p in params}
    name = getattr(kernel_spec.kernel, '__name__', 'kernel')

    _resolve_named_dtypes(kernel_spec, name)

    axes = []

    # --- tensor axes (grouped by choice variable) ---
    # A shared ChoiceVar groups several tensors into one axis; a literal dtype is
    # an anonymous single-tensor axis.
    tensor_groups = {}     # var_name -> list[TensorSpec]
    for t in kernel_spec.tensors:
        tensor_groups.setdefault(t.var_name, []).append(t)

    nonunit_strides = {}     # tensor arg -> [its non-unit (real, hideable) strides]
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
                if not is_unit:
                    nonunit_strides.setdefault(t.arg_name, []).append(sname)
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
    # Collect real->apparel wiring (rev0 §4.3) from wires_to= on tensor/scalar
    # specs. The IR stays keyed on REAL names; the wiring is applied only on the
    # outward codegen surface (Step 3).
    wiring = {}
    for spec in (*kernel_spec.tensors, *kernel_spec.scalars):
        w = getattr(spec, 'wires_to', None)
        if w is not None:
            wiring[spec.arg_name] = w
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

    # A functional override that constexpr-zeroes a TENSOR implicitly cascades to
    # that tensor's non-unit strides (a zeroed tensor needs no live strides). This
    # keeps the description terse: `ati.derives('B', to=0, when=...)` need not also
    # list stride_b*. Synthesize the stride overrides here, under the SAME
    # predicate. (Unit/contiguous strides are constexpr 1 already and excluded.)
    from .ir import Override as _Override
    extra_stride_ovs = []
    for ov in func_ovs:
        if ov.value != 0:
            continue
        for t in ov.targets:
            for sname in nonunit_strides.get(t, ()):
                extra_stride_ovs.append(_Override([sname], ov.predicate, 0))
    func_ovs += extra_stride_ovs

    return BuiltKernel(name, axes, func_ovs, arguments,
                       tune=kernel_spec.tune, perf_overrides=perf_ovs,
                       disables=list(kernel_spec.disables), wiring=wiring)
