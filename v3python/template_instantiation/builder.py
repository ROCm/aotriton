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
from .ir import Choice, Axis

# Annotation-string -> ATI type literal, for scalars with no explicit type
# (agent-plans/ati_rev1.md §3.2, fb Q5). Mirrors the project's own markers in
# tritonsrc (constexpr_or_i32 / constexpr_or_f32) plus tl.* primitives.
_ANNOTATION_TYPE = {
    'constexpr_or_i32': 'i32',
    'constexpr_or_f32': 'fp32',
    'tl.float32': 'fp32',
    'tl.int1': 'i1',
    'tl.uint64': 'u64',
    '*u64': '*u64',
}

_STRIDE_TYPE = 'u64:8'        # hidden stride dtype (matches the v2 stride_a8)
_STRIDE_ONE = 1              # contiguous stride -> constexpr 1


class BuiltKernel:
    """The lowered IR for one kernel: everything enumerate_functionals needs."""
    __slots__ = ('name', 'axes', 'overrides', 'arguments')

    def __init__(self, name, axes, overrides, arguments):
        self.name = name
        self.axes = axes              # list[Axis] (incl. trivial + hidden strides)
        self.overrides = overrides    # list[Override]
        self.arguments = arguments    # full signature order (list[str])

    def __repr__(self):
        nmulti = sum(1 for a in self.axes if not a.is_trivial)
        return (f'BuiltKernel({self.name!r}, {len(self.axes)} axes '
                f'({nmulti} multi-choice), {len(self.overrides)} overrides)')


def _scalar_type(spec: ScalarSpec, ann_by_name: dict) -> str | None:
    """Resolve a plain (non-options, non-shared) scalar's type: explicit wins,
    else fall back to the Triton annotation."""
    if spec.type_ is not None:
        return spec.type_
    ann = ann_by_name.get(spec.arg_name, '')
    if ann in _ANNOTATION_TYPE:
        return _ANNOTATION_TYPE[ann]
    assert False, (
        f'@ati.scalar({spec.arg_name!r}) has no explicit type and its annotation '
        f'{ann!r} is not a known fallback; add an explicit type or extend '
        f'_ANNOTATION_TYPE')


def _choices_from(values) -> list:
    return [Choice.parse(v) for v in values]


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
        arg_names = [t.arg_name for t in group]
        ranks = {t.arg_name: t.resolve_rank(kernel_spec.param_names) for t in group}
        contiguous = {}
        for t in group:
            cstride = t.resolve_contiguous(kernel_spec.param_names)
            if cstride is not None:
                contiguous[t.arg_name] = cstride
        anchor = min(param_index[a] for a in arg_names)
        axes.append(Axis(var_name, arg_names, choices, anchor,
                         ranks=ranks, contiguous=contiguous))
        # hidden stride axes for every matched stride of every tensor in the group
        for t in group:
            for sname in t.match_strides(kernel_spec.param_names):
                is_unit = (sname in contiguous.values())
                stype = _STRIDE_ONE if is_unit else _STRIDE_TYPE
                axes.append(Axis(sname, (sname,), _choices_from([stype]),
                                 param_index[sname]))

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
            choices = _choices_from([_scalar_type(first, ann_by_name)])
        arg_names = [s.arg_name for s in group]
        anchor = min(param_index[a] for a in arg_names)
        axes.append(Axis(var_name, arg_names, choices, anchor))

    axes.sort(key=lambda a: a.anchor)
    arguments = [p.name for p in params]
    return BuiltKernel(name, axes, list(kernel_spec.overrides), arguments)
