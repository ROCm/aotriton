# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
describe() + the stacked-@ sugar finalizer

This is the GLUE that turns the loose @ati.* spec-records into the passive Stage-2
"object files" (specs/kernel.py KernelSpec, specs/affine.py AffineDecl,
specs/operator.py OperatorDecl):

`ati.describe(kernel, *specs)` is the canonical primitive: it introspects the
kernel's parameter list, validates that the specs claim every argument exactly
once, and attaches a KernelSpec sidecar (`kernel.__ati_node__`). The stacked-@ form
lowers to the same path — each `@ati.tensor(...)` returns a spec and the eventual
`@triton.jit`-adjacent collection is replayed through describe() — so the two
authoring modes share one implementation and produce an identical KernelSpec.

This step stores the collected, validated specs. Lowering them to the
Axis/Override IR (enumerate_functionals input) is Step 2.4 (builder.py).
"""

from ..decorators import TensorSpec, ScalarSpec, ChoiceVar
from ..ir import Override
from ..introspect import kernel_params, kernel_annotations
from .kernel import KernelSpec
from .affine import AffineDecl, collect_affine_decl
from .operator import OperatorDecl, collect_operator_decl


def _build_tune_spec(tune_records):
    """Fold the collected tune spec-records into one TuneSpec, or None if there
    are no tuning decorators on this kernel."""
    from .tune import (
        TuneSpec, PerfSchema, ConfigsSpec, BinningSpec, FallbackSpec,
    )
    if not tune_records:
        return None
    ts = TuneSpec()
    for r in tune_records:
        if isinstance(r, PerfSchema):
            assert ts.schema is None, 'duplicate ati.tune.schema on one kernel'
            ts.schema = r.struct        # store the synthesized perf struct CLASS
        elif isinstance(r, ConfigsSpec):
            assert ts.configs is None, 'duplicate ati.tune.configs on one kernel'
            ts.configs = r.generator
        elif isinstance(r, BinningSpec):
            ts.binning.update(r.keys)
        elif isinstance(r, FallbackSpec):
            ts.fallback.update(r.values)
        else:
            raise AssertionError(f'unrecognized tune spec {r!r}')
    return ts


def _partition(specs):
    from .tune import PerfSchema, ConfigsSpec, BinningSpec, FallbackSpec
    from ..decorators import DisableSpec, CiteSpec
    tune_types = (PerfSchema, ConfigsSpec, BinningSpec, FallbackSpec)
    tensors, scalars, overrides, tune_records, disables, dtype_vars, cites, others = \
        [], [], [], [], [], [], [], []
    for s in specs:
        if isinstance(s, TensorSpec):
            tensors.append(s)
        elif isinstance(s, ScalarSpec):
            scalars.append(s)
        elif isinstance(s, ChoiceVar):
            dtype_vars.append(s)
        elif isinstance(s, CiteSpec):
            cites.append(s)
        elif isinstance(s, Override):
            overrides.append(s)
        elif isinstance(s, DisableSpec):
            disables.append(s)
        elif isinstance(s, tune_types):
            tune_records.append(s)
        else:
            others.append(s)
    return (tensors, scalars, overrides, tune_records, disables, dtype_vars,
            cites, others)


def _validate_completeness(params, tensors, scalars, tune_records, has_cite=False):
    """Every introspected parameter must be claimed exactly once — by a tensor
    (itself or one of its stride globs), a scalar, or a perf-schema field.
    Reports orphans and double-claims (the §9.1a completeness check, kernel-scoped).

    When the kernel has an @ati.cite (has_cite), an UNCLAIMED parameter is NOT an
    orphan: the cite resolver (operator/infer.py) fills it from the cited metro at
    build time. Unknown-parameter and double-claim errors still apply."""
    from .tune import PerfSchema

    param_names = [p.name for p in params]
    name_set = set(param_names)
    claims = {}      # arg_name -> list of claimant descriptions

    def claim(arg_name, who):
        claims.setdefault(arg_name, []).append(who)

    for t in tensors:
        for a in t.arg_names:
            claim(a, f'tensor({a})')
        for sname in t.match_strides(param_names):
            claim(sname, f'tensor({t.arg_name}).strides')
    for s in scalars:
        for a in s.arg_names:
            claim(a, f'scalar({a})')
    for r in tune_records:
        if isinstance(r, PerfSchema):
            for pname in r.struct.param_names():
                claim(pname, f'tune.schema({pname})')

    errors = []
    # claims referencing names not in the signature
    for arg_name, who in claims.items():
        if arg_name not in name_set:
            errors.append(f'{who[0]} references unknown parameter {arg_name!r}')
    # double-claims
    for arg_name, who in claims.items():
        if len(who) > 1:
            errors.append(f'parameter {arg_name!r} claimed by multiple specs: {who}')
    # orphans (unclaimed params) — only an error WITHOUT a cite; with a cite the
    # resolver supplies them and reports anything still unresolved.
    if not has_cite:
        for name in param_names:
            if name not in claims:
                errors.append(f'parameter {name!r} is not claimed by any '
                              f'@ati.tensor/@ati.scalar/tune.schema (or stride glob)')
    return errors


def _claimed_arg_names(tensors, scalars):
    """Every argument name claimed by an explicit @ati.tensor/@ati.scalar spec."""
    claimed = set()
    for t in tensors:
        claimed.update(t.arg_names)
    for s in scalars:
        claimed.update(s.arg_names)
    return claimed


def _annotation_specs(kernel, tensors, scalars):
    """Synthesize specs from STRING-annotated placeholder parameters.

    Tensor pointer annotations (starting with '*' or 'LazyTensor:') produce a
    rank-0 TensorSpec with no strides — suitable for strideless pointer arguments
    like philox seeds, LSE tensors, or encoded-softmax buffers. All other string
    annotations produce a ScalarSpec.

    An annotated parameter that an explicit @ati.* spec already claims is an error.
    """
    from ..builder import DescriptionError, _is_ati_type_string
    claimed = _claimed_arg_names(tensors, scalars)
    new_tensors, new_scalars = [], []
    for arg, type_str in kernel_annotations(kernel).items():
        if arg in claimed:
            raise DescriptionError(
                f"kernel {getattr(kernel, '__name__', kernel)!r}: parameter {arg!r} "
                f"has both a type annotation ({type_str!r} on the def) and an "
                f"explicit @ati.tensor/@ati.scalar; declare it only once.")
        if type_str.startswith('*') or type_str.startswith('LazyTensor:'):
            # Tensor pointer type: rank from '[N]' suffix, default 0 (strideless).
            from ..ir.typed_choice import _parse_rank_suffix
            _, rank = _parse_rank_suffix(type_str)
            new_tensors.append(TensorSpec(arg, type_str, rank=rank if rank is not None else 0))
        else:
            new_scalars.append(ScalarSpec(arg, type_str))
    return new_tensors, new_scalars


def describe(kernel, *specs, _validate=True):
    """Attach an ATI KernelSpec to a kernel. Canonical for both authoring modes."""
    params = kernel_params(kernel)
    tensors, scalars, overrides, tune_records, disables, dtype_vars, cites, others = \
        _partition(specs)
    assert not others, f'describe() got unrecognized specs: {others}'
    # Placeholder-def string annotations become TensorSpec (pointer types: '*...' /
    # 'LazyTensor:...') or ScalarSpec (all others). Appended so completeness sees
    # them like any other tensor/scalar.
    ann_tensors, ann_scalars = _annotation_specs(kernel, tensors, scalars)
    tensors = tensors + ann_tensors
    scalars = scalars + ann_scalars
    if _validate:
        errors = _validate_completeness(params, tensors, scalars, tune_records,
                                        has_cite=bool(cites))
        assert not errors, (
            f'ATI describe({getattr(kernel, "__name__", kernel)!r}) validation '
            f'failed:\n  ' + '\n  '.join(errors))
    spec = KernelSpec(kernel, params, tensors, scalars, overrides,
                      tune=_build_tune_spec(tune_records), disables=disables,
                      dtype_vars=dtype_vars, cites=cites)
    kernel.__ati_node__ = spec
    return kernel


# --- stacked-@ sugar (Mode A) ---------------------------------------------
#
# In Mode A the decorators sit directly above @triton.jit:
#
#     @ati.tensor('Q', T_io, strides='stride_q?')
#     @ati.scalar('CAUSAL_TYPE', options=[0, 3])
#     @ati.overrides('B', to=0, when=ati.eq('BIAS_TYPE', 0))
#     @triton.jit
#     def attn_fwd(...): ...
#
# `ati.tensor('Q', T)` must serve as *both* a spec (when passed to describe()) and
# a decorator (when written as `@`). It cannot tell which at call time, so the
# resolution is: the spec objects are themselves callable — `spec(kernel)`
# accumulates the spec onto the kernel and returns it. describe() never calls a
# spec, so it sees plain specs; the @ form calls each spec once.
#
# A terminal decorator `@ati.start` marks the end of the stack and triggers
# finalization explicitly (no lazy-on-access guessing):
#
#     @ati.start                                   # applied LAST -> finalizes
#     @ati.tensor('Q', T_io, strides='stride_q?')
#     @ati.scalar('CAUSAL_TYPE', options=[0, 3])
#     @triton.jit
#     def attn_fwd(...): ...
#
# Python applies decorators bottom-up, so by the time @ati.start runs, every
# spec is accumulated; the list is in bottom-up order, which finalize restores to
# source order before replaying through describe().

_PENDING = '__ati_pending__'


def accumulate_spec(spec, kernel):
    """Stacked-@ entry: append `spec` to the kernel's pending list, return kernel."""
    pending = getattr(kernel, _PENDING, None)
    if pending is None:
        pending = []
        setattr(kernel, _PENDING, pending)
    pending.append(spec)
    return kernel


def start(jit_fn):
    """Terminal decorator marking the end of a stacked-@ ATI block. Finalizes the
    accumulated specs and returns the described object.

    Generic over four stack kinds dispatched by the INNERMOST spec (specs[-1] after
    source-order reversal) — Python applies decorators bottom-up, so the innermost
    decorator's spec is always the kind discriminant (O(1), no scan):
      * OperatorSpec     → operator stack → OperatorDecl
      * AffineKernelSpec → affine stack   → AffineDecl
      * MetroPlan        → metro stack    → fn.__ati_node__ (MetroPlan)
      * anything else    → kernel stack   → KernelSpec via describe()
    """
    pending = getattr(jit_fn, _PENDING, None)
    assert pending is not None, (
        '@ati.start found no pending @ati.* specs below it; either stack at '
        'least one @ati.tensor/@ati.scalar/@ati.overrides above @ati.start, or '
        'use ati.describe(kernel, *specs) (Mode B) instead.')
    specs = list(reversed(pending))      # bottom-up application -> source order
    # Dispatch on the innermost spec (specs[-1]) — the kind discriminant.
    from ..decorators import OperatorSpec
    from ..decorators.affine import AffineKernelSpec
    from .metro import MetroPlan
    marker = specs[-1]
    if isinstance(marker, OperatorSpec):
        _finalize_operator(jit_fn, specs)
    elif isinstance(marker, AffineKernelSpec):
        _finalize_affine(jit_fn, specs)
    elif isinstance(marker, MetroPlan):
        _finalize_metro(jit_fn, specs)
    else:
        describe(jit_fn, *specs)
    delattr(jit_fn, _PENDING)
    return jit_fn


def _finalize_metro(fn, specs):
    """PASSIVE: attach the MetroPlan to fn.__ati_node__. Reads the optional
    UnionPrecedenceSpec from the pending list and stores it on the plan directly."""
    from ..decorators.hints import UnionPrecedenceSpec
    plan = specs[-1]          # innermost — the MetroPlan
    for s in specs[:-1]:
        if isinstance(s, UnionPrecedenceSpec):
            assert plan.precedence is None, 'duplicate @ati.hints.union_precedence'
            plan.precedence = s.names
    fn.__ati_node__ = plan


def _finalize_affine(placeholder, specs):
    """PASSIVE: attach the AffineDecl to fn.__ati_node__."""
    placeholder.__ati_node__ = collect_affine_decl(specs)
    return placeholder


def _finalize_operator(placeholder, specs):
    """PASSIVE: attach the OperatorDecl to fn.__ati_node__."""
    placeholder.__ati_node__ = collect_operator_decl(specs)
    return placeholder


def get_kernel_spec(kernel_obj):
    """The finalized KernelSpec for a kernel, or None. Consumers (the Step 2.4
    builder) use this. Asserts the stacked-@ block was terminated with
    @ati.start (no un-finalized pending specs left dangling)."""
    assert getattr(kernel_obj, _PENDING, None) is None, (
        f'{getattr(kernel_obj, "__name__", kernel_obj)!r} has un-finalized ATI '
        f'specs; a stacked-@ block must end with @ati.start at the top.')
    from .node import AtiNode
    from .kernel import KernelSpec
    node = getattr(kernel_obj, '__ati_node__', None)
    return node if isinstance(node, KernelSpec) else None
