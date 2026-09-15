# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
describe() + the stacked-@ sugar finalizer

This is the GLUE that turns the loose @ati.* spec-records into the passive Stage-2
"object files" (specs/kernel.py KernelDecl, specs/affine.py AffineDecl,
specs/operator.py OperatorDecl):

`ati.describe(kernel, *specs)` is the canonical primitive: it introspects the
kernel's parameter list, validates that the specs claim every argument exactly
once, and attaches a KernelDecl to the kernel (`kernel.__ati_node__`). The stacked-@ form
lowers to the same path — each `@ati.tensor(...)` returns a spec and the eventual
`@triton.jit`-adjacent collection is replayed through describe() — so the two
authoring modes share one implementation and produce an identical KernelDecl.

This step stores the collected, validated specs. Lowering them to the
Axis/Override IR (enumerate_functionals input) is Step 2.4 (builder.py).
"""

from ..decorators import TensorSpec, ScalarSpec, ChoiceVar
from ..ir import Override
from ..introspect import kernel_params, kernel_annotations
from .bundle import partition
from .kernel import KernelDecl
from .node import derived_decl_fields
from .affine import collect_affine_decl
from .operator import collect_operator_decl
from .flyc import collect_flyc_decl


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


def partition_kernel(specs):
    """Common vocabulary, unrestricted: describe() (both authoring modes --
    Mode A stacked-@ and Mode B ati.describe()) accepts every kind
    `partition()` recognises (tensors/scalars/overrides/dtype_vars/cites/
    disable/tune records), so there is nothing to `forbid()`. Still a named
    wrapper, rather than describe() calling `partition()` directly, so every
    stack kind (kernel/affine/flyc/operator) has a matching `partition_*`
    entry point and the same `reject_remaining()` idiom catches whatever a
    kernel stack does not recognise either."""
    b = partition(specs)
    b.reject_remaining('ati.describe()')
    return b


def describe(kernel, *specs, _validate=True):
    """Attach an ATI KernelDecl to a kernel. Canonical for both authoring modes."""
    params = kernel_params(kernel)
    b = partition_kernel(specs)
    # Placeholder-def string annotations become TensorSpec (pointer types: '*...' /
    # 'LazyTensor:...') or ScalarSpec (all others). Appended so completeness sees
    # them like any other tensor/scalar.
    ann_tensors, ann_scalars = _annotation_specs(kernel, b.tensors, b.scalars)
    tensors = b.tensors + ann_tensors
    scalars = b.scalars + ann_scalars
    if _validate:
        errors = _validate_completeness(params, tensors, scalars, b.tune_records,
                                        has_cite=bool(b.cites))
        assert not errors, (
            f'ATI describe({getattr(kernel, "__name__", kernel)!r}) validation '
            f'failed:\n  ' + '\n  '.join(errors))
    spec = KernelDecl(params=params, tensors=tensors, scalars=scalars,
                      overrides=b.overrides,
                      tune=_build_tune_spec(b.tune_records),
                      dtype_vars=b.dtype_vars, cites=b.cites,
                      **derived_decl_fields(kernel, b.disable))
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
    decorator's spec is always the kind discriminant (O(1), no scan). A
    {marker type: collect_*_decl} table maps it straight to the passive-record
    collector for that stack, attached uniformly as `fn.__ati_node__`:
      * OperatorSpec     → operator stack → OperatorDecl
      * AffineKernelSpec → affine stack   → AffineDecl
      * FlycKernelSpec   → flyc stack     → FlycDecl (NOT routed through
                                             describe() — see specs/flyc.py)
      * MetroSpec        → metro stack    → MetroSpec itself (collect_metro_decl
                                             only fills in precedence)
      * anything else    → kernel stack   → KernelDecl via describe()
    """
    pending = getattr(jit_fn, _PENDING, None)
    assert pending is not None, (
        '@ati.start found no pending @ati.* specs below it; either stack at '
        'least one @ati.tensor/@ati.scalar/@ati.overrides above @ati.start, or '
        'use ati.describe(kernel, *specs) (Mode B) instead.')
    specs = list(reversed(pending))      # bottom-up application -> source order
    # Dispatch on the innermost spec's TYPE (specs[-1]) — the kind discriminant.
    # Local imports: decorators/{affine,flyc}.py import from specs/base.py, so
    # importing their marker types at this module's top level would risk a
    # circular import; importing here, inside start(), sidesteps it.
    from ..decorators import OperatorSpec
    from ..decorators.affine import AffineKernelSpec
    from ..decorators.flyc import FlycKernelSpec
    from .metro import MetroSpec, collect_metro_decl
    collectors = {
        OperatorSpec: collect_operator_decl,
        AffineKernelSpec: collect_affine_decl,
        FlycKernelSpec: collect_flyc_decl,
        MetroSpec: collect_metro_decl,
    }
    marker = specs[-1]
    collector = collectors.get(type(marker))
    if collector is not None:
        jit_fn.__ati_node__ = collector(jit_fn, specs)
    else:
        describe(jit_fn, *specs)
    delattr(jit_fn, _PENDING)
    return jit_fn


def get_kernel_decl(kernel_obj):
    """The finalized KernelDecl for a kernel, or None. Consumers (the Step 2.4
    builder) use this. Asserts the stacked-@ block was terminated with
    @ati.start (no un-finalized pending specs left dangling)."""
    assert getattr(kernel_obj, _PENDING, None) is None, (
        f'{getattr(kernel_obj, "__name__", kernel_obj)!r} has un-finalized ATI '
        f'specs; a stacked-@ block must end with @ati.start at the top.')
    from .node import AtiNode
    from .kernel import KernelDecl
    node = getattr(kernel_obj, '__ati_node__', None)
    return node if isinstance(node, KernelDecl) else None
