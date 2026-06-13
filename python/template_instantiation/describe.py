# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
describe() + the stacked-@ sugar (executive plan Step 2.3; agent-plans/ati_rev1.md
§3.4).

`ati.describe(kernel, *specs)` is the canonical primitive: it introspects the
kernel's parameter list, validates that the specs claim every argument exactly
once, and attaches a KernelSpec sidecar (`kernel.__ati__`). The stacked-@ form
lowers to the same path — each `@ati.tensor(...)` returns a spec and the eventual
`@triton.jit`-adjacent collection is replayed through describe() — so the two
authoring modes share one implementation and produce an identical KernelSpec.

This step stores the collected, validated specs. Lowering them to the
Axis/Override IR (enumerate_functionals input) is Step 2.4 (builder.py).
"""

from .decorators import TensorSpec, ScalarSpec, ChoiceVar
from .ir import Override
from .introspect import kernel_params


class KernelSpec:
    """The ATI sidecar attached to a kernel as `kernel.__ati__`."""

    __slots__ = ('kernel', 'params', 'tensors', 'scalars', 'overrides', 'tune',
                 'disables', 'dtype_vars', 'cites')

    def __init__(self, kernel, params, tensors, scalars, overrides, tune=None,
                 disables=None, dtype_vars=None, cites=None):
        self.kernel = kernel
        self.params = params           # list[ParamSpec], signature order
        self.tensors = tensors         # list[TensorSpec]
        self.scalars = scalars         # list[ScalarSpec]
        self.overrides = overrides     # list[Override]
        self.tune = tune               # tune specs, attached later (Phase 3)
        self.disables = disables or []  # list[DisableSpec]
        # Named dtype/choice variables declared via @ati.tensor_dtype /
        # ati.choice_set as standalone records (the stacked-@ / string-ref form);
        # tensor/scalar specs refer to them by name. ChoiceVars passed inline by
        # object (the older form) are NOT here — they ride on the spec's dtype.
        self.dtype_vars = dtype_vars or []   # list[ChoiceVar]
        self.cites = cites or []             # list[CiteSpec] (@ati.cite targets)

    @property
    def param_names(self):
        return [p.name for p in self.params]

    def __repr__(self):
        return (f'KernelSpec({getattr(self.kernel, "__name__", self.kernel)!r}, '
                f'{len(self.tensors)} tensors, {len(self.scalars)} scalars, '
                f'{len(self.overrides)} overrides)')


def _build_tune_spec(tune_records):
    """Fold the collected tune spec-records into one TuneSpec, or None if there
    are no tuning decorators on this kernel."""
    from .tune import (
        TuneSpec, PerfSchema, ConfigsSpec, BinningSpec, FallbackSpec, DerivedSpec,
    )
    if not tune_records:
        return None
    ts = TuneSpec()
    for r in tune_records:
        if isinstance(r, PerfSchema):
            assert ts.schema is None, 'duplicate ati.tune.schema on one kernel'
            ts.schema = r
        elif isinstance(r, ConfigsSpec):
            assert ts.configs is None, 'duplicate ati.tune.configs on one kernel'
            ts.configs = r.generator
        elif isinstance(r, BinningSpec):
            ts.binning.update(r.keys)
        elif isinstance(r, FallbackSpec):
            ts.fallback.update(r.values)
        elif isinstance(r, DerivedSpec):
            ts.derived.update(r.programs)
        else:
            raise AssertionError(f'unrecognized tune spec {r!r}')
    return ts


def _partition(specs):
    from .tune import PerfSchema, ConfigsSpec, BinningSpec, FallbackSpec, DerivedSpec
    from .decorators import DisableSpec, CiteSpec
    tune_types = (PerfSchema, ConfigsSpec, BinningSpec, FallbackSpec, DerivedSpec)
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
            for pp in r.params:
                claim(pp.name, f'tune.schema({pp.name})')

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


def describe(kernel, *specs, _validate=True):
    """Attach an ATI KernelSpec to a kernel. Canonical for both authoring modes."""
    params = kernel_params(kernel)
    tensors, scalars, overrides, tune_records, disables, dtype_vars, cites, others = \
        _partition(specs)
    assert not others, f'describe() got unrecognized specs: {others}'
    if _validate:
        errors = _validate_completeness(params, tensors, scalars, tune_records,
                                        has_cite=bool(cites))
        assert not errors, (
            f'ATI describe({getattr(kernel, "__name__", kernel)!r}) validation '
            f'failed:\n  ' + '\n  '.join(errors))
    spec = KernelSpec(kernel, params, tensors, scalars, overrides,
                      tune=_build_tune_spec(tune_records), disables=disables,
                      dtype_vars=dtype_vars, cites=cites)
    kernel.__ati__ = spec
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
# A terminal decorator `@ati.kernel` marks the end of the stack and triggers
# finalization explicitly (no lazy-on-access guessing):
#
#     @ati.kernel                                  # applied LAST -> finalizes
#     @ati.tensor('Q', T_io, strides='stride_q?')
#     @ati.scalar('CAUSAL_TYPE', options=[0, 3])
#     @triton.jit
#     def attn_fwd(...): ...
#
# Python applies decorators bottom-up, so by the time @ati.kernel runs, every
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


def kernel(jit_fn):
    """Terminal decorator marking the end of a stacked-@ ATI block. Finalizes the
    accumulated specs and returns the described object.

    Two stack kinds share this finalizer (TODO: rename to a generic @ati.start
    facade): a KERNEL stack (the default — finalized through describe()) and an
    OPERATOR stack (marked by an innermost @ati.operator — finalized into an
    AtiOperator)."""
    pending = getattr(jit_fn, _PENDING, None)
    assert pending is not None, (
        '@ati.kernel found no pending @ati.* specs below it; either stack at '
        'least one @ati.tensor/@ati.scalar/@ati.overrides above @ati.kernel, or '
        'use ati.describe(kernel, *specs) (Mode B) instead.')
    specs = list(reversed(pending))      # bottom-up application -> source order
    from .decorators import OperatorSpec
    from .affine import AffineMarkerSpec
    if any(isinstance(s, OperatorSpec) for s in specs):
        op = _finalize_operator(jit_fn, specs)
        delattr(jit_fn, _PENDING)
        return op
    if any(isinstance(s, AffineMarkerSpec) for s in specs):
        ak = _finalize_affine(jit_fn, specs)
        delattr(jit_fn, _PENDING)
        return ak
    describe(jit_fn, *specs)
    delattr(jit_fn, _PENDING)
    return jit_fn


def _finalize_affine(placeholder, specs):
    """Build an AtiAffineKernel from a stacked-@ affine description: one
    @ati.affine.aiter_asm marker + the @ati.affine.* metadata specs + an optional
    @ati.disable. SHARED_IFACE is wired later by infer_shared_iface."""
    from .affine import (
        AtiAffineKernel, AffineMarkerSpec, SharedOperatorSpec, ArchSpec,
        LimitationsSpec, StructuresSpec, DirectoriesSpec, SuppliesSpec,
    )
    from .decorators import DisableSpec

    marker = None
    shared_op = None
    arches = []
    filters = {}
    cookie = None
    co_dir = None
    headers = []
    supplied = []
    supplies_after = None
    supplies_before = None
    disable = None
    for s in specs:
        if isinstance(s, AffineMarkerSpec):
            assert marker is None, 'multiple @ati.affine.aiter_asm markers in one stack'
            marker = s
        elif isinstance(s, SharedOperatorSpec):
            shared_op = s.op_name
        elif isinstance(s, ArchSpec):
            arches = s.arches
        elif isinstance(s, LimitationsSpec):
            filters.update(s.filters)
        elif isinstance(s, StructuresSpec):
            cookie = s.cookie
        elif isinstance(s, DirectoriesSpec):
            co_dir, headers = s.co_dir, s.headers
        elif isinstance(s, SuppliesSpec):
            supplied.extend(s.specs)
            supplies_after = s.after if s.after is not None else supplies_after
            supplies_before = s.before if s.before is not None else supplies_before
        elif isinstance(s, DisableSpec):
            disable = s
        else:
            raise AssertionError(
                f'unexpected spec {s!r} in an @ati.affine stack; affine kernels '
                f'accept @ati.affine.* and @ati.disable only')
    assert marker is not None, '@ati.kernel affine path without an @ati.affine marker'
    assert co_dir is not None, f'affine kernel {marker.name!r} missing @ati.affine.directories'
    return AtiAffineKernel(name=marker.name, family='flash', co_dir=co_dir,
                           cookie=cookie, headers=headers, supported_arch=arches,
                           choice_filters=filters, shared_operator_name=shared_op,
                           supplied_specs=supplied, disable=disable,
                           supplies_after=supplies_after, supplies_before=supplies_before)


def _finalize_operator(placeholder, specs):
    """Build an AtiOperator from a stacked-@ operator description: one
    @ati.operator marker (the construction params), @ati.backend specs (the
    interchangeable backends), and operator-level @ati.tune.* (binning ->
    OPTUNE_KEYS, fallback -> PARTIALLY_TUNED; configs warned + ignored)."""
    import warnings
    from .decorators import OperatorSpec, BackendSpec
    from .tune import BinningSpec, FallbackSpec, ConfigsSpec
    from .compat.operator_adapter import build_operator

    opspec = None
    backends = []
    binning = {}
    fallback = {}
    for s in specs:
        if isinstance(s, OperatorSpec):
            assert opspec is None, 'multiple @ati.operator markers in one stack'
            opspec = s
        elif isinstance(s, BackendSpec):
            backends.append(s)
        elif isinstance(s, BinningSpec):
            binning.update(s.keys)            # operator backend-selection keys
        elif isinstance(s, FallbackSpec):
            fallback.update(s.values)         # operator's OWN partial-tune (default {})
        elif isinstance(s, ConfigsSpec):
            warnings.warn(
                f'@ati.tune.configs on operator {opspec.name if opspec else "?"!r} '
                f'is ignored: operator tuning selects a backend (binning) and does '
                f'not generate perf configs. Move configs to the kernel/tune module.',
                stacklevel=3)
        else:
            raise AssertionError(
                f'unexpected spec {s!r} in an @ati.operator stack; operators accept '
                f'only @ati.backend and operator-level @ati.tune.binning/fallback')
    assert opspec is not None, '@ati.kernel operator path without an @ati.operator marker'
    assert backends, f'operator {opspec.name!r} declares no @ati.backend'
    backends.sort(key=lambda b: b.index)
    return build_operator(opspec, backends, binning=binning, fallback=fallback)


def get_kernel_spec(kernel_obj):
    """The finalized KernelSpec for a kernel, or None. Consumers (the Step 2.4
    builder) use this. Asserts the stacked-@ block was terminated with
    @ati.kernel (no un-finalized pending specs left dangling)."""
    assert getattr(kernel_obj, _PENDING, None) is None, (
        f'{getattr(kernel_obj, "__name__", kernel_obj)!r} has un-finalized ATI '
        f'specs; a stacked-@ block must end with @ati.kernel at the top.')
    return getattr(kernel_obj, '__ati__', None)
