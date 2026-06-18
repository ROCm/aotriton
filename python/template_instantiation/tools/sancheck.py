# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
--sancheck: a read-only validation pass over ATI descriptions (executive plan
Step 5.3; agent-plans/ati_rev1.md §9.1a). Collects ALL problems at once (not
first-failure) and never generates code. Centralizes the scattered asserts:

  * completeness — every introspected parameter claimed exactly once
    (orphans + double-claims), reusing describe._validate_completeness;
  * override scope — predicates/VarRefs read free axes only (or arch);
  * no arg in two axes — each argument owned by exactly one axis;
  * derive targets resolve — to a real argument or perf-schema field;
  * (operators) union cycles — the metro merge has no unresolved CycleError.

Returns a list of diagnostics keyed by kernel; exit non-zero if any.
"""

from graphlib import CycleError


def sancheck_kernel_spec(kernel_spec):
    """Validate one KernelSpec (from describe()). Returns a list of error strings
    (empty == clean). Does not raise."""
    from ..specs.finalize import _validate_completeness
    from ..builder import build_kernel, DescriptionError

    name = getattr(kernel_spec.kernel, '__name__', 'kernel')
    errors = []
    tune_records = _tune_specs(kernel_spec)

    # 1. completeness (orphans / double-claims / unknown names)
    errors += [f'{name}: {e}'
               for e in _validate_completeness(kernel_spec.params,
                                               kernel_spec.tensors,
                                               kernel_spec.scalars, tune_records)]

    # 2/3/4. build the IR (axes/overrides) and validate scope + resolution.
    try:
        built = build_kernel(kernel_spec)
    except DescriptionError as e:
        errors.append(f'{name}: {e}')
        return errors

    free_vars = {ax.var_name for ax in built.axes if not ax.is_stride}
    arg_names = set(built.arguments)
    perf_names = ({pp.name for pp in built.tune.schema.params}
                  if (built.tune and built.tune.schema) else set())

    # no arg in two axes
    seen = {}
    for ax in built.axes:
        for a in ax.arg_names:
            if a in seen:
                errors.append(f'{name}: argument {a!r} is in two axes '
                              f'({seen[a]!r} and {ax.var_name!r})')
            else:
                seen[a] = ax.var_name

    # override predicate/VarRef scope + target resolution
    for ov in (*built.overrides, *built.perf_overrides):
        try:
            ov.validate(free_vars)
        except AssertionError as e:
            errors.append(f'{name}: {e}')
        for t in ov.targets:
            if t not in arg_names and t not in perf_names:
                errors.append(f'{name}: derive target {t!r} is neither an '
                              f'argument nor a perf-schema field')

    return errors


def _tune_specs(kernel_spec):
    """The tune spec-records affecting completeness — only the PerfSchema (its
    fields are claimable params); configs/binning/fallback/derived do not."""
    ts = kernel_spec.tune
    if ts is None or ts.schema is None:
        return []
    return [ts.schema]


def sancheck_report(kernels=None, operators=None):
    """Run sancheck over the given ATI kernels/operators. Returns (ok, errors).
    Only ATI-described items (an KernelDescription with a `kernel_spec`, or an
    Operator) are checked; legacy descriptions are skipped."""
    from ..ir.operator import Operator

    errors = []
    seen_specs = set()

    def _check(kdesc):
        spec = getattr(kdesc, 'kernel_spec', None)
        if spec is not None and id(spec) not in seen_specs:
            seen_specs.add(id(spec))
            errors.extend(sancheck_kernel_spec(spec))

    for k in (kernels or []):
        _check(k)
    for op in (operators or []):
        if isinstance(op, Operator):
            _check(op._default)                  # the default backend's kernel spec
            try:
                _ = op.func_cfields              # exercises the struct build
            except CycleError as e:
                errors.append(f'{op.NAME}: operator param-struct merge cycle: {e}')
    return (not errors), errors
