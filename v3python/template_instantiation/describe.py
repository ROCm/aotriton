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

from .decorators import TensorSpec, ScalarSpec
from .ir import Override
from .introspect import kernel_params


class KernelSpec:
    """The ATI sidecar attached to a kernel as `kernel.__ati__`."""

    __slots__ = ('kernel', 'params', 'tensors', 'scalars', 'overrides', 'tune')

    def __init__(self, kernel, params, tensors, scalars, overrides, tune=None):
        self.kernel = kernel
        self.params = params           # list[ParamSpec], signature order
        self.tensors = tensors         # list[TensorSpec]
        self.scalars = scalars         # list[ScalarSpec]
        self.overrides = overrides     # list[Override]
        self.tune = tune               # tune specs, attached later (Phase 3)

    @property
    def param_names(self):
        return [p.name for p in self.params]

    def __repr__(self):
        return (f'KernelSpec({getattr(self.kernel, "__name__", self.kernel)!r}, '
                f'{len(self.tensors)} tensors, {len(self.scalars)} scalars, '
                f'{len(self.overrides)} overrides)')


def _partition(specs):
    tensors, scalars, overrides, others = [], [], [], []
    for s in specs:
        if isinstance(s, TensorSpec):
            tensors.append(s)
        elif isinstance(s, ScalarSpec):
            scalars.append(s)
        elif isinstance(s, Override):
            overrides.append(s)
        else:
            others.append(s)
    return tensors, scalars, overrides, others


def _validate_completeness(params, tensors, scalars):
    """Every introspected parameter must be claimed exactly once — either by a
    tensor (as the tensor itself or one of its stride globs) or by a scalar.
    Reports orphans and double-claims (the §9.1a completeness check, kernel-scoped).
    """
    param_names = [p.name for p in params]
    name_set = set(param_names)
    claims = {}      # arg_name -> list of claimant descriptions

    def claim(arg_name, who):
        claims.setdefault(arg_name, []).append(who)

    for t in tensors:
        claim(t.arg_name, f'tensor({t.arg_name})')
        for sname in t.match_strides(param_names):
            claim(sname, f'tensor({t.arg_name}).strides')
    for s in scalars:
        claim(s.arg_name, f'scalar({s.arg_name})')

    errors = []
    # claims referencing names not in the signature
    for arg_name, who in claims.items():
        if arg_name not in name_set:
            errors.append(f'{who[0]} references unknown parameter {arg_name!r}')
    # double-claims
    for arg_name, who in claims.items():
        if len(who) > 1:
            errors.append(f'parameter {arg_name!r} claimed by multiple specs: {who}')
    # orphans (unclaimed params)
    for name in param_names:
        if name not in claims:
            errors.append(f'parameter {name!r} is not claimed by any '
                          f'@ati.tensor/@ati.scalar (or stride glob)')
    return errors


def describe(kernel, *specs, _validate=True):
    """Attach an ATI KernelSpec to a kernel. Canonical for both authoring modes."""
    params = kernel_params(kernel)
    tensors, scalars, overrides, others = _partition(specs)
    assert not others, f'describe() got unrecognized specs: {others}'
    if _validate:
        errors = _validate_completeness(params, tensors, scalars)
        assert not errors, (
            f'ATI describe({getattr(kernel, "__name__", kernel)!r}) validation '
            f'failed:\n  ' + '\n  '.join(errors))
    spec = KernelSpec(kernel, params, tensors, scalars, overrides)
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
    accumulated specs through describe() and returns the kernel."""
    pending = getattr(jit_fn, _PENDING, None)
    assert pending is not None, (
        '@ati.kernel found no pending @ati.* specs below it; either stack at '
        'least one @ati.tensor/@ati.scalar/@ati.overrides above @ati.kernel, or '
        'use ati.describe(kernel, *specs) (Mode B) instead.')
    specs = list(reversed(pending))      # bottom-up application -> source order
    describe(jit_fn, *specs)
    delattr(jit_fn, _PENDING)
    return jit_fn


def get_kernel_spec(kernel_obj):
    """The finalized KernelSpec for a kernel, or None. Consumers (the Step 2.4
    builder) use this. Asserts the stacked-@ block was terminated with
    @ati.kernel (no un-finalized pending specs left dangling)."""
    assert getattr(kernel_obj, _PENDING, None) is None, (
        f'{getattr(kernel_obj, "__name__", kernel_obj)!r} has un-finalized ATI '
        f'specs; a stacked-@ block must end with @ati.kernel at the top.')
    return getattr(kernel_obj, '__ati__', None)
