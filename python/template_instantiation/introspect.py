# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Triton kernel introspection

ARGUMENTS is the kernel's real parameter order — the human never writes it.

PRIMARY path (triton-free): @ati.source AST-parses the kernel file and produces a
KernelStub. kernel_params reads KernelStub.params directly (no triton import).

FALLBACK path (plain Python callables, e.g. unit-test fixtures): stdlib
inspect.signature. Decorating on @triton.jit directly is NOT supported — the
generator is triton-free and reads no triton-internal attributes.
"""

import inspect


class ParamSpec:
    """One introspected kernel parameter."""
    __slots__ = ('name', 'is_constexpr', 'annotation')

    EMPTY = inspect.Parameter.empty

    def __init__(self, name, is_constexpr, annotation):
        self.name = name
        self.is_constexpr = is_constexpr
        self.annotation = annotation

    @property
    def has_annotation(self) -> bool:
        return self.annotation is not ParamSpec.EMPTY

    def __repr__(self):
        return (f'ParamSpec({self.name!r}, constexpr={self.is_constexpr}, '
                f'annotation={self.annotation!r})')


def _is_constexpr_annotation(ann) -> bool:
    """True if the annotation marks a compile-time constant (`tl.constexpr` or a
    class/string named 'constexpr'). Does NOT import triton."""
    if isinstance(ann, str):
        return ann == 'constexpr' or ann.endswith('.constexpr')
    return getattr(ann, '__name__', '') == 'constexpr'


def kernel_params(jit_fn) -> list[ParamSpec]:
    """The ordered parameter list of a kernel.

    Primary: a KernelStub (from @ati.source) — reads KernelStub.params directly,
    returning EMPTY annotations (types come from @ati.* decorators).
    Fallback: a plain Python callable (test fixtures) — uses inspect.signature."""
    from .decorators.source import KernelStub
    if isinstance(jit_fn, KernelStub):
        return [ParamSpec(name=n, is_constexpr=False, annotation=ParamSpec.EMPTY)
                for n in jit_fn.params]
    fn = getattr(jit_fn, 'fn', jit_fn)   # unwrap @triton.jit without importing triton
    if not callable(fn):
        raise TypeError(f'kernel_params expects a KernelStub or callable, got {jit_fn!r}')
    out = []
    for name, p in inspect.signature(fn).parameters.items():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        out.append(ParamSpec(name=name,
                             is_constexpr=_is_constexpr_annotation(p.annotation),
                             annotation=p.annotation))
    return out


def kernel_annotations(jit_fn) -> dict[str, str]:
    """String type annotations from the @ati.source placeholder def, or empty for
    plain callables (their annotations are not a type source for ATI)."""
    from .decorators.source import KernelStub
    if isinstance(jit_fn, KernelStub):
        return dict(jit_fn.annotations)
    return {}


def kernel_param_names(jit_fn) -> list[str]:
    return [p.name for p in kernel_params(jit_fn)]
