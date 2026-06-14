# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Triton kernel introspection (executive plan Step 2.3; agent-plans/ati_rev1.md §2).

ARGUMENTS is the kernel's real parameter order — the human never writes it.

PRIMARY path (triton-free, agent-plans/ati_triton-free_exec0.md): @ati.source AST-parses
the kernel file and produces a KernelStub carrying `__ati_params__` (the parameter
NAMES). kernel_params returns those with EMPTY annotations — the generator never imports
triton, and every parameter's type is supplied explicitly by the @ati.* decorators.

FALLBACK path (a real callable / @triton.jit function, e.g. a unit test passing a plain
function): we use stdlib inspect.signature over the underlying function and keep each
annotation as the RAW object the author wrote — a tl.dtype instance, the tl.constexpr
class, a type string ('*u64'), or empty. We deliberately do NOT read triton's
JITFunction.params (its .is_constexpr flag and annotation normalization are undocumented
internals); the builder maps raw annotations by identity/value (builder._ANNOTATION_TYPE),
depending only on Triton's public type objects when triton is present.
"""

import inspect


class ParamSpec:
    """One introspected @triton.jit parameter. `annotation` is the raw object
    from the signature (tl.dtype | the tl.constexpr class | str | EMPTY)."""
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


def _underlying_fn(jit_fn):
    """The plain python function behind a @triton.jit kernel, or the callable
    itself. `.fn` is the only triton attribute we touch and is far more stable
    than .params."""
    return getattr(jit_fn, 'fn', jit_fn)


def _is_constexpr_annotation(ann) -> bool:
    """True if the annotation marks a triton compile-time constant. Detected by
    identity against the public tl.constexpr class when triton is importable;
    falls back to a name check when it is not (the annotation is then a string or
    a stand-in class)."""
    try:
        from triton.language import constexpr as _constexpr
        if ann is _constexpr:
            return True
    except Exception:
        pass
    # triton-absent / plain-function path: annotation may be a string or a class
    # named 'constexpr'.
    if isinstance(ann, str):
        return ann == 'constexpr' or ann.endswith('.constexpr')
    return getattr(ann, '__name__', '') == 'constexpr'


def kernel_params(jit_fn) -> list[ParamSpec]:
    """The ordered parameter list of a kernel.

    Primary path: a KernelStub from @ati.source (AST-parsed, no triton import) carries
    `__ati_params__` — the parameter NAMES. Types are supplied by the @ati.* decorators,
    so the stub has no annotations (annotation=EMPTY, is_constexpr=False). This is the
    triton-free generator path (agent-plans/ati_triton-free_exec0.md).

    Fallback path: a real callable / @triton.jit function (e.g. a unit test passing a
    plain function) is introspected via stdlib inspect.signature with raw annotations."""
    ati_params = getattr(jit_fn, '__ati_params__', None)
    if ati_params is not None:
        return [ParamSpec(name=n, is_constexpr=False, annotation=ParamSpec.EMPTY)
                for n in ati_params]
    fn = _underlying_fn(jit_fn)
    if not callable(fn):
        raise TypeError(f'kernel_params expects a callable/JITFunction, got {jit_fn!r}')
    out = []
    for name, p in inspect.signature(fn).parameters.items():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL,
                      inspect.Parameter.VAR_KEYWORD):
            continue
        out.append(ParamSpec(
            name=name,
            is_constexpr=_is_constexpr_annotation(p.annotation),
            annotation=p.annotation))
    return out


def kernel_param_names(jit_fn) -> list[str]:
    return [p.name for p in kernel_params(jit_fn)]
