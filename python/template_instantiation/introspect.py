# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Triton kernel introspection (executive plan Step 2.3; agent-plans/ati_rev1.md §2).

ARGUMENTS is the kernel's real parameter order, read from the @triton.jit
function — the human never writes it.

We deliberately do NOT read triton's JITFunction.params: that object, its
.is_constexpr flag, and especially its annotation *normalization* (collapsing
tl.int32 / constexpr_or_i32 / 'i32' all to the string 'i32', tl.int1 to 'u1',
etc.) are undocumented internals that can change between releases. Instead we use
the stdlib inspect.signature over the underlying function and keep each
annotation as the RAW object the author wrote — a tl.dtype instance
(tl.float32, ...), the tl.constexpr class, a type string ('*u64'), or empty. The
builder maps those by identity/value (see builder._ANNOTATION_TYPE), so we depend
only on Triton's public type objects, never on internal strings.
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
    """The ordered parameter list of a @triton.jit kernel (or plain function),
    via stdlib inspect.signature. Annotations are kept raw."""
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
