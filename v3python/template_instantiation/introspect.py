# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Triton kernel introspection (executive plan Step 2.3; agent-plans/ati_rev1.md §2).

ARGUMENTS is the kernel's real parameter order, read from the @triton.jit
function — the human never writes it. We read triton's JITFunction.params when
available (name, is_constexpr, annotation string); for a plain function (tests,
or a kernel imported where triton is absent) we fall back to inspect.signature so
introspection still yields names + constexpr-ness + annotations.
"""

import inspect


class ParamSpec:
    """One introspected @triton.jit parameter."""
    __slots__ = ('name', 'is_constexpr', 'annotation')

    def __init__(self, name, is_constexpr, annotation):
        self.name = name
        self.is_constexpr = is_constexpr
        self.annotation = annotation       # raw string ('', '*u64', 'constexpr', ...)

    def __repr__(self):
        return (f'ParamSpec({self.name!r}, constexpr={self.is_constexpr}, '
                f'annotation={self.annotation!r})')


def _is_jit_function(fn) -> bool:
    # Avoid importing triton (absent in the generation venv). A JITFunction
    # exposes a `.params` list of objects with name/is_constexpr/annotation.
    return hasattr(fn, 'params') and hasattr(fn, 'fn')


def _from_jit(fn) -> list[ParamSpec]:
    out = []
    for p in fn.params:
        out.append(ParamSpec(
            name=p.name,
            is_constexpr=bool(getattr(p, 'is_constexpr', False)),
            annotation=getattr(p, 'annotation', '') or ''))
    return out


def _annotation_str(ann) -> str:
    if ann is inspect.Parameter.empty or ann is None:
        return ''
    if isinstance(ann, str):
        return ann
    # tl.constexpr and similar objects -> their repr's tail; good enough for the
    # fallback path, which is only used when triton is unavailable.
    return getattr(ann, '__name__', str(ann))


def _from_plain(fn) -> list[ParamSpec]:
    out = []
    for name, p in inspect.signature(fn).parameters.items():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL,
                      inspect.Parameter.VAR_KEYWORD):
            continue
        ann = _annotation_str(p.annotation)
        is_constexpr = 'constexpr' in ann
        out.append(ParamSpec(name=name, is_constexpr=is_constexpr, annotation=ann))
    return out


def kernel_params(jit_fn) -> list[ParamSpec]:
    """The ordered parameter list of a @triton.jit kernel (or plain function)."""
    if _is_jit_function(jit_fn):
        return _from_jit(jit_fn)
    if callable(jit_fn):
        return _from_plain(jit_fn)
    raise TypeError(f'kernel_params expects a callable/JITFunction, got {jit_fn!r}')


def kernel_param_names(jit_fn) -> list[str]:
    return [p.name for p in kernel_params(jit_fn)]
