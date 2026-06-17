# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
`ati.source` — bind a description to a Triton kernel by AST-parsing its source file
(no import; the generator needs no `triton` package — agent-plans/ati_triton-free_exec0.md).
Produces a KernelStub the stacked-@ specs above it attach to.
"""

import ast
import inspect
from pathlib import Path


class SourceError(Exception):
    """A bad @ati.source: missing file or kernel symbol."""


class KernelStub:
    """A non-importing stand-in for a @triton.jit kernel produced by @ati.source.

    ATI never CALLS the kernel — it only introspects it: the parameter names (the
    ARGUMENTS order), the kernel symbol name, and the source-file path. So instead of
    importing the kernel module (which would require `triton` in the venv), @ati.source
    parses the file with `ast` and returns this stub. The triton kernels stay pure
    triton; their types are supplied entirely by the @ati.* decorators (annotations on
    the kernel are intentionally NOT read — see agent-plans/ati_triton-free_exec0.md).

    `__name__` / `__ati_source_path__` mirror the attributes the old JITFunction
    carried (consumed by builder.build_kernel and KernelSpec.source_path);
    `__ati_params__` is the AST-extracted parameter-name list (consumed by
    introspect.kernel_params). `__ati_pending__` / `__ati__` are the stacked-@ sidecars
    describe.py sets on the kernel object (the pending spec list during stacking, then
    the finalized KernelSpec)."""

    __slots__ = ('__name__', '__ati_params__', '__ati_source_path__',
                 '__ati_pending__', '__ati__')

    def __init__(self, name, params, source_path):
        self.__name__ = name
        self.__ati_params__ = list(params)
        self.__ati_source_path__ = source_path

    def __repr__(self):
        return (f'KernelStub({self.__name__!r}, {len(self.__ati_params__)} params, '
                f'{self.__ati_source_path__!r})')


def _ast_kernel_param_names(src, sym, path):
    """Parameter names of the function `sym` in the source file `src`, via AST — no
    import, no execution. Skips *args/**kwargs (triton kernels never use them).
    Raises SourceError if the file has no such top-level function."""
    tree = ast.parse(src.read_text(encoding='utf-8'), filename=str(src))
    fn = next((n for n in tree.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
               and n.name == sym), None)
    if fn is None:
        raise SourceError(
            f"@ati.source({path!r}): file {src.name} has no top-level function "
            f"{sym!r} (pass name= if the kernel symbol differs from the def name)")
    a = fn.args
    if a.vararg is not None or a.kwarg is not None:
        raise SourceError(
            f"@ati.source({path!r}): kernel {sym!r} uses *args/**kwargs, which ATI "
            f"cannot introspect into a fixed ARGUMENTS order")
    return [p.arg for p in (a.posonlyargs + a.args + a.kwonlyargs)]


def source(path, name=None):
    """Innermost stacked-@ decorator: AST-parse the Triton source at `path` and return
    a KernelStub for the kernel it defines, so the @ati.* decorators ABOVE stack onto
    it without copying the kernel into the description (agent-plans/ati_modular_rev0.md
    §5a). The source file is NOT imported — only parsed — so the generator needs no
    `triton` package (agent-plans/ati_triton-free_exec0.md).

        @ati.kernel
        @ati.tensor('Q', 'T_io', strides='stride_q?', contiguous=-1)
        # ... more @ati.* specs ...
        @ati.source("../kernel/fwd_kernel.py")   # innermost, just above the def
        def attn_fwd():                          # placeholder: no args, body `pass`
            pass

    `path` is resolved relative to the DESCRIPTION file (the caller's __file__).
    The kernel symbol parsed from the source defaults to the placeholder `def`'s name;
    pass `name=` to override (the source filename, the kernel symbol, and the
    description module name are all independent).

    MUST be the innermost @ati.* (directly above the def): decorators apply
    bottom-up, so source() runs first and supplies the object every spec above then
    attaches to. A spec placed BELOW it would attach to the placeholder.
    """
    # Resolve `path` relative to the caller's file (the description module), not CWD.
    caller_file = inspect.stack()[1].filename
    base = Path(caller_file).resolve().parent
    src = (base / path).resolve()

    def _decorator(placeholder):
        sym = name or placeholder.__name__
        if not src.exists():
            raise SourceError(
                f"@ati.source: kernel source {src} (from {path!r}, relative to "
                f"{base}) does not exist")
        params = _ast_kernel_param_names(src, sym, path)
        return KernelStub(sym, params, str(src))

    return _decorator
