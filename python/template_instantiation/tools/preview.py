# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
--preview: dump the IMPLICIT structures an ATI description derives, so an author
can review them before committing It is a formatter over the already-built description objects (axes,
overrides, merged argument order, cfields); it imports nothing from codegen/
beyond the small struct-formatting helper, preserving the CLAUDE.md layering.

Rendered for a kernel/operator:
  * the params struct (`<Name>Params { ... }`) — the ABI from the axes;
  * the perf struct, if any;
  * the merged argument order (what replaces a hand-written ARGUMENTS);
  * the canonical axis manifest — anchor order, signature_name, radix, choices,
    and the total functional count (a database-breaking reorder shows in a diff).
"""

from aotriton.codegen.common import codegen_struct_cfields


def _params_struct(kdesc) -> str:
    cfields = kdesc.func_cfields
    body = codegen_struct_cfields(cfields, nalign=4)
    return f'struct {kdesc.param_class_name} {{\n    {body};\n}};'


def _perf_struct(kdesc) -> str:
    cfields = getattr(kdesc, 'perf_cfields', None)
    if not cfields:
        return ''
    body = codegen_struct_cfields(cfields, nalign=4)
    base = kdesc.class_name_base
    return f'struct {base}Perf {{   // nbits-desc packed\n    {body};\n}};'


def _axis_manifest(kdesc) -> str:
    axes = kdesc.axes_all_ordered
    multi = [a for a in axes if not a.is_stride and not a.is_trivial]
    lines = ['// axis manifest (canonical anchor order):']
    total = 1
    for a in multi:
        choices = ', '.join(str(c.triton_compile_signature) for c in a.choices)
        lines.append(f'//   {a.signature_name:<16} radix={a.radix:<3} '
                     f'godel_stride={a.godel_stride:<5} [{choices}]')
        total *= a.radix
    lines.append(f'// total functionals per arch = {total}')
    return '\n'.join(lines)


def preview_kdesc(kdesc) -> str:
    """Render the implicit structures of an KernelDescription / Operator."""
    parts = [
        f'// ==== {kdesc.unique_path.as_posix()} ====',
        _axis_manifest(kdesc),
        '',
        '// merged argument order (replaces a hand-written ARGUMENTS):',
        '//   ' + ', '.join(kdesc.ARGUMENTS if hasattr(kdesc, 'ARGUMENTS')
                            else kdesc._default.ARGUMENTS),
        '',
        _params_struct(kdesc),
    ]
    perf = _perf_struct(kdesc)
    if perf:
        parts += ['', perf]
    return '\n'.join(parts)


def preview(selective=None, kernels=None, operators=None) -> str:
    """Render --preview output for the ATI items matching `selective` (a
    unique_path string/glob), or all ATI items if None."""
    from ..ir.operator import Operator
    out = []
    items = list(kernels or []) + list(operators or [])
    for it in items:
        # only ATI-backed items have the axis surface
        if not hasattr(it, 'axes_all_ordered') and not isinstance(it, Operator):
            continue
        if selective is not None and not it.unique_path.match(selective):
            continue
        out.append(preview_kdesc(it))
    return '\n\n'.join(out)
