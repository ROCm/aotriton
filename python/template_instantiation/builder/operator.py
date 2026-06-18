# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Operator params-struct builder (pipeline Stage 4 — LOWER).

Merges the operator's backends' sub-kernel func_cfields into the one
order-preserving params-struct field list the Operator IR (ir/operator.py) carries.
The linker calls this when an operator's struct is NOT a single kernel's superset
(the bwd case); the fwd case reuses the default kernel's func_cfields directly.
"""


def build_merged_struct_cfields(subkernels):
    """The operator params-struct field list, merged across all backends' sub-kernels.

    For an operator whose params struct is NOT a single kernel's superset (the bwd
    operator: the struct is the union of dk_dv/dq + the preprocess kernels' `Out` +
    the affine kernel's `DQ_ACC`), merge the sub-kernels' `func_cfields` into one
    order-preserving list via union_params over their (apparel) field names. The
    FIRST sub-kernel to define a name owns its cfield (so each operand's ctype/nbits
    come from its defining kernel). `subkernels` must be given in the desired priority
    order (key kernels first), since union_params resolves order conflicts by
    first-listed-wins.

    A backend may provide a `union_order` (e.g. the affine kernel: an anchored chain
    like [DB, DQ_ACC, L]) used purely for ORDERING — so an operand only that backend
    supplies lands between its declared neighbors — while its cfields still come from
    `func_cfields`.
    """
    from ..ir.cfield import cfield
    from ..ir.ops import union_params
    cfield_by_name = {}
    name_lists = []
    for s in subkernels:
        for cf in s.func_cfields:
            cfield_by_name.setdefault(cf.aname, cf)   # first definer owns the cfield
        order_hint = getattr(s, 'union_order', None)
        if order_hint:
            name_lists.append(list(order_hint))       # anchored ordering chain
        else:
            name_lists.append([cf.aname for cf in s.func_cfields])
    order = union_params(name_lists)
    merged = []
    for i, name in enumerate(order):
        cf = cfield_by_name[name]
        merged.append(cfield(ctype=cf.ctype, aname=cf.aname, ctext=cf.ctext,
                             index=i, nbits=cf.nbits))
    return merged
