# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI metro-kernel builder.

A metro is a LAUNCHER: it sequences its sub-kernels' contexts to implement one
operator functional (e.g. attn_fwd then the debug kernel; or preprocess + dk_dv +
dq). It owns no params struct or functional space of its own — the operator does.

`MetroKernel` is the ATI metro launcher and `ConditionalKernel` its if/else step;
both are ATI-native (subclass the ATI Interface base, no legacy aotriton.op). They
expose only the launcher surface the operator codegen reads (`enum_name`,
`list_kernels`, `iter_subkernels`, `get_kernel`, `iter_kernel_slot_names`).
`build_metro` lowers a transpiled @ati.metro_kernel plan straight to one.
"""

from ..ir import Interface
from .transpile import lower_plan


def _iter_concrete_subkernels(node):
    """Yield the concrete sub-kernels of a metro step, descending into
    ConditionalKernel branches (if/else)."""
    if hasattr(node, 'list_kernels'):           # a nested MetroKernel
        for step in node.list_kernels():
            yield from _iter_concrete_subkernels(step)
        return
    if hasattr(node, 'if_kernel'):              # a ConditionalKernel
        yield from _iter_concrete_subkernels(node.if_kernel)
        if node.else_kernel is not None:
            yield from _iter_concrete_subkernels(node.else_kernel)
        return
    yield node                                  # a concrete kernel


class ConditionalKernel(Interface):
    """One if/else metro step: a condition (if_parameter, if_expr) and the then/else
    sub-kernels (the C++ launcher template supports one kernel per branch)."""

    CODEGEN_MODULE = 'op'
    ENUM_PREFIX = 'kConditional_'

    def __init__(self, if_parameter, if_expr, if_kernel, else_kernel=None):
        self._if_parameter = if_parameter
        self._if_expr = if_expr
        self._if_kernel = if_kernel
        self._else_kernel = else_kernel
        if else_kernel is not None:
            self.NAME = f'if_{if_kernel.NAME}_else_{else_kernel.NAME}'
        else:
            self.NAME = f'if_{if_kernel.NAME}'

    @property
    def if_parameter(self):
        return self._if_parameter

    @property
    def if_expr(self):
        return self._if_expr

    @property
    def if_kernel(self):
        return self._if_kernel

    @property
    def else_kernel(self):
        return self._else_kernel

    def iter_kernel_slot_names(self):
        yield from self._if_kernel.iter_kernel_slot_names()
        if self._else_kernel is not None:
            yield from self._else_kernel.iter_kernel_slot_names()


class MetroKernel(Interface):
    """A metro launcher built from a transpiled @ati.metro_kernel plan. Has no
    functional space of its own — it sequences sub-kernel contexts."""

    CODEGEN_MODULE = 'op'
    TUNE_NAME = None
    ENUM_PREFIX = 'kMetro_'

    def __init__(self, name, kernels, *, family='flash'):
        self.NAME = name
        self.FAMILY = family
        self._kernels = list(kernels)

    def list_kernels(self):
        return self._kernels

    def iter_subkernels(self):
        """Concrete sub-kernels in metro call order (flattening ConditionalKernel
        branches: if-kernel then else-kernel)."""
        for step in self._kernels:
            yield from _iter_concrete_subkernels(step)

    def get_kernel(self, name):
        """The concrete sub-kernel named `name`, for @ati.cite resolution. Raises
        KeyError if absent."""
        for sub in self.iter_subkernels():
            if getattr(sub, 'NAME', None) == name:
                return sub
        raise KeyError(
            f'metro {self.NAME!r} has no sub-kernel named {name!r}; '
            f'sub-kernels: {[getattr(s, "NAME", None) for s in self.iter_subkernels()]}')

    def iter_kernel_slot_names(self):
        for kdesc in self._kernels:
            yield from kdesc.iter_kernel_slot_names()

    def merged_operand_order(self):
        """Order-preserving merge (union_params) of every sub-kernel's ARGUMENTS,
        each translated through its APPAREL map (real -> operand, via apparel_of)."""
        from ..ops import union_params
        subs = list(self.iter_subkernels())
        arg_lists = [list(s.ARGUMENTS) for s in subs]
        renames = []
        for s in subs:
            apparel_of = getattr(s, 'apparel_of', None)
            if apparel_of is None:
                renames.append({})
            else:
                renames.append({a: apparel_of(a) for a in s.ARGUMENTS
                                if apparel_of(a) != a})
        return union_params(arg_lists, renames=renames)


def build_metro(plan, kernel_map, name, *, family='flash'):
    """Lower a transpiled MetroPlan to a MetroKernel (a metro launcher).

    plan:        the @ati.metro_kernel transpiler output (fn.__ati_metro__).
    kernel_map:  {sub-kernel name -> built kdesc}.
    name:        the metro/backend NAME (-> kMetro_<Name> enum).
    """
    return lower_plan(plan, kernel_map,
                      lambda steps: MetroKernel(name, steps, family=family),
                      ConditionalKernel)
