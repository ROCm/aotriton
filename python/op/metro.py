# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from ..base import (
    Interface,
    Functional,
)
from ..kernel import KernelDescription
import numpy as np

def _iter_concrete_subkernels(node):
    """Yield the concrete sub-kernels of a metro step, descending into
    ConditionalKernel branches (if/else). Duck-typed so this description-layer
    module needs no import of ConditionalKernel."""
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


class MetroKernel(Interface):
    TUNE_NAME = None

    def __init__(self,
                 metro_name : str,
                 kernels : list[KernelDescription]):
        self.NAME = metro_name
        super().__init__()
        self._late_init()
        self._kernels = kernels

    @property
    def class_name_base(self):
        return "".join(x.capitalize() for x in self.NAME.lower().split("_"))

    @property
    def enum_name(self):
        return f'kMetro_{self.class_name_base}'

    def list_non_functional_params(self):
        return []

    def translate_dataframe(self, f : Functional, df : 'pandas.DataFrame'):
        raise RuntimeError(f'translate_dataframe should not be called over any MetroKernel {self.NAME=}')

    def translate_empty_dataframe(self, f : Functional):
        raise RuntimeError(f'translate_empty_dataframe should not be called over any MetroKernel {self.NAME=}')

    def list_kernels(self):
        return self._kernels

    def get_kernel(self, name):
        """The concrete sub-kernel named `name` (flattening ConditionalKernel
        branches), for @ati.cite("<op>.<metro>.<kernel>") resolution. Raises
        KeyError if absent."""
        for sub in self.iter_subkernels():
            if getattr(sub, 'NAME', None) == name:
                return sub
        raise KeyError(
            f'metro {self.NAME!r} has no sub-kernel named {name!r}; '
            f'sub-kernels: {[getattr(s, "NAME", None) for s in self.iter_subkernels()]}')

    def iter_subkernels(self):
        """Yield the concrete sub-kernels in metro call order (flattening any
        ConditionalKernel branches: if-kernel then else-kernel)."""
        for step in self._kernels:
            yield from _iter_concrete_subkernels(step)

    def merged_operand_order(self):
        """The operator's operand list: an order-preserving merge (union_params) of
        every sub-kernel's ARGUMENTS, each first translated through its APPAREL
        mapping (real -> operator operand, read from the sub-kernel's kdesc via
        apparel_of). A wired argument (debug's `R` -> `encoded_softmax`) collapses
        into the operand node it is dressed as, so it is never added to the params
        struct as a separate field."""
        from aotriton.template_instantiation.ops import union_params
        subs = list(self.iter_subkernels())
        arg_lists = [list(s.ARGUMENTS) for s in subs]
        # Per-sub-kernel real->apparel map, from the kdesc (wires_to=); legacy
        # kdescs have no apparel_of, so they pass through unchanged.
        renames = []
        for s in subs:
            apparel_of = getattr(s, 'apparel_of', None)
            if apparel_of is None:
                renames.append({})
            else:
                renames.append({a: apparel_of(a) for a in s.ARGUMENTS
                                if apparel_of(a) != a})
        return union_params(arg_lists, renames=renames)

    def iter_kernel_slot_names(self):
        """Generator that yields KernelSlot enum names for all kernels.

        Delegates to each kernel's iter_kernel_slot_names() method.
        """
        for kdesc in self._kernels:
            yield from kdesc.iter_kernel_slot_names()
