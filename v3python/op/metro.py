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
                 kernels : list[KernelDescription],
                 renames : dict = None):
        self.NAME = metro_name
        super().__init__()
        self._late_init()
        self._kernels = kernels
        # How a sub-kernel's own arguments wire to the operator's operands:
        # {concrete sub-kernel object -> {kernel_arg: operand}}. The metro DSL
        # (e.g. `debug(params, R=params.encoded_softmax)`) is the source; renaming
        # is a property of the COLLABORATION, so it lives on the metro, not the
        # sub-kernel. Used to merge wired arguments into one node (so a renamed arg
        # is not added to the params struct twice) and to wire the launch call.
        self._renames = dict(renames) if renames else {}

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

    def rename_for(self, subkernel) -> dict:
        """The {kernel_arg: operand} rename map wiring `subkernel`'s own arguments
        to the operator's operands (empty if the sub-kernel needs no renaming)."""
        return self._renames.get(subkernel, {})

    def iter_subkernels(self):
        """Yield the concrete sub-kernels in metro call order (flattening any
        ConditionalKernel branches: if-kernel then else-kernel)."""
        for step in self._kernels:
            yield from _iter_concrete_subkernels(step)

    def merged_operand_order(self):
        """The operator's operand list: an order-preserving merge (union_params) of
        every sub-kernel's ARGUMENTS, each first translated through its rename map.
        A wired argument (debug's `R` -> `encoded_softmax`) collapses into the
        operand node it is wired to, so it is never added to the params struct as a
        separate field."""
        from v3python.template_instantiation.operator import union_params
        subs = list(self.iter_subkernels())
        arg_lists = [list(s.ARGUMENTS) for s in subs]
        renames = [self.rename_for(s) for s in subs]
        return union_params(arg_lists, renames=renames)

    def iter_kernel_slot_names(self):
        """Generator that yields KernelSlot enum names for all kernels.

        Delegates to each kernel's iter_kernel_slot_names() method.
        """
        for kdesc in self._kernels:
            yield from kdesc.iter_kernel_slot_names()
