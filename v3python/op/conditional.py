# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from ..base import (
    Interface,
    Functional,
)
from ..kernel import KernelDescription
from ..base.typed_choice import constexpr_base

class ConditionalKernel(Interface):
    def __init__(self,
                 if_parameter : str,
                 if_expr : str,
                 if_kernel : KernelDescription,
                 else_kernel : KernelDescription = None):
        self._if_parameter = if_parameter
        self._if_expr = if_expr
        self._if_kenrel = if_kernel
        self._else_kenrel = else_kernel
        if self._else_kenrel is not None:
            self.NAME = f'if_{if_kernel.NAME}_else_{else_kernel.NAME}'
        else:
            self.NAME = f'if_{if_kernel.NAME}'

    @property
    def class_name_base(self):
        return "".join(x.capitalize() for x in self.NAME.lower().split("_"))

    @property
    def enum_name(self):
        return f'kConditional_{self.class_name_base}'

    def list_non_functional_params(self):
        return []

    @property
    def if_parameter(self):
        return self._if_parameter

    # Returns the value for C++ source code
    @property
    def if_expr(self):
        return self._if_expr

    @property
    def if_kernel(self):
        return self._if_kenrel

    @property
    def else_kernel(self):
        return self._else_kenrel

    def translate_dataframe(self, f : Functional, df : 'pandas.DataFrame'):
        raise RuntimeError(f'translate_dataframe should not be calle over any ConditionalKernel {self.NAME=}')

    def translate_empty_dataframe(self, f : Functional):
        raise RuntimeError(f'translate_empty_dataframe should not be calle over any ConditionalKernel {self.NAME=}')
