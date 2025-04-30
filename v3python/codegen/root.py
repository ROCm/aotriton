# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Root of the Generation process

from ..rules import kernels as triton_kernels
from .kernel import KernelShimGenerator
# from .op import OpGenerator

class RootGenerator(object):
    def __init__(self, args):
        self._args = args

    def generate(self):
        for k in triton_kernels:
            KernelShimGenerator(self._args, k).generate()
        # for op in dispatcher_operators:
        #     OpGenerator(self._args, op).generate()
