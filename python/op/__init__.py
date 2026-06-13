# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from .operator import Operator
from .metro import MetroKernel
from .conditional import ConditionalKernel

class NO_OPERATOR(Operator):
    NAME = None
