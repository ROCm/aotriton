# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The FlyDSL language module of ir/: KernelDescription (kdesc.py) and
KernelSignature (ksignature.py), how flyc describes a kernel and one compiled
instance of it.
"""

from .kdesc import KernelDescription
from .ksignature import KernelSignature

__all__ = ['KernelDescription', 'KernelSignature']
