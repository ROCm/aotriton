# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The affine (AITER-ASM) language module of ir/: AffineKernel (kdesc.py).

No ksignature.py here: an affine kernel has no functional space
(gen_functionals yields nothing), no perf, and its per-image unit is
co_gen() over prebuilt .co files.
"""

from .kdesc import AffineKernel

__all__ = ['AffineKernel']
