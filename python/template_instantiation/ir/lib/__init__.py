# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Shared helper FUNCTIONS the per-language ir/ modules (triton/, affine/) call.

This is a library, not a base class: there is no KernelSignatureBase for a language
to inherit and override. Only what must agree across languages lives here — today
that is the entry-name grammar and the hsaco-entry hash. Everything specific to one
language's vocabulary (e.g. Triton's num_warps/num_stages/waves_per_eu, the
gfx1250 workaround) stays in that language's own ksignature.py.
"""
