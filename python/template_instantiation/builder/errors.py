# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Builder diagnostics (pipeline Stage 4 — LOWER).
"""


class DescriptionError(Exception):
    """A diagnostic from the ATI front-end. Like the Triton compiler frontend it
    partially mimics, it names the kernel and parameter at fault and says how to
    fix it."""
