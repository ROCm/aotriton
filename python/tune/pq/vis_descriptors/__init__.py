# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Registry of per-family visperf descriptors (modular-tune.md §3d.1).

This package used to hold the flash descriptor directly
(`vis_descriptors/flash.py`); it now holds only the registry shim. Each
family's descriptor lives at `modules/<family>/visperf/__init__.py` and is
loaded by path via `aotriton.tune.registry.load_family_visperf` -- the same
mechanism `load_family_tune` uses for `modules/<family>/tune/` (F6), so
`modules/<family>` stays a plain directory rather than a package.

Adding a family means registering it in `registry.py`'s family list; no
import here changes.
"""

from ...registry import available_module_names, load_family_visperf

# Registry: id -> descriptor dict
DESCRIPTORS: dict[str, dict] = {}
for _family in available_module_names():
    _descriptor = load_family_visperf(_family).DESCRIPTOR
    DESCRIPTORS[_descriptor['id']] = _descriptor
del _family, _descriptor
