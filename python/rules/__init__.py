# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The family registry the generator consumes (`aotriton.rules`).

Interim protocol (pre-`--module_dir`): each kernel *family* lives in
`<repo>/modules/<family>/aot`, a python package whose `__init__` exposes
`kernels` / `operators` / `affine_kernels`. This aggregator puts the repo
`modules/` dir on `sys.path` (ONE entry) and enumerates the `modules/*/aot`
packages, concatenating their exports.

`--module_dir` (taking the modules root from the generator CLI instead of the
fixed repo-relative path) is deferred to a later phase; today the repo-relative
`modules/` dir IS the protocol.
"""

import sys
import importlib
from pathlib import Path

# <repo>/modules — repo root is the parent of the aotriton package dir (python/).
_MODULES_DIR = Path(__file__).resolve().parent.parent.parent / 'modules'
if _MODULES_DIR.is_dir():
    _p = str(_MODULES_DIR)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _discover_families():
    """Family names = subdirs of modules/ that contain an `aot` package."""
    if not _MODULES_DIR.is_dir():
        return []
    names = []
    for child in sorted(_MODULES_DIR.iterdir()):
        if (child / 'aot' / '__init__.py').is_file():
            names.append(child.name)
    return names


kernels = []
operators = []
affine_kernels = []

for _family in _discover_families():
    _mod = importlib.import_module(f'{_family}.aot')
    kernels.extend(getattr(_mod, 'kernels', []))
    operators.extend(getattr(_mod, 'operators', []))
    affine_kernels.extend(getattr(_mod, 'affine_kernels', []))
