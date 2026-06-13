# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The family registry the generator consumes (`aotriton.rules`).

Interim protocol (pre-`--module_dir`): each kernel *family* lives in
`<repo>/modules/<family>/aot`, a python package whose `__init__` exposes
`kernels` / `operators` / `affine_kernels`. This aggregator enumerates the
`modules/*/aot` packages and concatenates their exports.

Each family's `aot` package is loaded by EXPLICIT file path under a synthetic
unique top-level name (`_aotriton_modules_<family>_aot`), NOT as `<family>.aot`.
`modules/<family>` must stay a plain directory (not a package) so that its
`kernel/` sources keep importing each other by bare name; loading `aot` by name
would require `<family>` to be a clean namespace package, which any sys.path entry
containing a `<family>.py` (e.g. `tritonsrc/flash.py`) would shadow. Loading by
path sidesteps that entirely; the package's relative imports (`from .ops`,
`from ._common`) resolve against the synthetic name via its __path__.

`--module_dir` (taking the modules root from the generator CLI instead of the
fixed repo-relative path) is deferred to a later phase; today the repo-relative
`modules/` dir IS the protocol.
"""

import sys
import importlib.util
from pathlib import Path

# <repo>/modules — repo root is the parent of the aotriton package dir (python/).
_MODULES_DIR = Path(__file__).resolve().parent.parent.parent / 'modules'


def _discover_families():
    """Family names = subdirs of modules/ that contain an `aot` package."""
    if not _MODULES_DIR.is_dir():
        return []
    return [child.name for child in sorted(_MODULES_DIR.iterdir())
            if (child / 'aot' / '__init__.py').is_file()]


def _load_family_aot(family):
    """Import modules/<family>/aot/__init__.py by path under a synthetic unique
    package name so its relative imports work without a <family> namespace pkg."""
    modname = f'_aotriton_modules_{family}_aot'
    cached = sys.modules.get(modname)
    if cached is not None:
        return cached
    aot_dir = _MODULES_DIR / family / 'aot'
    spec = importlib.util.spec_from_file_location(
        modname, aot_dir / '__init__.py',
        submodule_search_locations=[str(aot_dir)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def family_aot(family):
    """The loaded `aot` package object for one family (e.g. to reach its support
    classes like `_common.FlashKernel`). Loads it if not already loaded."""
    return _load_family_aot(family)


kernels = []
operators = []
affine_kernels = []

for _family in _discover_families():
    _mod = _load_family_aot(_family)
    kernels.extend(getattr(_mod, 'kernels', []))
    operators.extend(getattr(_mod, 'operators', []))
    affine_kernels.extend(getattr(_mod, 'affine_kernels', []))
