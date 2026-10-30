# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
perfmon's family registry: loads `modules/<family>/perfmon/` by path, the way
tune loads `modules/<family>/tune/`, and locates the family's perfdb.

The list of families is static: family blocks live outside this package, so
nothing here can discover them by globbing.
"""

from pathlib import Path

from aotriton.pgdef.utils import default_modules_dir, load_family_block

_FAMILIES = ('flash',)

BLOCK = 'perfmon'


def available_module_names() -> list[str]:
    return sorted(_FAMILIES)


def load_family_perfmon(family: str, modules_dir: Path | None = None):
    """The family's perfmon package, which exports `PerfDesc`."""
    if family not in _FAMILIES:
        raise ImportError(f'unknown perfmon family {family!r}; available: {available_module_names()}')
    return load_family_block(family, BLOCK, modules_dir or default_modules_dir())


def make_perf_desc(family: str, modules_dir: Path | None = None):
    return load_family_perfmon(family, modules_dir).PerfDesc()


def default_perfdb_dir(family: str, modules_dir: Path | None = None) -> Path:
    """`modules/<family>/perfdb`, where a family's results are committed."""
    return Path(modules_dir or default_modules_dir()) / family / 'perfdb'
