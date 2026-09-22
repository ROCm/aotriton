# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Maps python/tune/ to the `aotriton.tune` namespace-package subtree:
#   python/tune/__init__.py -> aotriton.tune, python/tune/pq/* -> aotriton.tune.pq,
#   ... etc, for every sub-package discovered under python/tune/.
#
# Version parsed from the repo-root CMakeLists.txt, same source of truth as
# the main distribution's setup.py.

import re
import pathlib

from setuptools import setup, find_packages

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_TUNEDIR = _ROOT / 'python' / 'tune'


def _aotriton_version() -> str:
    cml = (_ROOT / 'CMakeLists.txt').read_text()

    def field(name: str) -> str:
        m = re.search(rf'set\(AOTRITON_VERSION_{name}_INT\s+(\d+)\)', cml)
        if m is None:
            raise RuntimeError(
                f'AOTRITON_VERSION_{name}_INT not found in CMakeLists.txt; '
                f'the version must stay defined there (single source of truth)')
        return m.group(1)

    return f"{field('MAJOR')}.{field('MINOR')}.{field('PATCH')}"


# '.examples' is data (sample tuning-tree dumps), not a Python package.
_subs = find_packages(where=str(_TUNEDIR), exclude=['.examples', '.examples.*'])
_packages = ['aotriton.tune'] + [f'aotriton.tune.{p}' for p in _subs]
# package_dir is resolved relative to THIS distribution's own root
# (python/tune/), not the repo root -- `_ROOT` is only used for CMakeLists.txt.
_package_dir = {'aotriton.tune': '.'}
_package_dir.update({f'aotriton.tune.{p}': p.replace('.', '/')
                     for p in _subs})

_package_data = {
    'aotriton.tune.pq': ['*.sql', '*.html'],
}

setup(
    version=_aotriton_version(),
    packages=_packages,
    package_dir=_package_dir,
    package_data=_package_data,
    include_package_data=True,
)
