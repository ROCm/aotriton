# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Maps python/perfmon/ to the `aotriton.perfmon` namespace-package subtree:
#   python/perfmon/__init__.py -> aotriton.perfmon, python/perfmon/pq/* -> aotriton.perfmon.pq,
#   ... etc, for every sub-package discovered under python/perfmon/.
#
# Version parsed from the repo-root CMakeLists.txt, same source of truth as
# the main distribution's setup.py.

import re
import pathlib

from setuptools import setup, find_packages

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PERFMONDIR = _ROOT / 'python' / 'perfmon'


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


_subs = find_packages(where=str(_PERFMONDIR))
_packages = ['aotriton.perfmon'] + [f'aotriton.perfmon.{p}' for p in _subs]
# package_dir is resolved relative to THIS distribution's own root
# (python/perfmon/), not the repo root -- `_ROOT` is only used for CMakeLists.txt.
_package_dir = {'aotriton.perfmon': '.'}
_package_dir.update({f'aotriton.perfmon.{p}': p.replace('.', '/')
                     for p in _subs})

# schema.sql is applied by .def/bin/initdb straight from a checkout, and also
# read from the installed package.
_package_data = {
    'aotriton.perfmon.pq': ['*.sql'],
}

setup(
    version=_aotriton_version(),
    packages=_packages,
    package_dir=_package_dir,
    package_data=_package_data,
    include_package_data=True,
)
