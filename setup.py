# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Maps the python/ directory to the importable top-level `aotriton` package:
#   python/__init__.py        -> aotriton
#   python/codegen/*          -> aotriton.codegen
#   python/template_instantiation/* -> aotriton.template_instantiation
#   ... etc, for every sub-package discovered under python/.
#
# The version is parsed from CMakeLists.txt, which is the single source of truth
# for the AOTriton version (set(AOTRITON_VERSION_{MAJOR,MINOR,PATCH}_INT ...)).
# Never hard-code the version here — keep CMake authoritative.

import re
import pathlib

from setuptools import setup, find_packages

_ROOT = pathlib.Path(__file__).resolve().parent
_PYDIR = _ROOT / 'python'


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


# Sub-packages under python/ (e.g. 'codegen', 'template_instantiation.compat').
# modules/ holds user application source code (Triton kernels, AOT artifacts)
# compiled BY aotriton — a compiler does not ship the application it compiles.
_subs = find_packages(where=str(_PYDIR), exclude=['modules', 'modules.*'])  # exclude is technically redundant, but to stop complains from /code-review
_packages = ['aotriton'] + [f'aotriton.{p}' for p in _subs]
_package_dir = {'aotriton': 'python'}
_package_dir.update({f'aotriton.{p}': 'python/' + p.replace('.', '/')
                     for p in _subs})

# Non-.py runtime data shipped inside the package: the codegen Jinja/C++ templates
# (codegen/template/**.{cc,h}). Required for a NON-editable install — they are read at
# runtime via open() (see codegen/template.py), so they must be copied into the
# installed package, not just left in the source tree.
_package_data = {
    'aotriton.codegen': [
        'template/*.cc', 'template/*.h',
        'template/snippet/*.cc', 'template/snippet/*.h',
    ],
}

setup(
    version=_aotriton_version(),
    packages=_packages,
    package_dir=_package_dir,
    package_data=_package_data,
    include_package_data=True,
)
