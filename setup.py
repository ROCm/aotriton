# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# `aotriton` is a PEP 420 implicit namespace package: this distribution ships
# every `aotriton.<sub>` subpackage under python/ EXCEPT `tune`/`tune.*` and
# `perfmon`/`perfmon.*`, and ships no `aotriton/__init__.py` of its own.
#
# python/tune/ and python/perfmon/ are separate distributions
# (python/<app>/setup.py, package names `aotriton-tune` and `aotriton-perfmon`)
# merging into the same namespace at import time -- split out so the main wheel
# (what the code generator depends on) never bundles application infra; pip
# extras can't gate wheel contents, only extra dependencies.
#
# The version is parsed from CMakeLists.txt, the single source of truth
# (set(AOTRITON_VERSION_{MAJOR,MINOR,PATCH}_INT ...)). Never hard-code it here.

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
# modules/ holds user application source (Triton kernels, AOT artifacts)
# compiled BY aotriton, so it's excluded; tune/tune.* and perfmon/perfmon.* are
# excluded because each is a separate distribution (`aotriton-tune`,
# `aotriton-perfmon`; python/<app>/setup.py). pgdef/ IS in
# this distribution: find_packages() picks it up because it has an __init__.py
# (its pq/ subpackage's *.sql file needs the package_data entry below).
#
# Must stay find_packages(), never find_namespace_packages(): the latter
# would sweep the standalone python/pytest-gpu-lease/ project (its own
# pyproject.toml, no __init__.py) into this wheel too.
_subs = find_packages(where=str(_PYDIR), exclude=['modules', 'modules.*', 'tune', 'tune.*', 'perfmon', 'perfmon.*'])
# 'aotriton' itself is listed so that an editable install (pip install -e)
# works: setuptools' editable finder only resolves a mapped package that has an
# __init__.py, or one it knows to be a namespace, and it learns the namespaces
# from `packages`. Listing it ships no aotriton/__init__.py (python/ has none),
# so the regular wheel stays a namespace package with the same files.
_packages = ['aotriton'] + [f'aotriton.{p}' for p in _subs]
_package_dir = {f'aotriton.{p}': 'python/' + p.replace('.', '/')
                for p in _subs}

# Loose top-level .py files directly under python/ (gpu_targets.py,
# generate.py, flyc_compile.py, ...) have no __init__.py, so find_packages()
# never sees them -- but they're load-bearing (e.g. codegen/root.py shells out
# to `python -m aotriton.flyc_compile`). Each needs an explicit py_modules
# entry now that `aotriton` has no `__init__.py` of its own.
_top_level_modules = sorted(p.stem for p in _PYDIR.glob('*.py'))
_py_modules = [f'aotriton.{m}' for m in _top_level_modules]
_package_dir['aotriton'] = str(_PYDIR.relative_to(_ROOT))

# Non-.py runtime data: codegen's Jinja/C++ templates, read via open() at
# runtime (see codegen/template.py), so they must ship with a non-editable
# install. Declared explicitly so `include_package_data` need not be True.
_package_data = {
    'aotriton.codegen': [
        'template/*.cc', 'template/*.h',
        'template/snippet/*.cc', 'template/snippet/*.h',
    ],
    # Applied by .def/bin/initdb straight from a checkout, and also read from
    # the installed package by aotriton.pgdef.pq.
    'aotriton.pgdef.pq': ['*.sql'],
}

setup(
    version=_aotriton_version(),
    packages=_packages,
    package_dir=_package_dir,
    py_modules=_py_modules,
    package_data=_package_data,
    # MUST stay False: `package_dir['aotriton']` points at the whole `python/`
    # directory (for the loose `py_modules` above), which also physically
    # contains `python/tune/` -- the separate distribution this one must NOT
    # bundle. include_package_data=True walks that whole directory regardless
    # of `packages`/`py_modules` and would ship it anyway, undoing the split.
    # The one real data need (codegen's templates) is already covered above.
    include_package_data=False,
)
