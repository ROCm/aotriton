# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# `aotriton` is a PEP 420 implicit namespace package: this distribution ships
# every `aotriton.<sub>` subpackage found under python/ EXCEPT `tune`/`tune.*`,
# and ships no `aotriton/__init__.py` of its own. Maps:
#   python/codegen/*          -> aotriton.codegen
#   python/template_instantiation/* -> aotriton.template_instantiation
#   ... etc, for every sub-package discovered under python/ (excluding tune).
#
# python/tune/ is a separate distribution (see python/tune/setup.py, package
# name `aotriton-tune`) that merges into the same `aotriton` namespace at
# import time. It is split out so that the main `aotriton` wheel — the one
# the code generator depends on — never bundles tuning infrastructure (and
# its Unix-only dependencies like fcntl); pip extras cannot gate wheel
# contents, only extra dependencies, so a separate distribution is the only
# way to make it truly optional.
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
# tune/tune.* is excluded because it is a separate distribution (see
# python/tune/setup.py) shipped as `aotriton-tune`, not bundled here.
#
# Must stay find_packages(), never find_namespace_packages(): the latter does not
# require __init__.py to descend into a directory, and would sweep the standalone
# python/pytest-gpu-lease/ project (its own pyproject.toml, no __init__.py) into
# the aotriton wheel.
_subs = find_packages(where=str(_PYDIR), exclude=['modules', 'modules.*', 'tune', 'tune.*'])  # 'modules' exclude is technically redundant (no __init__.py there); 'tune' exclude is NOT -- python/tune/ has __init__.py files and would otherwise be swept in here
_packages = [f'aotriton.{p}' for p in _subs]
_package_dir = {f'aotriton.{p}': 'python/' + p.replace('.', '/')
                for p in _subs}

# Loose top-level .py files directly under python/ (gpu_targets.py,
# generate.py, flyc_compile.py, ...) are NOT sub-packages (no __init__.py, so
# find_packages() above never sees them), but they are still part of the
# `aotriton` namespace and load-bearing: e.g. codegen/root.py shells out to
# `python -m aotriton.flyc_compile`, and functional.py does
# `from aotriton.gpu_targets import ...`. Before this distribution shipped a
# real `aotriton/__init__.py` (package_dir={'aotriton': 'python'}), so every
# .py directly in python/ was automatically an `aotriton.<name>` submodule for
# free; now that `aotriton` is a namespace package (no __init__.py), each one
# needs an explicit py_modules entry, with its own package_dir key so
# setuptools can still resolve 'aotriton.<name>' -> 'python/<name>.py'.
# 'aotriton' is deliberately NOT added to `packages` above: that would ship an
# __init__.py-bearing package artifact for it, unmaking the namespace package.
_top_level_modules = sorted(p.stem for p in _PYDIR.glob('*.py'))
_py_modules = [f'aotriton.{m}' for m in _top_level_modules]
_package_dir['aotriton'] = str(_PYDIR.relative_to(_ROOT))

# Non-.py runtime data shipped inside the package: the codegen Jinja/C++ templates
# (codegen/template/**.{cc,h}). Required for a NON-editable install — they are read at
# runtime via open() (see codegen/template.py), so they must be copied into the
# installed package, not just left in the source tree. Declared explicitly here,
# so `include_package_data` (below) does not need to be True for these to ship.
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
    py_modules=_py_modules,
    package_data=_package_data,
    # MUST stay False. `package_dir['aotriton']` above points at the whole
    # `python/` directory (needed only so setuptools can resolve the loose
    # top-level `py_modules` entries) -- `python/` also physically contains
    # `python/tune/`, the separate `aotriton-tune` distribution this one must
    # NOT bundle. include_package_data=True walks that entire directory for
    # data files with no regard for `packages`/`py_modules`, and would ship
    # every `python/tune/**` file as inert (but still importable) package
    # data, silently undoing the whole split. The one legitimate data need
    # (codegen's template/*.cc,*.h) is covered by the explicit
    # `package_data` dict above, which works independently of this flag.
    include_package_data=False,
)
