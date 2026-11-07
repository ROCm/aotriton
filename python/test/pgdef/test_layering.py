# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Layering guard for aotriton.pgdef.

pgdef is a library that applications (aotriton.tune, and later applications) call.
It must never import an application, torch or triton, so that an application
image can ship pgdef without the other applications' dependencies, and so that
adding an application needs zero edits to the engine.

Two checks, because each catches what the other misses:

* a static scan of every pgdef source file's import statements (catches a
  lazy `import torch` buried in a function that nothing executes here);
* a fresh interpreter that imports every pgdef module and then looks at
  sys.modules (catches an import hidden behind something a scan cannot read).
"""

import ast
import pathlib
import subprocess
import sys

import pytest

_PYTHON = pathlib.Path(__file__).resolve().parents[2]       # python/
_PGDEF = _PYTHON / 'pgdef'

FORBIDDEN = ('aotriton.tune', 'aotriton.perfmon', 'torch', 'triton')


def _is_forbidden(name: str) -> bool:
    return any(name == f or name.startswith(f + '.') for f in FORBIDDEN)


def _module_name(path: pathlib.Path) -> str:
    rel = path.relative_to(_PYTHON).with_suffix('')
    parts = list(rel.parts)
    if parts[-1] == '__init__':
        parts.pop()
    return '.'.join(['aotriton'] + parts)


def _imports(path: pathlib.Path):
    """Yield (lineno, absolute module name) for every import in `path`."""
    pkg = _module_name(path).split('.')
    if path.name != '__init__.py':
        pkg = pkg[:-1]
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg[:len(pkg) - (node.level - 1)]
                prefix = '.'.join(base + ([node.module] if node.module else []))
            else:
                prefix = node.module or ''
            yield node.lineno, prefix
            # `from aotriton import tune` names the forbidden module in the alias
            for alias in node.names:
                yield node.lineno, f'{prefix}.{alias.name}'


def _sources():
    return sorted(p for p in _PGDEF.rglob('*.py') if '__pycache__' not in p.parts)


def test_pgdef_has_sources():
    assert _sources(), f'no pgdef sources found under {_PGDEF}'


@pytest.mark.parametrize('path', _sources(), ids=lambda p: str(p.relative_to(_PGDEF)))
def test_no_forbidden_imports(path):
    bad = [(line, name) for line, name in _imports(path) if _is_forbidden(name)]
    assert not bad, (f'{path.relative_to(_PYTHON)} imports an application or a '
                     f'GPU framework: {bad}')


def test_importing_pgdef_pulls_in_nothing_forbidden():
    mods = [_module_name(p) for p in _sources()]
    code = (
        'import importlib, sys\n'
        f'mods = {mods!r}\n'
        'failed = []\n'
        'for m in mods:\n'
        '    try:\n'
        '        importlib.import_module(m)\n'
        '    except ModuleNotFoundError as e:\n'
        '        failed.append((m, str(e)))\n'
        f'forbidden = {FORBIDDEN!r}\n'
        'bad = sorted(n for n in sys.modules if any(n == f or n.startswith(f + ".") for f in forbidden))\n'
        'print(repr((failed, bad)))\n'
    )
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                         env={**__import__('os').environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    assert out.returncode == 0, out.stderr
    failed, bad = eval(out.stdout.strip().splitlines()[-1])
    assert not bad, f'importing aotriton.pgdef loaded {bad}'
    # A module that cannot import for want of a third-party package (psycopg is
    # a declared engine dependency) would hide the check above, so list them.
    assert not failed, f'pgdef modules that failed to import: {failed}'


_CORE = ('pq', 'localq')


@pytest.mark.parametrize('path', [p for p in _sources()
                                  if p.relative_to(_PGDEF).parts[0] in _CORE],
                         ids=lambda p: str(p.relative_to(_PYTHON)))
def test_engine_core_does_not_import_the_workdir_layer(path):
    """pq and localq read only PG* variables. config.rc and workers.db are the
    deployment layer's business (aotriton.pgdef.workdir)."""
    bad = [(line, name) for line, name in _imports(path)
           if name == 'aotriton.pgdef.workdir' or name.startswith('aotriton.pgdef.workdir.')]
    assert not bad, f'{path}: engine core imports the workdir layer: {bad}'


# ---------------------------------------------------------------------------
# aotriton.perfmon: an application, a sibling of aotriton.tune
# ---------------------------------------------------------------------------

_PERFMON = _PYTHON / 'perfmon'
_FAMILY_PERFMON = _PYTHON.parent / 'modules' / 'flash' / 'perfmon'

#: perfmon is built on pgdef, never on the other application.
PERFMON_FORBIDDEN = ('aotriton.tune',)
#: A family's perfmon block runs on control hosts with none of these, and
#: describes its operators without the engine: it sees only
#: aotriton.perfmon.pdesc and aotriton.utils.
FAMILY_FORBIDDEN = ('aotriton.tune', 'aotriton.pgdef', 'torch', 'triton')


def _matches(name: str, forbidden) -> bool:
    return any(name == f or name.startswith(f + '.') for f in forbidden)


def _perfmon_sources():
    return sorted(p for p in _PERFMON.rglob('*.py') if '__pycache__' not in p.parts)


def _family_sources():
    return sorted(p for p in _FAMILY_PERFMON.rglob('*.py')
                  if '__pycache__' not in p.parts and 'runner' not in p.relative_to(_FAMILY_PERFMON).parts)


def _absolute_imports(path: pathlib.Path):
    """(lineno, name) of every import in a file outside the aotriton package
    tree. A relative import that leaves the block (`from .. import x`) is
    reported as '..'."""
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level > 1:
                yield node.lineno, '..'
            elif node.level == 0:
                yield node.lineno, node.module or ''
                for alias in node.names:
                    yield node.lineno, f'{node.module}.{alias.name}'


def test_perfmon_has_sources():
    assert _perfmon_sources(), f'no perfmon sources found under {_PERFMON}'
    assert _family_sources(), f'no family sources found under {_FAMILY_PERFMON}'


@pytest.mark.parametrize('path', _perfmon_sources(), ids=lambda p: str(p.relative_to(_PERFMON)))
def test_perfmon_does_not_import_tune(path):
    bad = [(line, name) for line, name in _imports(path) if _matches(name, PERFMON_FORBIDDEN)]
    assert not bad, f'{path.relative_to(_PYTHON)} imports the tune application: {bad}'


@pytest.mark.parametrize('path', _family_sources(), ids=lambda p: str(p.relative_to(_FAMILY_PERFMON)))
def test_family_perfmon_block_imports_no_engine_or_gpu_framework(path):
    bad = [(line, name) for line, name in _absolute_imports(path) if _matches(name, FAMILY_FORBIDDEN)]
    assert not bad, f'{path}: the family block imports an engine, an application or a GPU framework: {bad}'


def test_importing_perfmon_pulls_in_no_tune():
    mods = [_module_name(p) for p in _perfmon_sources() if p.name not in ('setup.py', '__main__.py')]
    code = (
        'import importlib, sys\n'
        f'mods = {mods!r}\n'
        'failed = []\n'
        'for m in mods:\n'
        '    try:\n'
        '        importlib.import_module(m)\n'
        '    except ModuleNotFoundError as e:\n'
        '        failed.append((m, str(e)))\n'
        f'forbidden = {PERFMON_FORBIDDEN!r}\n'
        'bad = sorted(n for n in sys.modules if any(n == f or n.startswith(f + ".") for f in forbidden))\n'
        'print(repr((failed, bad)))\n'
    )
    out = subprocess.run([sys.executable, '-c', code],
                         capture_output=True,
                         text=True,
                         env={**__import__('os').environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    assert out.returncode == 0, out.stderr
    failed, bad = eval(out.stdout.strip().splitlines()[-1])
    assert not bad, f'importing aotriton.perfmon loaded {bad}'
    assert not failed, f'perfmon modules that failed to import: {failed}'


def test_loading_the_family_block_through_the_registry_pulls_in_nothing_forbidden():
    """The registry itself uses pgdef's loader, so pgdef is allowed to be
    loaded by then; what the family block adds must stay clear of the
    engine, tune, torch and triton."""
    code = (
        'import sys\n'
        'from aotriton.perfmon.registry import make_perf_desc\n'
        'before = set(sys.modules)\n'
        'desc = make_perf_desc("flash")\n'
        'desc.entry_sets(); desc.llm_cases()\n'
        f'forbidden = {FAMILY_FORBIDDEN!r}\n'
        'added = sorted(n for n in set(sys.modules) - before\n'
        '               if any(n == f or n.startswith(f + ".") for f in forbidden))\n'
        'present = sorted(n for n in ("torch", "triton", "aotriton.tune") if n in sys.modules)\n'
        'print(repr((added, present)))\n'
    )
    out = subprocess.run([sys.executable, '-c', code],
                         capture_output=True,
                         text=True,
                         cwd=_PYTHON.parent,
                         env={**__import__('os').environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    assert out.returncode == 0, out.stderr
    added, present = eval(out.stdout.strip().splitlines()[-1])
    assert not added, f'loading modules/flash/perfmon loaded {added}'
    assert not present, f'{present} are loaded after loading the family block'
