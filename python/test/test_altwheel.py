# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The three venvs.* entry forms of the resolved altwheel config, as read by
aotriton.altwheel (CMake's reader) and RootGenerator (the generator's)."""

import re
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from aotriton import altwheel
from aotriton.codegen.root import REL_PYTHON, RootGenerator

WHL = '/w/triton-3.4.0-cp313-cp313-linux_x86_64.whl'
MARKER = 'triton==3.9.0; python_version >= "3.10"'


@pytest.mark.parametrize('raw, mode, value', [
    (WHL, 'wheel', WHL),
    ('rel/triton-3.4.0-cp313-cp313-linux_x86_64.whl', 'wheel', 'rel/triton-3.4.0-cp313-cp313-linux_x86_64.whl'),
    ('python:/opt/venv/bin/python', 'python', '/opt/venv/bin/python'),
    ('pep508:triton>=3.9', 'pep508', 'triton>=3.9'),
    (f'pep508:{MARKER}', 'pep508', MARKER),
    ('pep508:triton @ https://h/triton-3.9.0-cp313-cp313-linux_x86_64.whl',
     'pep508',
     'triton @ https://h/triton-3.9.0-cp313-cp313-linux_x86_64.whl'),
])
def test_entry_forms(raw, mode, value):
    entry = altwheel.parse_entry('v', raw)
    assert (entry.mode, entry.value) == (mode, value)


@pytest.mark.parametrize('name, raw, message', [
    ('v', 'triton==3.9.0', 'pep508:triton==3.9.0'),
    ('v', '/w/triton.tar.gz', '.whl'),
    ('v', 'python:bin/python', 'absolute'),
    ('v', 'pep508:/w/x.whl', 'PEP 508'),
    ('v', 'pep508:', 'PEP 508'),
    ('v', ['a.whl'], 'expected a string'),
    ('a/b', WHL, 'name'),
])
def test_entry_errors(name, raw, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        altwheel.parse_entry(name, raw)


def _config(tmp_path, venvs):
    path = tmp_path / 'altwheel.yaml'
    path.write_text('venvs:\n' + ''.join(f"  {k}: '{v}'\n" for k, v in venvs.items()))
    return path


def _cli(*args):
    return subprocess.run([sys.executable, '-m', 'aotriton.altwheel', *map(str, args)],
                          capture_output=True,
                          text=True)


def test_cli(tmp_path):
    path = _config(tmp_path, {'default': WHL, 'm': f'pep508:{MARKER}'})
    assert _cli(path).stdout == 'default;m'
    assert _cli(path, '--venv', 'm', '--field', 'mode').stdout == 'pep508'
    assert _cli(path, '--venv', 'm', '--field', 'value').stdout == MARKER
    missing = _cli(path, '--venv', 'nope')
    assert missing.returncode != 0 and 'venvs.nope' in missing.stderr
    bad = _cli(_config(tmp_path, {'default': 'triton==3.9.0'}))
    assert bad.returncode != 0 and 'pep508:triton==3.9.0' in bad.stderr


def _load(tmp_path, monkeypatch, venvs):
    monkeypatch.setenv('VIRTUAL_ENV', str(tmp_path / 'venv'))
    gen = object.__new__(RootGenerator)
    gen._args = Namespace(build_dir=tmp_path / 'build' / 'v3src')
    gen._load_altwheel_config({'venvs': venvs})
    return gen


def test_root_mapping(tmp_path, monkeypatch):
    gen = _load(tmp_path,
                monkeypatch,
                {'default': 'python:/opt/py/bin/python',
                 'w': WHL,
                 'p': 'pep508:triton>=3.9'})
    main = tmp_path / 'venv' / REL_PYTHON
    altvenvs = tmp_path / 'build' / 'altvenvs'
    assert gen._venvpython == {'default': Path('/opt/py/bin/python'),
                               'w': altvenvs / 'w' / REL_PYTHON,
                               'p': altvenvs / 'p' / REL_PYTHON}
    # flyc compiles need flydsl + aotriton: always the main venv.
    assert gen._main_python == main


def test_root_default_falls_back_to_main_venv(tmp_path, monkeypatch):
    gen = _load(tmp_path, monkeypatch, {'w': WHL})
    assert gen._venvpython['default'] == gen._main_python == tmp_path / 'venv' / REL_PYTHON


def test_root_rejects_bare_pin(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='pep508:'):
        _load(tmp_path, monkeypatch, {'default': 'triton==3.9.0'})
