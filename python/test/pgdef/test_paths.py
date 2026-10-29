# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""aotriton.pgdef.paths and .tune/lib/paths.sh: every filesystem location a
node assumes, defined in one place on each side.

Three checks:

* each function's default, and its environment override;
* the shell library gives the same values as the Python module;
* no other file under python/ or .tune/ spells the workdir or tmpfs literal,
  like test_layering does for imports.
"""

import os
import pathlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from aotriton.pgdef import paths

_REPO = pathlib.Path(__file__).resolve().parents[3]
_PATHS_SH = _REPO / '.tune' / 'lib' / 'paths.sh'

VARIABLES = ('AOTRITON_WORKDIR',
             'AOTRITON_SRC',
             'AOTRITON_RUNDIR',
             'AOTRITON_TMPFS',
             'AOTRITON_BROKER_SOCKET')

FUNCTIONS = {'AOTRITON_WORKDIR': paths.workdir,
             'AOTRITON_SRC': paths.src_dir,
             'AOTRITON_RUNDIR': paths.rundir,
             'AOTRITON_TMPFS': paths.tmpfs,
             'AOTRITON_BROKER_SOCKET': paths.broker_socket}


@pytest.fixture
def clean_env(monkeypatch):
    for name in VARIABLES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_defaults(clean_env):
    assert paths.workdir() == Path('/wkdir')
    assert paths.src_dir() == Path('/wkdir/aotriton.src')
    assert paths.rundir() == Path('/wkdir/run')
    assert paths.tmpfs() == Path('/dev/shm')
    assert paths.broker_socket() == Path('/tmp/aotriton-broker.sock')


def test_each_variable_overrides_its_function(clean_env):
    for name, function in FUNCTIONS.items():
        clean_env.setenv(name, f'/x/{name}')
        assert function() == Path(f'/x/{name}')


def test_workdir_moves_the_derived_defaults(clean_env):
    clean_env.setenv('AOTRITON_WORKDIR', '/scratch/wd')
    assert paths.workdir() == Path('/scratch/wd')
    assert paths.src_dir() == Path('/scratch/wd/aotriton.src')
    assert paths.rundir() == Path('/scratch/wd/run')
    # Not derived from the workdir
    assert paths.tmpfs() == Path('/dev/shm')
    assert paths.broker_socket() == Path('/tmp/aotriton-broker.sock')


def test_explicit_values_win_over_derived_ones(clean_env):
    clean_env.setenv('AOTRITON_WORKDIR', '/scratch/wd')
    clean_env.setenv('AOTRITON_SRC', '/home/u/aotriton')
    clean_env.setenv('AOTRITON_RUNDIR', '/run/user/1000/aotriton')
    assert paths.src_dir() == Path('/home/u/aotriton')
    assert paths.rundir() == Path('/run/user/1000/aotriton')


def test_read_at_call_time_and_empty_counts_as_unset(clean_env):
    clean_env.setenv('AOTRITON_TMPFS', '/a')
    assert paths.tmpfs() == Path('/a')
    clean_env.setenv('AOTRITON_TMPFS', '/b')
    assert paths.tmpfs() == Path('/b')
    clean_env.setenv('AOTRITON_TMPFS', '')
    assert paths.tmpfs() == Path('/dev/shm')


def _shell_values(env: dict[str, str]) -> dict[str, str]:
    script = f'. {_PATHS_SH} && ' + ' && '.join(f'echo "{name}=${name}"' for name in VARIABLES)
    out = subprocess.run(['bash', '-c', script],
                         env=env,
                         check=True,
                         capture_output=True,
                         text=True).stdout
    return dict(line.split('=', 1) for line in out.splitlines())


@pytest.mark.skipif(shutil.which('bash') is None, reason='needs bash')
@pytest.mark.parametrize('overrides', [{},
                                       {'AOTRITON_WORKDIR': '/scratch/wd'},
                                       {'AOTRITON_WORKDIR': '/scratch/wd',
                                        'AOTRITON_RUNDIR': '/tmp/node-rundir',
                                        'AOTRITON_TMPFS': '/tmp/fast',
                                        'AOTRITON_BROKER_SOCKET': '/tmp/node-rundir/b.sock'}])
def test_shell_library_agrees_with_python(clean_env, overrides):
    env = {k: v for k, v in os.environ.items() if k not in VARIABLES}
    env.update(overrides)
    for name, value in overrides.items():
        clean_env.setenv(name, value)
    shell = _shell_values(env)
    for name, function in FUNCTIONS.items():
        assert shell[name] == str(function()), name


# Literals that only .tune/lib/paths.sh and python/pgdef/paths.py may spell.
# `~/wkdir...` and `/path/to/wkdir` in usage examples are host paths, not the
# image's workdir, hence the look-behind.
LITERALS = re.compile(r'(?<![\w~])/wkdir\b|/dev/shm\b')

SCANNED = ('python', '.tune')

ALLOWED = {_REPO / 'python' / 'pgdef' / 'paths.py',
           _PATHS_SH,
           pathlib.Path(__file__).resolve()}

EXCLUDED_DIRS = {
    # An independent pytest plugin, distributed on its own: its /dev/shm is the
    # default of its own lock file, not a location of this project.
    _REPO / 'python' / 'pytest-gpu-lease',
    # Recorded testrun REPL sessions (data, not code): the paths in them are
    # whatever the user typed.
    _REPO / 'python' / 'tune' / '.examples',
}


def _scanned_files():
    # Resolved and deduplicated: .tune/lib/pkgroot/aotriton links back to python/.
    seen = set()
    for top in SCANNED:
        for path in sorted((_REPO / top).rglob('*')):
            path = path.resolve()
            if path in seen or not path.is_file() or '__pycache__' in path.parts:
                continue
            seen.add(path)
            if any(path.is_relative_to(d) for d in EXCLUDED_DIRS) or path in ALLOWED:
                continue
            yield path


def test_no_hard_coded_paths_outside_the_paths_modules():
    offenders = []
    for path in _scanned_files():
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if LITERALS.search(line):
                offenders.append(f'{path.relative_to(_REPO)}:{lineno}: {line.strip()}')
    assert not offenders, ('use aotriton.pgdef.paths / .tune/lib/paths.sh instead:\n'
                           + '\n'.join(offenders))
