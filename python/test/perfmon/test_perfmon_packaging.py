# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""aotriton.perfmon is its own distribution (`aotriton-perfmon`), like
aotriton.tune: the root distribution does not ship it, and its own setup.py
ships the package with its schema file."""

import os
import pathlib
import subprocess
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[3]


def _build(cwd: pathlib.Path, out: pathlib.Path):
    command = [sys.executable, 'setup.py', '-q', 'build_py', '--build-lib', str(out)]
    proc = subprocess.run(command,
                          cwd=cwd,
                          capture_output=True,
                          text=True,
                          env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    if proc.returncode != 0:
        pytest.skip(f'setuptools build_py unavailable: {proc.stderr[-300:]}')
    return out / 'aotriton'


@pytest.fixture(scope='module')
def root_build(tmp_path_factory):
    if not (_REPO / 'setup.py').is_file():
        pytest.skip('not a source checkout')
    return _build(_REPO, tmp_path_factory.mktemp('root'))


@pytest.fixture(scope='module')
def perfmon_build(tmp_path_factory):
    if not (_REPO / 'python' / 'perfmon' / 'setup.py').is_file():
        pytest.skip('not a source checkout')
    return _build(_REPO / 'python' / 'perfmon', tmp_path_factory.mktemp('perfmon'))


def test_perfmon_stays_out_of_the_root_distribution(root_build):
    assert not (root_build / 'perfmon').exists()
    assert (root_build / 'pgdef' / '__init__.py').is_file()


def test_the_perfmon_distribution_ships_the_package_and_its_schema(perfmon_build):
    for name in ('__init__.py', 'handlers.py', 'perfdb.py', 'export.py', 'dispatch.py', 'node.py',
                 'manifest.py', 'pq/__init__.py', 'pq/measurements.py', 'pq/schema.sql'):
        assert (perfmon_build / 'perfmon' / name).is_file(), name
    assert not (perfmon_build / '__init__.py').exists()        # `aotriton` stays a namespace package
    assert not (perfmon_build / 'tune').exists()


def test_the_distribution_is_named_aotriton_perfmon():
    text = (_REPO / 'python' / 'perfmon' / 'pyproject.toml').read_text()
    assert 'name = "aotriton-perfmon"' in text
