# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The root `aotriton` distribution ships aotriton.pgdef, with its pq
schema.sql file, and does not ship aotriton.tune (its own `aotriton-tune`
distribution)."""

import pathlib
import subprocess
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[3]


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    if not (_REPO / 'setup.py').is_file():
        pytest.skip('not a source checkout')
    out = tmp_path_factory.mktemp('build_lib')
    proc = subprocess.run(
        [sys.executable, 'setup.py', '-q', 'build_py', '--build-lib', str(out)],
        cwd=_REPO, capture_output=True, text=True,
        env={**__import__('os').environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    if proc.returncode != 0:
        pytest.skip(f'setuptools build_py unavailable: {proc.stderr[-300:]}')
    return out / 'aotriton'


def test_pgdef_is_shipped_with_their_schema_files(built):
    for pkg in ('pgdef', 'pgdef/pq', 'pgdef/localq'):
        assert (built / pkg / '__init__.py').is_file(), pkg
    assert (built / 'pgdef/pq/schema.sql').is_file()
    assert (built / 'pgdef/manifest.py').is_file()


def test_tune_stays_out_of_the_root_distribution(built):
    assert not (built / 'tune').exists()


def test_the_infrastructure_helpers_are_shipped(built):
    assert (built / 'pgdef/workdir.py').is_file()
