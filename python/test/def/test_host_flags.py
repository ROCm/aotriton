# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Per-host settings (thermal gate) are rows of workers.db's config table,
`thermal_gate::<hostname>` = 1, not columns of the workers table.

Flask is not needed: the webui modules are imported against a stub `flask`,
and the accessors and the route function are called directly.
"""

import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

_REPO = Path(__file__).resolve().parents[3]
_DB_QUERY_SH = _REPO / '.def' / 'lib' / 'db_query.sh'
_OLD_SCHEMA = """
CREATE TABLE workers (id INTEGER PRIMARY KEY AUTOINCREMENT, hostname TEXT NOT NULL UNIQUE,
                      arch TEXT NOT NULL, workdir_override TEXT, subpool TEXT,
                      thermal_gate INTEGER NOT NULL DEFAULT 0,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT NOT NULL,
                     updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
"""


class _Blueprint:
    def __init__(self, *args, **kwargs):
        pass

    def route(self, *args, **kwargs):
        return lambda f: f

    def before_request(self, f):
        return f

    def __getattr__(self, name):
        return lambda f: f


@pytest.fixture
def webui(monkeypatch):
    """(tasks, routes) of the webui, imported against a stub flask."""
    stub = mock.MagicMock()
    stub.Blueprint = _Blueprint
    stub.jsonify = lambda x: x
    monkeypatch.setitem(sys.modules, 'flask', stub)
    monkeypatch.syspath_prepend(str(_REPO / '.def'))
    try:
        import webui.tasks as tasks
        import webui.routes as routes
    except ImportError as e:
        pytest.skip(f'webui is not importable here: {e}')
    yield tasks, routes
    for name in [n for n in sys.modules if n == 'webui' or n.startswith('webui.')]:
        del sys.modules[name]


def _workers(tasks, workdir, *hosts):
    tasks.init_workers_db(workdir)
    with sqlite3.connect(workdir / 'workers.db') as conn:
        for h in hosts:
            conn.execute("INSERT INTO workers (hostname, arch) VALUES (?, 'gfx942')", (h,))


def test_fresh_schema_has_no_thermal_gate_column(webui, tmp_path):
    tasks, _ = webui
    tasks.init_workers_db(tmp_path)
    with sqlite3.connect(tmp_path / 'workers.db') as conn:
        assert 'thermal_gate' not in [r[1] for r in conn.execute("PRAGMA table_info(workers)")]


def test_thermal_gate_rows(webui, tmp_path):
    tasks, _ = webui
    _workers(tasks, tmp_path, 'a', 'b')
    assert tasks.get_worker_thermal_gate(tmp_path, 'a') is False
    assert tasks.set_worker_thermal_gate(tmp_path, 'a', True)['success']
    assert tasks.get_worker_thermal_gate(tmp_path, 'a') is True
    assert tasks.get_worker_thermal_gates(tmp_path) == {'a': True}
    with sqlite3.connect(tmp_path / 'workers.db') as conn:
        assert conn.execute("SELECT value FROM config WHERE key = 'thermal_gate::a'").fetchone() == ('1',)
    assert tasks.set_worker_thermal_gate(tmp_path, 'a', False)['success']
    assert tasks.get_worker_thermal_gate(tmp_path, 'a') is False
    assert not tasks.set_worker_thermal_gate(tmp_path, 'nope', True)['success']


def test_thermal_gate_route(webui, tmp_path):
    tasks, routes = webui
    _workers(tasks, tmp_path, 'a')
    routes.current_app = SimpleNamespace(config={'WORKDIR': str(tmp_path)})
    routes.request = SimpleNamespace(form={'enabled': 'true'})
    assert routes.api_save_thermal_gate('a')['success']
    assert tasks.get_worker_thermal_gate(tmp_path, 'a') is True


def test_migration_moves_column_to_rows(webui, tmp_path):
    tasks, _ = webui
    with sqlite3.connect(tmp_path / 'workers.db') as conn:
        conn.executescript(_OLD_SCHEMA)
        conn.executemany("INSERT INTO workers (hostname, arch, thermal_gate) VALUES (?, 'gfx942', ?)",
                         [('on1', 1), ('off1', 0), ('on2', 1)])
        # An explicit row wins over the column.
        conn.execute("INSERT INTO config (key, value) VALUES ('thermal_gate::on2', '0')")
    tasks.init_workers_db(tmp_path)
    with sqlite3.connect(tmp_path / 'workers.db') as conn:
        rows = dict(conn.execute("SELECT key, value FROM config WHERE key LIKE 'thermal_gate::%'"))
        columns = [r[1] for r in conn.execute("PRAGMA table_info(workers)")]
    assert rows == {'thermal_gate::on1': '1', 'thermal_gate::on2': '0'}
    assert 'thermal_gate' not in columns
    tasks.init_workers_db(tmp_path)   # idempotent


_SQLITE3_SHIM = """#!{python}
import sqlite3, sys
for row in sqlite3.connect(sys.argv[1]).execute(sys.argv[2]):
    print('|'.join('' if v is None else str(v) for v in row))
"""


@pytest.fixture
def sqlite3_cli(tmp_path, monkeypatch):
    """The sqlite3 command-line tool, or a shim of it (it is not installed everywhere)."""
    if shutil.which('sqlite3') is None:
        shim = tmp_path / 'shimbin' / 'sqlite3'
        shim.parent.mkdir()
        shim.write_text(_SQLITE3_SHIM.format(python=sys.executable))
        shim.chmod(0o755)
        monkeypatch.setenv('PATH', f'{shim.parent}:/usr/bin:/bin')


def test_shell_helper(webui, tmp_path, sqlite3_cli):
    tasks, _ = webui
    _workers(tasks, tmp_path, 'a', 'b')
    tasks.set_worker_thermal_gate(tmp_path, 'a', True)

    def helper(host):
        return subprocess.run(['bash', '-c', f'. {_DB_QUERY_SH}; get_worker_thermal_gate {tmp_path} {host}'],
                              capture_output=True,
                              text=True,
                              check=True).stdout.strip()

    assert [helper('a'), helper('b'), helper('unknown')] == ['on', 'off', 'off']


def test_firewalled_rows_and_touch_file(webui, tmp_path, sqlite3_cli):
    tasks, routes = webui
    _workers(tasks, tmp_path, 'a', 'b')
    touch = tmp_path / 'run' / 'touch-me-to-refresh-ssh-tunnel'
    assert tasks.get_worker_firewalled(tmp_path, 'a') is False
    assert not touch.exists()

    routes.current_app = SimpleNamespace(config={'WORKDIR': str(tmp_path)})
    routes.request = SimpleNamespace(form={'enabled': 'true'})
    assert routes.api_save_firewalled('a')['success']
    assert tasks.get_worker_firewalleds(tmp_path) == {'a': True}
    assert touch.exists()
    # Independent of the thermal gate.
    assert tasks.get_worker_thermal_gates(tmp_path) == {}

    touch.unlink()
    assert not tasks.set_worker_firewalled(tmp_path, 'nope', True)['success']
    assert not touch.exists()

    def helper(host):
        return subprocess.run(['bash', '-c', f'. {_DB_QUERY_SH}; get_worker_firewalled {tmp_path} {host}'],
                              capture_output=True,
                              text=True,
                              check=True).stdout.strip()

    assert [helper('a'), helper('b'), helper('unknown')] == ['1', '0', '0']


def test_workers_by_arch_carries_the_flags(webui, tmp_path):
    tasks, _ = webui
    _workers(tasks, tmp_path, 'a')
    tasks.set_worker_firewalled(tmp_path, 'a', True)
    workers = tasks.get_workers_by_architecture(tmp_path)['gfx942']
    assert [(w['hostname'], w['firewalled'], w['thermal_gate']) for w in workers] == [('a', True, False)]


def test_ssh_tunnel_status_file(webui, tmp_path):
    tasks, _ = webui
    assert tasks.get_ssh_tunnel_status(tmp_path) is None
    (tmp_path / 'run').mkdir()
    (tmp_path / 'run' / 'ssh-tunnels.status').write_text('a up\n')
    assert tasks.get_ssh_tunnel_status(tmp_path) == 'a up\n'


_FAKE_SSH = """#!/bin/bash
# Fake ssh: writes its arguments to $FAKE_SSH_OUT.
printf '%s\\n' "$@" > "$FAKE_SSH_OUT"
"""


def _fake_ssh_run(tmp_path, args, extra_env=None):
    bin_dir = tmp_path / 'fakebin'
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / 'ssh').write_text(_FAKE_SSH)
    (bin_dir / 'ssh').chmod(0o755)
    out = tmp_path / 'ssh.out'
    env = dict(os.environ, PATH=f'{bin_dir}:{os.environ["PATH"]}', FAKE_SSH_OUT=str(out))
    result = subprocess.run(args, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return out.read_text().splitlines()


def _prepare(tasks, tmp_path):
    workdir = tmp_path / 'wk'
    workdir.mkdir()
    (workdir / 'config.rc').write_text('CELERY_SERVICE_HOST=server.example\nPOSTGRES_PORT=6789\n')
    _workers(tasks, workdir, 'fw', 'direct')
    tasks.set_worker_firewalled(workdir, 'fw', True)
    return workdir


@pytest.mark.parametrize('host,expected', [('fw', ['127.0.0.1', '6789']), ('direct', ['', ''])])
def test_node_passes_the_endpoint(webui, tmp_path, sqlite3_cli, host, expected):
    tasks, _ = webui
    workdir = _prepare(tasks, tmp_path)
    lines = _fake_ssh_run(tmp_path,
                          [str(_REPO / '.perfmon' / 'bin' / 'node'), '--workdir', str(workdir), '--host', host,
                           '--rocm', '7.0', '--scratch', str(tmp_path / 'scratch'), '--arch', 'gfx942', 'start'])
    assert lines[0] == host
    remote = shlex.split(lines[1])
    # node_run.sh <workdir> <scratch> <rocm> <thermal_gate> <pghost> <pgport> <command> <arch> ...
    assert remote[1].endswith('node_run.sh')
    assert remote[5:9] == ['off', *expected, 'start']


def test_start_worker_passes_the_endpoint(webui, tmp_path, sqlite3_cli):
    tasks, _ = webui
    workdir = _prepare(tasks, tmp_path)
    with sqlite3.connect(workdir / 'workers.db') as conn:
        conn.execute("INSERT INTO config (key, value) VALUES ('default_workdir', '/w')")
    script = _REPO / '.def' / 'libexec' / 'start_worker.sh'
    for host, expect_env in (('fw', True), ('direct', False)):
        lines = _fake_ssh_run(tmp_path, ['bash', str(script), str(workdir), host, '--', '--category', 'perfmon'])
        assert lines[0] == host
        # bash -s <WORKER_WORKDIR> <arch> <image> <4 paths> <thermal_gate> <pghost> <pgport> <extra args...>
        args = lines[1:]
        pg = args[args.index('off') + 1:args.index('off') + 3]
        assert pg == (['127.0.0.1', '6789'] if expect_env else ['', ''])
