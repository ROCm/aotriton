# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""`.def/libexec/tunnels.py`, the SSH tunnel manager for firewalled hosts.

Unit tests of its pure parts (diff, back-off, failure classification, status
line format, the ssh command), and one integration test that runs the manager
against a fake `ssh` on PATH and drives a reload through the touch file.
"""

import importlib.util
import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

_TUNNELS = Path(__file__).resolve().parents[3] / '.def' / 'libexec' / 'tunnels.py'
_spec = importlib.util.spec_from_file_location('def_tunnels', _TUNNELS)
tunnels = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tunnels)


def test_diff_hosts():
    assert tunnels.diff_hosts({'a', 'b'}, {'b', 'c'}) == (['a'], ['c'])
    assert tunnels.diff_hosts(set(), set()) == ([], [])
    assert tunnels.diff_hosts({'a'}, {'a'}) == ([], [])


def test_backoff_is_exponential_and_capped():
    delays = [tunnels.backoff_delay(n) for n in range(0, 9)]
    assert delays[0] == 0.0
    assert delays[1:6] == [2.0, 4.0, 8.0, 16.0, 32.0]
    assert delays[6:] == [60.0, 60.0, 60.0]


@pytest.mark.parametrize('stderr,category,words', [
    ('user@h: Permission denied (publickey,password).', 'auth', ['ssh-agent', 'ssh-add']),
    ('Host key verification failed.', 'auth', ['non-interactive']),
    ('Warning: remote port forwarding failed for listen port 6789', 'forward',
     ['AllowTcpForwarding', 'GatewayPorts', 'SELinux', 'rocm-250']),
    ('ssh: connect to host h port 22: Connection refused', 'other', []),
    ('', 'other', []),
])
def test_classify_failure(stderr, category, words):
    got, hint = tunnels.classify_failure(stderr, 'rocm-250')
    assert got == category
    for w in words:
        assert w in hint
    if category == 'other':
        assert hint == ''


def test_ssh_command_has_one_forward_per_service():
    cmd = tunnels.ssh_command('h1', [('postgresql', 6789)])
    assert cmd[:2] == ['ssh', '-N']
    for option in ('ExitOnForwardFailure=yes', 'ServerAliveInterval=15', 'ServerAliveCountMax=3', 'BatchMode=yes'):
        assert option in cmd
    assert cmd[-3:] == ['-R', f'{tunnels.BIND_ADDRESS}:6789:127.0.0.1:6789', 'h1'][-3:]
    assert cmd.count('-R') == 1
    two = tunnels.ssh_command('h1', [('postgresql', 6789), ('other', 7000)])
    assert two.count('-R') == 2 and two[-1] == 'h1'


def test_forwards_for_defaults_the_port():
    assert tunnels.forwards_for({'POSTGRES_PORT': '6789'}) == [('postgresql', 6789)]
    assert tunnels.forwards_for({'POSTGRES_PORT': ''}) == [('postgresql', 5432)]


def test_status_format():
    line = tunnels.format_status_line('h1', [('postgresql', 6789)], 'up', 0, 'note')
    host, services, state, since, note = line.split('\t')
    assert (host, services, state, note) == ('h1', 'postgresql:6789', 'up', 'note')
    assert len(since) == len('2026-11-02T22:00:00')
    text = tunnels.format_status([line], 0, 42)
    assert text.splitlines()[0].startswith('# ssh tunnels (manager pid 42), reloaded ')
    assert text.splitlines()[-1] == line
    assert '(no firewalled hosts)' in tunnels.format_status([], 0, 42)


def _make_workdir(path, hosts, port='6789'):
    path.mkdir(parents=True, exist_ok=True)
    (path / 'config.rc').write_text(f'CELERY_SERVICE_HOST=server.example\nPOSTGRES_PORT={port}\n')
    with sqlite3.connect(path / 'workers.db') as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        conn.execute('DELETE FROM config')
        conn.executemany('INSERT INTO config (key, value) VALUES (?, ?)', hosts)


def test_read_config_and_hosts(tmp_path):
    _make_workdir(tmp_path,
                  [('firewalled::a', '1'), ('firewalled::b', '0'), ('thermal_gate::c', '1'),
                   ('buildnode::hostname', 'd')])
    assert tunnels.read_config(tmp_path)['POSTGRES_PORT'] == '6789'
    assert tunnels.read_firewalled_hosts(tmp_path) == {'a'}
    assert tunnels.read_firewalled_hosts(tmp_path / 'missing') == set()


_FAKE_SSH = """#!/bin/bash
# Fake ssh: behaviour by the last argument (the host).
echo "$@" >> "$FAKE_SSH_LOG"
for host; do :; done
case "$host" in
  deny*) echo "$host: Permission denied (publickey)." >&2; exit 255 ;;
  nofwd*) echo "Error: remote port forwarding failed for listen port 6789" >&2; exit 255 ;;
  *) exec sleep 60 ;;
esac
"""


def _wait_for(path, predicate, timeout=15.0):
    deadline = time.time() + timeout
    text = ''
    while time.time() < deadline:
        try:
            text = path.read_text()
        except OSError:
            text = ''
        if predicate(text):
            return text
        time.sleep(0.05)
    raise AssertionError(f'timeout; last content of {path}:\n{text}')


def _row(text, host):
    for line in text.splitlines():
        if line.startswith(host + '\t'):
            return line.split('\t')
    return None


def test_manager_with_fake_ssh(tmp_path):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    (bin_dir / 'ssh').write_text(_FAKE_SSH)
    (bin_dir / 'ssh').chmod(0o755)
    workdir = tmp_path / 'wk'
    _make_workdir(workdir, [('firewalled::good1', '1'), ('firewalled::deny1', '1')])
    status = workdir / 'run' / 'ssh-tunnels.status'
    touch = workdir / 'run' / 'touch-me-to-refresh-ssh-tunnel'
    env = dict(os.environ, PATH=f'{bin_dir}:{os.environ["PATH"]}', FAKE_SSH_LOG=str(tmp_path / 'ssh.log'))
    proc = subprocess.Popen([sys.executable, str(_TUNNELS), str(workdir), '--poll_interval', '0.1', '--up_after', '0.3'],
                            env=env)
    try:
        # Start: a full reload without any touch.
        text = _wait_for(status, lambda t: (_row(t, 'good1') or [0, 0, ''])[2] == 'up' and _row(t, 'deny1'))
        deny = _row(text, 'deny1')
        assert deny[1] == 'postgresql:6789' and deny[2] == 'restarting'
        assert 'Permission denied' in deny[4] and 'ssh-agent' in deny[4]
        assert '-R 127.0.0.1:6789:127.0.0.1:6789 good1' in (tmp_path / 'ssh.log').read_text()

        # Reload through the touch file: drop good1 and deny1, add nofwd1.
        _make_workdir(workdir, [('firewalled::nofwd1', '1'), ('firewalled::good1', '0')])
        touch.touch()
        os.utime(touch, (time.time() + 10, time.time() + 10))
        text = _wait_for(status, lambda t: 'SELinux' in t and not _row(t, 'good1') and not _row(t, 'deny1'))
        nofwd = _row(text, 'nofwd1')
        assert nofwd[2] == 'restarting' and 'SELinux' in nofwd[4]
        assert 'reloaded' in text.splitlines()[0]
    finally:
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=15) == 0
    final = status.read_text()
    assert _row(final, 'nofwd1')[2] == 'down'
    assert 'starting: ssh -N' in (workdir / 'run' / 'ssh-tunnels.log').read_text()
