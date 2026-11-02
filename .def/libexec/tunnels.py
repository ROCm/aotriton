#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
SSH tunnel manager for the hosts marked "firewalled" in the workdir.

Usage: tunnels.py <workdir> [--poll_interval SECONDS] [--up_after SECONDS]

A firewalled host cannot reach the server's services (today: PostgreSQL)
directly. For each host with a config row `firewalled::<hostname>` = 1 in
<workdir>/workers.db this keeps one reverse tunnel up,

    ssh -N -R <socket>:127.0.0.1:<port> <hostname>

where <socket> is a Unix socket in the host's own workdir (as the server
knows it: workers.db workdir_override, else default_workdir),
<workdir>/run/pgsql/.s.PGSQL.<port> for PostgreSQL. A process on the host
(a claudepak peer sees the workdir at the same path, a docker worker mounts it)
sets PGHOST=<workdir>/run/pgsql and PGPORT=<port>, and libpq connects to the
socket. Unlike a TCP port on the host's loopback this works from a container
on a Docker bridge. Before each (re)start the stale socket is removed over
ssh, because sshd does not unlink it by itself. Design and assumptions:
perfmon-pgsql-ssh-tunnel.md.

Started and stopped by `srvctl <workdir> start|stop` on the server's bare
metal (pid file <workdir>/run/ssh-tunnels.pid). Standard library only, and it
does not import `aotriton`: it runs with any python3 (3.10+).

  * Reload: whoever changes the firewalled rows touches
    <workdir>/run/touch-me-to-refresh-ssh-tunnel. The main loop polls the
    file's mtime; on a change (and always at start) it re-reads workers.db
    and starts or stops tunnels to match.
  * Supervision: a dead ssh is restarted after an exponential back-off.
  * Status: <workdir>/run/ssh-tunnels.status, rewritten atomically after every
    change; the log is <workdir>/run/ssh-tunnels.log.
  * SIGTERM / SIGINT: stop every ssh, write a final status, exit.
"""

import argparse
import datetime
import os
import re
import shlex
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

# The services forwarded to a firewalled host: (name, config.rc variable
# holding its port, socket directory under <host workdir>/run, socket file name
# for a port). Adding a service adds one `-R` per tunnel. The PostgreSQL
# socket is where libpq looks for it: PGHOST=<directory>, PGPORT=<port>.
SERVICES = (('postgresql', 'POSTGRES_PORT', 'pgsql', '.s.PGSQL.{port}'),)
DEFAULT_PORTS = {'POSTGRES_PORT': 5432}

POLL_INTERVAL = 2.0          # seconds between main loop wakeups
# A Unix socket path must fit in sockaddr_un.sun_path, 108 bytes including the
# terminating NUL, so at most 107 bytes. The path depends on the host's
# workdir; a longer one cannot be bound, so such a tunnel is not started.
MAX_SOCKET_PATH = 107

SSH_PREP_TIMEOUT = 30.0      # seconds for the stale-socket clean-up ssh
UP_AFTER = 3.0               # an ssh alive this long (no exit on a refused forward) counts as up
STABLE_AFTER = 60.0          # an ssh alive this long resets the back-off
BACKOFF_BASE = 2.0
BACKOFF_CAP = 60.0
KILL_GRACE = 5.0             # seconds an ssh gets after SIGTERM

TOUCH_FILE = 'touch-me-to-refresh-ssh-tunnel'
STATUS_FILE = 'ssh-tunnels.status'
LOG_FILE = 'ssh-tunnels.log'

UP, RESTARTING, DOWN = 'up', 'restarting', 'down'

_HOSTNAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]*$')


# ---------------------------------------------------------------------------
# Pure parts


def diff_hosts(wanted, running):
    """(to_start, to_stop), sorted, to turn the running hosts into the wanted."""
    wanted, running = set(wanted), set(running)
    return sorted(wanted - running), sorted(running - wanted)


def backoff_delay(failures):
    """Seconds to wait before restarting after the n-th consecutive failure."""
    if failures < 1:
        return 0.0
    return min(BACKOFF_CAP, BACKOFF_BASE * 2 ** (failures - 1))


def classify_failure(stderr_text, host):
    """(category, hint) of an ssh failure from its stderr. category is
    'auth', 'forward' or 'other'."""
    text = stderr_text or ''
    low = text.lower()
    if ('permission denied' in low or 'host key verification failed' in low
            or 'remote host identification has changed' in low
            or 'no matching host key' in low):
        return ('auth',
                'non-interactive ssh is needed: start ssh-agent and ssh-add your key, '
                f'and make sure {host} is in known_hosts')
    if 'remote port forwarding failed for listen path' in low:
        return ('forward',
                f'check sshd AllowStreamLocalForwarding and SELinux on {host} '
                '(its audit log names the denial)')
    return 'other', ''


def forwards_for(config, host_workdir):
    """[(service, port, socket)] to forward to a host whose workdir is
    <host_workdir>, from the config.rc values. [] without a workdir."""
    if not host_workdir:
        return []
    forwards = []
    for name, var, subdir, filename in SERVICES:
        port = int(config.get(var) or DEFAULT_PORTS[var])
        socket = f'{host_workdir}/run/{subdir}/{filename.format(port=port)}'
        forwards.append((name, port, socket))
    return forwards


def socket_path_problem(host, forwards):
    """The hint for a socket path over the limit, or '' when all fit."""
    for _, _, socket in forwards:
        size = len(os.fsencode(socket))
        if size > MAX_SOCKET_PATH:
            return (f'Unix socket path is {size} bytes, over the {MAX_SOCKET_PATH}-byte limit; '
                    f'shorten the workdir path on {host}')
    return ''


def prepare_command(host, forwards):
    """The argv of the ssh that makes the socket directories on <host> and
    removes stale sockets (sshd does not unlink one itself)."""
    dirs = sorted({os.path.dirname(socket) for _, _, socket in forwards})
    sockets = [socket for _, _, socket in forwards]
    remote = 'mkdir -p ' + ' '.join(shlex.quote(d) for d in dirs)
    remote += ' && rm -f ' + ' '.join(shlex.quote(s) for s in sockets)
    return ['ssh', '-o', 'BatchMode=yes', host, remote]


def ssh_command(host, forwards):
    """The argv of the ssh child that keeps the tunnels to <host> up."""
    cmd = ['ssh', '-N',
           '-o', 'ExitOnForwardFailure=yes',
           '-o', 'ServerAliveInterval=15',
           '-o', 'ServerAliveCountMax=3',
           '-o', 'BatchMode=yes']
    for _, port, socket in forwards:
        cmd += ['-R', f'{socket}:127.0.0.1:{port}']
    cmd.append(host)
    return cmd


def format_time(t):
    return datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%dT%H:%M:%S')


def format_status_line(host, forwards, state, since, note):
    """host, services, state, since, note: tab separated."""
    services = ','.join(f'{name}:{socket}' for name, _, socket in forwards) or '-'
    return '\t'.join((host, services, state, format_time(since), note))


def format_status(lines, reload_time, pid):
    """The text of the status file: a header, then one line per host."""
    head = f'# ssh tunnels (manager pid {pid}), reloaded {format_time(reload_time)}\n'
    head += '# host\tservices\tstate\tsince\tlast log line / hint\n'
    if not lines:
        return head + '(no firewalled hosts)\n'
    return head + ''.join(line + '\n' for line in lines)


# ---------------------------------------------------------------------------
# Reading the workdir


def read_config(workdir):
    """The service ports of <workdir>/config.rc, by letting bash source it
    (like aotriton.pgdef.workdir, which this must not import)."""
    config_rc = Path(workdir) / 'config.rc'
    script = '. "$1" && echo "$CELERY_SERVICE_HOST" && echo "$POSTGRES_PORT"'
    result = subprocess.run(['/bin/bash', '-c', script, 'bash', str(config_rc)],
                            capture_output=True,
                            text=True)
    if result.returncode != 0:
        raise RuntimeError(f'cannot source {config_rc}: {result.stderr.strip()}')
    lines = result.stdout.split('\n')
    return {'CELERY_SERVICE_HOST': lines[0], 'POSTGRES_PORT': lines[1] if len(lines) > 1 else ''}


def read_firewalled_hosts(workdir):
    """The hosts with a firewalled::<hostname> = 1 config row in workers.db."""
    db = Path(workdir) / 'workers.db'
    if not db.exists():
        return set()
    conn = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    try:
        rows = conn.execute("SELECT key FROM config WHERE key GLOB 'firewalled::*' AND value = '1'").fetchall()
    finally:
        conn.close()
    return {key[len('firewalled::'):] for (key,) in rows}


def read_host_workdir(workdir, host):
    """The workdir of <host> as the server knows it: workers.db
    workdir_override, else the default_workdir config row; '' if neither."""
    conn = sqlite3.connect(f'file:{Path(workdir) / "workers.db"}?mode=ro', uri=True)
    try:
        row = conn.execute('SELECT workdir_override FROM workers WHERE hostname = ?', (host,)).fetchone()
        if row and row[0]:
            return row[0]
        row = conn.execute("SELECT value FROM config WHERE key = 'default_workdir'").fetchone()
        return row[0] if row and row[0] else ''
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Supervision


class Tunnel:
    """One ssh child, with its back-off and status."""

    def __init__(self, host, forwards, log, now, up_after=UP_AFTER):
        self.host = host
        self.up_after = up_after
        self.forwards = forwards
        self.log = log
        self.proc = None
        self.started = 0.0
        self.next_start = now
        self.failures = 0
        self.state = RESTARTING
        self.since = now
        self.note = 'starting'
        self.blocked = socket_path_problem(host, forwards)
        if not forwards:
            self.blocked = 'no workdir known for this host (workers.db workdir_override or default_workdir)'
        if self.blocked:
            self.state = DOWN
            self.note = self.blocked
            self.log(host, f'tunnel not started: {self.blocked}')
        self._stderr = []
        self._reader = None

    def _set(self, state, now, note):
        changed = (state, note) != (self.state, self.note)
        if state != self.state:
            self.since = now
        self.state, self.note = state, note
        return changed

    def _read_stderr(self, pipe):
        for line in pipe:
            line = line.rstrip('\n')
            if line:
                self._stderr.append(line)
                self.log(self.host, f'ssh: {line}')

    def _prepare(self):
        """Make the socket directory and remove stale sockets on the host;
        the error text on failure, '' on success."""
        cmd = prepare_command(self.host, self.forwards)
        self.log(self.host, 'preparing: ' + ' '.join(cmd))
        try:
            result = subprocess.run(cmd,
                                    stdin=subprocess.DEVNULL,
                                    capture_output=True,
                                    text=True,
                                    timeout=SSH_PREP_TIMEOUT)
        except (OSError, subprocess.TimeoutExpired) as e:
            return str(e) or 'timeout'
        if result.returncode != 0:
            return result.stderr.strip() or f'ssh exited with status {result.returncode}'
        return ''

    def start(self, now):
        error = self._prepare()
        if error:
            self.proc = None
            return self._failed(now, error, 0.0)
        cmd = ssh_command(self.host, self.forwards)
        self.log(self.host, 'starting: ' + ' '.join(cmd))
        self._stderr = []
        try:
            self.proc = subprocess.Popen(cmd,
                                         stdin=subprocess.DEVNULL,
                                         stdout=subprocess.DEVNULL,
                                         stderr=subprocess.PIPE,
                                         text=True)
        except OSError as e:
            self.proc = None
            return self._failed(now, str(e), 0.0)
        self.started = now
        self._reader = threading.Thread(target=self._read_stderr, args=(self.proc.stderr,), daemon=True)
        self._reader.start()
        return self._set(RESTARTING, now, 'connecting')

    def _failed(self, now, text, ran_for):
        if ran_for >= STABLE_AFTER:
            self.failures = 0
        self.failures += 1
        delay = backoff_delay(self.failures)
        self.next_start = now + delay
        category, hint = classify_failure(text, self.host)
        last = text.strip().splitlines()[-1] if text.strip() else 'ssh exited'
        note = f'{last} -- {hint}' if hint else last
        self.log(self.host, f'failed ({category}), retry in {delay:.0f}s: {note}')
        return self._set(RESTARTING, now, note)

    def poll(self, now):
        """Advance the state machine; True when the status changed."""
        if self.blocked:
            return False
        if self.proc is None:
            return self.start(now) if now >= self.next_start else False
        code = self.proc.poll()
        if code is None:
            if now - self.started >= STABLE_AFTER:
                self.failures = 0
            if now - self.started >= self.up_after and self.state != UP:
                self.log(self.host, 'up')
                return self._set(UP, now, '')
            return False
        self._reader.join(timeout=1.0)
        self.proc = None
        text = '\n'.join(self._stderr) or f'ssh exited with status {code}'
        return self._failed(now, text, now - self.started)

    def stop(self, now):
        """Terminate the child; leaves the state to the caller."""
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=KILL_GRACE)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        if self._reader is not None:
            self._reader.join(timeout=1.0)
        self.proc = None
        self.log(self.host, 'stopped')

    def mark_down(self, now, note):
        self._set(DOWN, now, note)

    def status_line(self):
        return format_status_line(self.host, self.forwards, self.state, self.since, self.note)


class Manager:
    def __init__(self, workdir, poll_interval=POLL_INTERVAL, up_after=UP_AFTER, clock=time.time):
        self.workdir = Path(workdir)
        self.run_dir = self.workdir / 'run'
        self.poll_interval = poll_interval
        self.up_after = up_after
        self.clock = clock
        self.tunnels = {}
        self.reload_time = clock()
        self.stopping = False
        self._touch_seen = None
        self._log_lock = threading.Lock()
        self._log_file = None

    # -- files

    def log(self, host, message):
        line = f'{format_time(self.clock())} [{host or "-"}] {message}\n'
        with self._log_lock:
            if self._log_file is not None:
                self._log_file.write(line)
                self._log_file.flush()

    def write_status(self):
        lines = [t.status_line() for _, t in sorted(self.tunnels.items())]
        text = format_status(lines, self.reload_time, os.getpid())
        tmp = self.run_dir / (STATUS_FILE + '.tmp')
        tmp.write_text(text)
        os.replace(tmp, self.run_dir / STATUS_FILE)

    def _touch_mtime(self):
        try:
            return (self.run_dir / TOUCH_FILE).stat().st_mtime_ns
        except OSError:
            return None

    # -- reload

    def reload(self):
        """Re-read workers.db and config.rc; start or stop tunnels to match."""
        now = self.clock()
        self.reload_time = now
        try:
            config = read_config(self.workdir)
            wanted = {h for h in read_firewalled_hosts(self.workdir) if _HOSTNAME.match(h)}
            host_forwards = {h: forwards_for(config, read_host_workdir(self.workdir, h)) for h in wanted}
        except (RuntimeError, sqlite3.Error, ValueError) as e:
            self.log(None, f'reload failed, keeping the current tunnels: {e}')
            return
        to_start, to_stop = diff_hosts(wanted, self.tunnels)
        for host in to_stop:
            self.tunnels.pop(host).stop(now)
            self.log(host, 'no longer firewalled: tunnel removed')
        for host, tunnel in sorted(self.tunnels.items()):
            if tunnel.forwards != host_forwards[host]:
                self.log(host, 'forwarded services changed: restarting')
                tunnel.stop(now)
                self.tunnels[host] = Tunnel(host, host_forwards[host], self.log, now, self.up_after)
        for host in to_start:
            self.tunnels[host] = Tunnel(host, host_forwards[host], self.log, now, self.up_after)
        self.log(None, f'reload: wanted={sorted(wanted)} started={to_start} stopped={to_stop}')

    # -- main loop

    def tick(self):
        """One wakeup: notice a touch, advance the tunnels; status on change."""
        changed = False
        mtime = self._touch_mtime()
        if mtime != self._touch_seen:
            self._touch_seen = mtime
            self.reload()
            changed = True
        now = self.clock()
        for _, tunnel in sorted(self.tunnels.items()):
            changed |= tunnel.poll(now)
        if changed:
            self.write_status()

    def shutdown(self):
        now = self.clock()
        self.log(None, 'stopping')
        for _, tunnel in sorted(self.tunnels.items()):
            tunnel.stop(now)
            tunnel.mark_down(now, 'manager stopped')
        self.reload_time = now
        self.write_status()

    def run(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        signal.signal(signal.SIGTERM, self._on_signal)
        signal.signal(signal.SIGINT, self._on_signal)
        with open(self.run_dir / LOG_FILE, 'a') as self._log_file:
            self.log(None, f'started, pid {os.getpid()}')
            self._touch_seen = self._touch_mtime()
            self.reload()          # always at start
            self.write_status()
            try:
                while not self.stopping:
                    self.tick()
                    time.sleep(self.poll_interval)
            finally:
                self.shutdown()
        return 0

    def _on_signal(self, signum, frame):
        self.stopping = True


def main(argv=None):
    parser = argparse.ArgumentParser(description='SSH tunnels to the firewalled hosts of a workdir.')
    parser.add_argument('workdir')
    parser.add_argument('--poll_interval',
                        type=float,
                        default=POLL_INTERVAL,
                        help='seconds between main loop wakeups')
    parser.add_argument('--up_after',
                        type=float,
                        default=UP_AFTER,
                        help='seconds an ssh must stay alive to count as up')
    args = parser.parse_args(argv)
    return Manager(args.workdir,
                   poll_interval=args.poll_interval,
                   up_after=args.up_after).run()


if __name__ == '__main__':
    sys.exit(main())
