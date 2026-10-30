# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Every filesystem location a node assumes, in one place.

The tuning image mounts the workdir at /wkdir and runs as root; a fleet
container has the workdir wherever the scratch is mounted and runs as the
user. Code therefore never spells these locations: it calls the functions
below, each of which reads its environment variable at call time and falls
back to the tuning image's value. With no variable set, every path is what it
has always been.

The shell side is `.def/lib/paths.sh`: the same variables, the same defaults.
Keep the two in sync.

| Variable                 | Function          | Default                       |
|--------------------------|-------------------|-------------------------------|
| `AOTRITON_WORKDIR`       | `workdir()`       | `/wkdir`                      |
| `AOTRITON_SRC`           | `src_dir()`       | `workdir() / 'aotriton.src'`  |
| `AOTRITON_RUNDIR`        | `rundir()`        | `workdir() / 'run'`           |
| `AOTRITON_TMPFS`         | `tmpfs()`         | `/dev/shm`                    |
| `AOTRITON_BROKER_SOCKET` | `broker_socket()` | `/tmp/aotriton-broker.sock`   |

`AOTRITON_RUNDIR` holds node-local runtime files (sockets, pid files, the
bytecode cache) and must be on a local filesystem, never on a network share.
`AOTRITON_TMPFS` holds prepared data and should be memory-backed.

`broker_socket()` keeps the localq default it has always had. The tuner's
worker_service.sh sets `AOTRITON_BROKER_SOCKET` to `<workdir>/run/broker.sock`;
perfmon sets it under its node-local `AOTRITON_RUNDIR`, since a Unix socket
cannot live on the shared scratch.
"""

import os
from pathlib import Path

_DEFAULT_WORKDIR = '/wkdir'
_DEFAULT_TMPFS = '/dev/shm'
_DEFAULT_BROKER_SOCKET = '/tmp/aotriton-broker.sock'


def _env(name: str) -> str | None:
    # An empty variable counts as unset, like `${VAR:-default}` in paths.sh.
    return os.environ.get(name) or None


def workdir() -> Path:
    """The workdir: config.rc, installed/, run/."""
    return Path(_env('AOTRITON_WORKDIR') or _DEFAULT_WORKDIR)


def src_dir() -> Path:
    """The AOTriton checkout the node runs from."""
    value = _env('AOTRITON_SRC')
    return Path(value) if value else workdir() / 'aotriton.src'


def rundir() -> Path:
    """Node-local runtime files: sockets, pid files, pycache."""
    value = _env('AOTRITON_RUNDIR')
    return Path(value) if value else workdir() / 'run'


def tmpfs() -> Path:
    """Fast (memory-backed) scratch for prepared data."""
    return Path(_env('AOTRITON_TMPFS') or _DEFAULT_TMPFS)


def broker_socket() -> Path:
    """The localq broker's Unix socket."""
    return Path(_env('AOTRITON_BROKER_SOCKET') or _DEFAULT_BROKER_SOCKET)
