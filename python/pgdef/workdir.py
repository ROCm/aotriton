# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The workdir: pgdef's configuration data on disk, for tools that are handed
`--workdir`. It holds config.rc (PostgreSQL and container settings, bash
syntax, owned by the `.tune` tooling) and workers.db (the host inventory).
Some application settings are mixed in; that is accepted for now.

This is pgdef's deployment layer, a sibling of `pq` and `localq`, not part of
them: the engine core (aotriton.pgdef.pq.conn) only reads PG* variables from
the environment and must not import this module. The engine itself (aotriton.pgdef.pq.conn) only reads
PG* variables from the environment; worker_service.sh exports them for the
processes it starts.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def get_db_connection_params(workdir: Path) -> dict[str, Any]:
    """
    Get PostgreSQL connection parameters from workdir config.

    Args:
        workdir: Path to workdir containing config.rc

    Returns:
        Connection parameters dictionary with keys: host, port, user, password
        (never a database name: the server default is used)
    """
    # Source config.rc and extract environment variables
    config_rc = workdir / 'config.rc'
    if not config_rc.exists():
        raise FileNotFoundError(f"Config file not found: {config_rc}")

    # Source the config file and print specific variables
    # Note: config.rc sets variables but doesn't export them, so we can't use 'env'
    result = subprocess.run(
        f'. {config_rc} && echo "$CELERY_SERVICE_HOST" && echo "$POSTGRES_PORT" && echo "$POSTGRES_USER" && echo "$POSTGRES_PASSWORD"',
        shell=True,
        capture_output=True,
        text=True,
        executable='/bin/bash'
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to source config.rc: {result.stderr}")

    # Parse output lines
    lines = result.stdout.strip().split('\n')
    if len(lines) < 4:
        raise RuntimeError(f"Incomplete config.rc output: {result.stdout}")

    return {
        'host': lines[0] or 'localhost',
        'port': int(lines[1]) if lines[1] else 5432,
        'user': lines[2] or 'aotriton',
        'password': lines[3] or None
    }


def load_config(workdir: Path) -> None:
    """Load <workdir>/config.rc into os.environ (simple KEY=VALUE lines), where
    the engine reads the PostgreSQL connection parameters from
    (pgdef.pq.conn.params_from_env). Exits with a message if there is no config.rc."""
    config_rc = workdir / 'config.rc'
    if not config_rc.exists():
        sys.exit(f"Error: config.rc not found at {config_rc}")
    with open(config_rc) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Remove inline comments
            if '#' in line:
                line = line[:line.index('#')].strip()
            if '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value.strip().strip('"').strip("'")
