# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Connection parameters and a reconnecting connection wrapper.

pq functions take a connection object, never parameters: the caller creates one
connection per worker or process and reuses it. Parameters have only host,
port, user and password. NEVER add a database name: pgdef always uses the
server's default database (the one named after the user).
"""

import logging
import os

import psycopg

logger = logging.getLogger(__name__)


def params_from_env(environ=None) -> dict:
    """PostgreSQL connection parameters from the environment.

    The standard PGHOST/PGPORT/PGUSER/PGPASSWORD win. The names `config.rc`
    uses (CELERY_SERVICE_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD)
    are accepted as a fallback, so a shell that merely sourced config.rc and
    exported its variables works too. Sourcing config.rc itself is the
    infrastructure's job (`.tune/lib/config_load.sh`), not the engine's.
    """
    env = os.environ if environ is None else environ

    def first(*names, default=None):
        for name in names:
            value = env.get(name)
            if value:
                return value
        return default

    return {
        'host': first('PGHOST', 'CELERY_SERVICE_HOST', default='localhost'),
        'port': int(first('PGPORT', 'POSTGRES_PORT', default='5432')),
        'user': first('PGUSER', 'POSTGRES_USER', default='aotriton'),
        'password': first('PGPASSWORD', 'POSTGRES_PASSWORD'),
    }


class ReconnectableConn:
    """
    Wraps a psycopg connection with automatic reconnection on close.

    All CPU handlers within one worker process share this single object.
    If PostgreSQL drops the connection (server restart, idle timeout),
    cursor() reconnects transparently before returning.

    NOTE: This is not thread-safe. Each process should have its own instance.
    """

    def __init__(self, conn_params: dict, **connect_kwargs):
        """
        Args:
            conn_params: dict of psycopg connection parameters (host, port, user, password)
            **connect_kwargs: extra args forwarded to psycopg.connect (e.g. autocommit, row_factory)
        """
        self._params = conn_params
        self._kwargs = connect_kwargs
        self._conn = None
        self._connect()

    def _connect(self):
        self._conn = psycopg.connect(**self._params, **self._kwargs)
        logger.info("Database connection (re)established")

    def cursor(self, **kwargs):
        if self._conn.closed:
            logger.warning("Database connection was closed; reconnecting...")
            self._connect()
        return self._conn.cursor(**kwargs)

    @property
    def closed(self):
        return self._conn.closed

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()
