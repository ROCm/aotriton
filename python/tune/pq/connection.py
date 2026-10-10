# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Reconnectable PostgreSQL connection wrapper.
"""

import logging
import psycopg

logger = logging.getLogger(__name__)


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
