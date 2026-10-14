# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Helpers for tests that need a real PostgreSQL server (pytest marker `pg`).

These tests DROP and recreate every layer's schema (pgdef, tune), so
they must only ever run against a disposable server. They are skipped unless
AOTRITON_PG_TEST_DISPOSABLE=1 says so AND a server answers on the usual
PGHOST/PGPORT/PGUSER/PGPASSWORD (never a database name: the server default is
used). Typical use:

    docker run --rm -d -p 55432:5432 -e POSTGRES_PASSWORD=pw postgres:16
    PGHOST=localhost PGPORT=55432 PGUSER=postgres PGPASSWORD=pw \\
        AOTRITON_PG_TEST_DISPOSABLE=1 pytest -m pg python/test python/tune/tests
"""

import os

from . import admin
from .conn import params_from_env


def params_or_skip(pytest) -> dict:
    """Connection parameters, or pytest.skip() when no disposable server is set up."""
    if os.environ.get('AOTRITON_PG_TEST_DISPOSABLE') != '1':
        pytest.skip('set AOTRITON_PG_TEST_DISPOSABLE=1 to run against a disposable PostgreSQL server')
    import psycopg
    params = params_from_env()
    try:
        with psycopg.connect(**params, connect_timeout=3):
            pass
    except Exception as e:
        pytest.skip(f'no PostgreSQL server reachable: {e}')
    return params


def connect(params: dict):
    import psycopg
    return psycopg.connect(**params, autocommit=True)


def fresh_schemas(params: dict):
    """Drop every layer's schema, re-apply them all in initdb's order, and
    return an autocommit connection."""
    conn = connect(params)
    with conn.cursor() as cur:
        for name in reversed(admin.schema_names()):
            cur.execute(f'DROP SCHEMA IF EXISTS {name} CASCADE')
    for path in admin.all_schema_files():
        admin.apply_schema(conn, path)
    return conn


def ensure_pool(conn, pool: str) -> None:
    """Create `pool`'s partitions, as an application's schema.sql does with
    `SELECT pgdef.ensure_pool(...)`; for tests that need a pool of their own."""
    with conn.cursor() as cur:
        cur.execute('SELECT pgdef.ensure_pool(%s)', (pool,))
