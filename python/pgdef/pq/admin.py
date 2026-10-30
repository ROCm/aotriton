# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Schema administration for the engine's tables.

Operators run `.def/bin/initdb`, which feeds every layer's schema.sql to psql.
This module is the same thing for Python callers (tests, tools): it applies the
engine's schema.sql through a connection.
"""

from pathlib import Path

SCHEMA_NAME = 'pgdef'


def schema_path() -> Path:
    return Path(__file__).with_name('schema.sql')


def apply_schema(conn, path: Path | None = None) -> None:
    """Apply a schema.sql (default: the engine's) through `conn`.

    Safe to repeat: schema.sql is idempotent. It is plain SQL with no psql
    meta-commands precisely so that this can hand it to psycopg unchanged.
    """
    with conn.cursor() as cur:
        cur.execute((path or schema_path()).read_text())


def drop_schema(conn) -> None:
    """Drop the engine's schema and everything in it (`initdb --recreate`).
    Application schemas that reference it must be dropped first or with it."""
    with conn.cursor() as cur:
        cur.execute(f'DROP SCHEMA IF EXISTS {SCHEMA_NAME} CASCADE')


def all_schema_files(python_dir: Path | None = None) -> list[Path]:
    """Every layer's schema.sql, in the order they must be applied: the
    engine's first, then each application's (`python/<app>/pq/schema.sql`, by
    name). `.def/bin/initdb` does the same with a shell glob; an application
    needs no edit anywhere to be picked up.
    """
    python_dir = python_dir or Path(__file__).resolve().parent.parent.parent
    engine = python_dir / 'pgdef' / 'pq' / 'schema.sql'
    others = sorted(p for p in python_dir.glob('*/pq/schema.sql') if p != engine)
    return [engine] + others


def schema_names(python_dir: Path | None = None) -> list[str]:
    """The PostgreSQL schema of every layer: the directory name of each
    schema.sql's application (pgdef, tune)."""
    return [p.parent.parent.name for p in all_schema_files(python_dir)]
