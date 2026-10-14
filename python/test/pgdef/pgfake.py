# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A recording fake of the slice of psycopg that pgdef.pq uses.

No PostgreSQL is needed: it checks that the SQL text and the parameters a pq
function passes agree with each other (every named placeholder has a value,
every positional placeholder has one), that parentheses and quotes balance,
and what the function does with canned rows. It cannot check that PostgreSQL
accepts the statement; the `pg`-marked tests do that.
"""

import contextlib
import re


class FakeCursor:
    def __init__(self, conn, row_factory=None):
        self.conn = conn
        self.row_factory = row_factory
        self.rowcount = 1
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, query, params=None):
        text = query if isinstance(query, str) else query.as_string(None)
        self.conn.calls.append((text, params))
        self.conn.row_factories.append(self.row_factory)
        self._rows = list(self.conn.next_rows())
        self.rowcount = self.conn.rowcount

    def executemany(self, query, params_seq):
        text = query if isinstance(query, str) else query.as_string(None)
        params_seq = list(params_seq)
        self.conn.calls.append((text, params_seq))
        self.rowcount = len(params_seq)

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None


class FakeConn:
    def __init__(self, rows=(), rowcount=1):
        self.calls = []
        self.row_factories = []     # the row_factory each execute() ran with
        self.transactions = 0
        self._queue = list(rows)
        self.rowcount = rowcount

    def next_rows(self):
        return self._queue.pop(0) if self._queue else []

    def cursor(self, **kwargs):
        return FakeCursor(self, kwargs.get('row_factory'))

    def transaction(self):
        self.transactions += 1
        return contextlib.nullcontext()


_NAMED = re.compile(r'%\((\w+)\)s')
_POSITIONAL = re.compile(r'%s')


def check_statement(text: str, params):
    """Assert placeholders and params agree, and that the text is well-formed."""
    # Strip quoted literals before counting parentheses.
    stripped = re.sub(r"'(?:[^']|'')*'", "''", text)
    assert stripped.count('(') == stripped.count(')'), f'unbalanced parentheses in:\n{text}'
    assert text.count("'") % 2 == 0, f'unbalanced quotes in:\n{text}'
    assert '{' not in stripped.replace('{}', ''), f'unformatted f-string brace in:\n{text}'
    named = set(_NAMED.findall(text))
    if isinstance(params, dict):
        assert named <= set(params), f'placeholders without values: {named - set(params)}\n{text}'
        assert not _POSITIONAL.search(_NAMED.sub('', text)), 'mixed placeholder styles'
    elif params is not None and not isinstance(params, list) or (isinstance(params, list) and params and not isinstance(params[0], dict)):
        assert not named, f'named placeholders with positional params:\n{text}'
        assert len(_POSITIONAL.findall(text)) == len(params), (
            f'{len(_POSITIONAL.findall(text))} placeholders, {len(params)} params:\n{text}')
