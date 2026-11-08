# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Keepalives and the one-shot retry after a dropped database connection.
No PostgreSQL needed: psycopg.connect and the connection are fakes."""

import psycopg
import pytest

from aotriton.pgdef.localq.generic_worker import GenericWorker
from aotriton.pgdef.localq.handlers import HandlerSet, MessageHandler, NodeContext
from aotriton.pgdef.pq import conn as pgconn

CALLS: list = []
FAIL: dict = {}


class Writer(MessageHandler):
    CLASS = 'retry/write'
    QUEUE = 'cpu'
    EMITS = ()

    def handle(self, msg, ctx):
        CALLS.append(msg['class'])
        exc = FAIL.get('exc')
        if exc is not None and len(CALLS) <= FAIL.get('times', 1):
            raise exc
        return None


class GpuThing(Writer):
    CLASS = 'retry/gpu'
    QUEUE = 'gpu'


class FakeConn:
    def __init__(self):
        self.reconnects = 0

    def reconnect(self):
        self.reconnects += 1


@pytest.fixture(autouse=True)
def _reset():
    CALLS.clear()
    FAIL.clear()


def _worker(queue='cpu'):
    db = FakeConn()
    ctx = NodeContext(app='toy', role=f'{queue}_worker', worker_id='w', db_conn=db)
    w = GenericWorker('w', queue, HandlerSet([Writer, GpuThing]), ctx, broker_socket='/nonexistent')
    return w, db


def _run(w, cls='retry/write'):
    return w._run_handler(w.handlers[cls], {'class': cls, 'task_id': 1})


def test_keepalive_kwargs_reach_psycopg_connect(monkeypatch):
    seen = {}
    monkeypatch.setattr(pgconn.psycopg, 'connect', lambda **kw: seen.update(kw) or FakeRawConn())
    pgconn.ReconnectableConn({'host': 'h', 'port': 1, 'user': 'u', 'password': 'p'}, autocommit=True)
    assert seen['keepalives'] == 1
    assert seen['keepalives_idle'] == 60
    assert seen['keepalives_interval'] == 15
    assert seen['keepalives_count'] == 4
    assert seen['autocommit'] is True
    assert 'dbname' not in seen


class FakeRawConn:
    closed = False

    def close(self):
        self.closed = True


def test_reconnect_opens_a_fresh_connection(monkeypatch):
    made = []
    monkeypatch.setattr(pgconn.psycopg, 'connect', lambda **kw: made.append(FakeRawConn()) or made[-1])
    c = pgconn.ReconnectableConn({'host': 'h'})
    c.reconnect()
    assert len(made) == 2
    assert made[0].closed


def test_connection_error_is_retried_once_and_succeeds():
    FAIL['exc'] = psycopg.OperationalError('consuming input failed: server closed the connection unexpectedly')
    w, db = _worker()
    _run(w)
    assert len(CALLS) == 2
    assert db.reconnects == 1


def test_non_connection_error_is_not_retried():
    FAIL['exc'] = psycopg.errors.IntegrityError('dup')
    w, db = _worker()
    with pytest.raises(psycopg.errors.IntegrityError):
        _run(w)
    assert len(CALLS) == 1
    assert db.reconnects == 0


def test_sql_level_operational_error_is_not_retried():
    FAIL['exc'] = psycopg.errors.QueryCanceled('statement timeout')
    w, db = _worker()
    with pytest.raises(psycopg.errors.QueryCanceled):
        _run(w)
    assert len(CALLS) == 1


def test_retry_happens_at_most_once():
    FAIL['exc'] = psycopg.OperationalError('connection closed')
    FAIL['times'] = 5
    w, db = _worker()
    with pytest.raises(psycopg.OperationalError):
        _run(w)
    assert len(CALLS) == 2
    assert db.reconnects == 1


def test_gpu_handler_is_never_retried():
    FAIL['exc'] = psycopg.OperationalError('connection closed')
    w, db = _worker('gpu')
    with pytest.raises(psycopg.OperationalError):
        _run(w, 'retry/gpu')
    assert len(CALLS) == 1
