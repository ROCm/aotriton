# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An in-process node for tests: a real LocalBroker on a Unix socket, real
GenericWorkers and a real PGReader, each in a thread, with the database
replaced by an in-memory fake of the pq functions the engine calls.

Everything except PostgreSQL and the GPU runners is the production code, so
this exercises message routing, EMITS checking, fan-in, the single failure
path, undeliverable messages, lease renewal and the reader's ack handling.
"""

import threading
import time
from dataclasses import dataclass, field

from aotriton.pgdef.localq.broker import LocalBroker
from aotriton.pgdef.localq.generic_worker import GenericWorker
from aotriton.pgdef.localq.handlers import HandlerSet, NodeContext
from aotriton.pgdef.localq.pg_reader import PGReader
from aotriton.pgdef.pq import logistics as lg
from aotriton.pgdef.pq import tasks as pgtasks
from aotriton.pgdef.pq import workers as pgworkers


@dataclass
class FakeRow:
    id: int
    pool: str
    payload: dict
    category: tuple
    status: str = 'pending'
    attempt: int = 0
    max_attempts: int = lg.DEFAULT_MAX_ATTEMPTS
    errors: list = field(default_factory=list)
    renewals: int = 0
    lease_lost: bool = False
    completed: bool = False
    released: bool = False


class FakeDB:
    """In-memory stand-in for the pq calls the engine makes at run time."""

    def __init__(self):
        self.rows: dict[int, FakeRow] = {}
        self.lock = threading.Lock()
        self.next_id = 1
        self.reports: list[tuple] = []

    def add(self, pool='gfx950', payload=None, category=('toy', None), max_attempts=lg.DEFAULT_MAX_ATTEMPTS) -> int:
        with self.lock:
            row = FakeRow(self.next_id, pool, payload or {}, category, max_attempts=max_attempts)
            self.rows[row.id] = row
            self.next_id += 1
            return row.id

    # -- the pq surface ---------------------------------------------------------

    def claim(self, conn, *, pool, category, subpool, worker, lease_s=0, limit=1):
        out = []
        with self.lock:
            for row in sorted(self.rows.values(), key=lambda r: r.id):
                if len(out) >= limit:
                    break
                if row.status == 'pending' and row.pool == pool and row.category[0] == category[0] \
                        and (row.category[1] is None or row.category[1] == category[1]):
                    row.status = 'running'
                    row.attempt += 1
                    doc = {'v': 1, 'attempt': row.attempt, 'retry': {'max_attempts': row.max_attempts}}
                    out.append(pgtasks.Task(
                        id=row.id, pool=row.pool, subpool=None, cat_level0=row.category[0],
                        cat_level1=row.category[1], payload=row.payload, appcookie=None, priority=5,
                        status='running', logistics=lg.Logistics.from_json(doc)))
        return out

    def renew(self, conn, ref, *, worker, lease_s=0):
        with self.lock:
            row = self.rows[ref[0]]
            if row.status != 'running':
                return False
            row.renewals += 1
            return not row.lease_lost

    def complete(self, conn, ref):
        with self.lock:
            row = self.rows[ref[0]]
            if row.status != 'running':
                return False
            row.status, row.completed = 'done', True
            return True

    def fail(self, conn, ref, error, *, permanent=False):
        with self.lock:
            row = self.rows[ref[0]]
            if row.status != 'running':
                return None
            row.errors.append((error, permanent))
            row.status = 'failed' if permanent or row.attempt >= row.max_attempts else 'pending'
            return row.status

    def release(self, conn, ref):
        with self.lock:
            row = self.rows[ref[0]]
            row.status, row.released = 'pending', True
            row.attempt -= 1
            return True

    def reap_expired(self, conn, pool):
        return []

    def status(self, task_id):
        with self.lock:
            return self.rows[task_id].status

    def install(self, monkeypatch):
        for name in ('claim', 'renew', 'complete', 'fail', 'release', 'reap_expired'):
            monkeypatch.setattr(pgtasks, name, getattr(self, name))
        monkeypatch.setattr(pgworkers, 'upsert', lambda *a, **k: None)
        monkeypatch.setattr(pgworkers, 'remove', lambda *a, **k: None)


class DummyConn:
    def close(self):
        pass


def wait_for(predicate, timeout=10.0, step=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return False


class Node:
    """One node: broker + workers + readers, in threads."""

    def __init__(self, tmp_path, handlers, start, db: FakeDB, *, queues=('gpu', 'cpu'), n_gpu=1,
                 n_cpu=1, n_readers=1, lease_s=0.6, runners=None, enrich=None,
                 category=('toy', None), pool='gfx950'):
        self.socket = str(tmp_path / 'broker.sock')
        self.handler_set = HandlerSet(handlers)
        self.db = db
        self.threads = []
        self.workers = []
        self.broker = LocalBroker(self.socket, handlers=self.handler_set.by_class,
                                  priorities=self.handler_set.priorities(), queues=queues)
        self.broker.start()
        self._spawn(self.broker.run)

        for i in range(n_gpu if 'gpu' in queues else 0):
            ctx = NodeContext(app='toy', role='gpu_worker', worker_id=f'gpu-{i}', gpu_id=i,
                              runner_types=runners or {}, start=start)
            self._add_worker(GenericWorker(f'gpu-{i}', 'gpu', self.handler_set, ctx, self.socket, idle_sleep_s=0.02))
        for i in range(n_cpu if 'cpu' in queues else 0):
            ctx = NodeContext(app='toy', role='cpu_worker', worker_id=f'cpu-{i}', db_conn=DummyConn(),
                              start=start)
            self._add_worker(GenericWorker(f'cpu-{i}', 'cpu', self.handler_set, ctx, self.socket, idle_sleep_s=0.02))
        self.readers = []
        for i in range(n_readers):
            reader = PGReader(worker_id=f'host/reader-{i}', pool=pool, subpool=None, category=category,
                              broker_socket=self.socket, db_conn=DummyConn(), start=start,
                              enrich=enrich, lease_s=lease_s, poll_s=0.02, reap_interval_s=3600)
            self.readers.append(reader)
            self._spawn(reader.run)

    def _spawn(self, target):
        th = threading.Thread(target=target, daemon=True)
        th.start()
        self.threads.append(th)

    def _add_worker(self, worker):
        self.workers.append(worker)
        self._spawn(worker.run)

    def stop(self):
        for r in self.readers:
            r.shutdown()
        for w in self.workers:
            w.shutdown()
        self.broker.shutdown()
        for th in self.threads:
            th.join(timeout=5)
