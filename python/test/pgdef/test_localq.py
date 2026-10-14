# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""aotriton.pgdef.localq: the handler graph, fan-in, the single failure path,
undeliverable messages and lease renewal.

CPU-only. The pure-logic tests need nothing; the node tests run the production
broker, workers and PG reader in threads (localq_harness.py) with an in-memory
fake of the pq calls, because no PostgreSQL is available here.
"""

import pytest

from aotriton.pgdef.localq import semaphore
from aotriton.pgdef.localq.generic_worker import GenericWorker
from aotriton.pgdef.localq.handlers import (DAG_ACK, GRACEFUL_CANCEL, TASK_FAILED, EmitsViolation,
                                            HandlerSet, MessageHandler, NodeContext, Permanent,
                                            StartHandler, ack, check_declaration, child, task_header)
from aotriton.pgdef.localq import graph

from localq_harness import FakeDB, Node, wait_for

# ---------------------------------------------------------------------------
# A toy application: start -> work x N (gpu) -> result (cpu) -> join (fan-in) -> ack
# ---------------------------------------------------------------------------

BEHAVIOUR: dict = {}      # tests steer the toy handlers through this


class ToyStart(StartHandler):
    CLASS = 'toy/start'
    QUEUE = 'gpu'
    EMITS = ('toy/work', 'toy/join')

    def handle(self, msg, ctx):
        if BEHAVIOUR.get('fail_start_on_attempt') == msg['attempt']:
            raise RuntimeError('transient start failure')
        n = msg['payload'].get('n', 3)
        keys = [i for i in range(n)]
        join = semaphore.wait('toy/result', keys, child(msg, 'toy/join'))
        return [join] + [child(msg, 'toy/work', index=i) for i in keys]

    @classmethod
    def cleanup(cls, msg, ctx):
        BEHAVIOUR.setdefault('cleaned', []).append(msg.get('task_id'))


class ToyWork(MessageHandler):
    CLASS = 'toy/work'
    QUEUE = 'gpu'
    EMITS = ('toy/result',)
    PRIORITY = 2

    def handle(self, msg, ctx):
        if BEHAVIOUR.get('work_raises'):
            raise BEHAVIOUR['work_raises']
        if BEHAVIOUR.get('emit_wrong'):
            return child(msg, 'toy/join')               # not declared in EMITS
        if BEHAVIOUR.get('work_sleep'):
            import time
            time.sleep(BEHAVIOUR['work_sleep'])
        return semaphore.signal(child(msg, 'toy/result', index=msg['index']), key=msg['index'])


class ToyResult(MessageHandler):
    CLASS = 'toy/result'
    QUEUE = 'cpu'
    EMITS = ()

    def handle(self, msg, ctx):
        BEHAVIOUR.setdefault('results', []).append((msg['task_id'], msg['index']))


class ToyJoin(MessageHandler):
    CLASS = 'toy/join'
    QUEUE = 'cpu'
    EMITS = (DAG_ACK,)
    PRIORITY = 4

    def handle(self, msg, ctx):
        from aotriton.pgdef.pq import tasks as pgtasks
        pgtasks.complete(ctx.db_conn, (msg['task_id'], msg['pool']))
        BEHAVIOUR.setdefault('joined', []).append((msg['task_id'], sorted(semaphore.get_aggregate_results(msg))))
        return ack(msg)


TOY = [ToyStart, ToyWork, ToyResult, ToyJoin]


@pytest.fixture(autouse=True)
def _reset_behaviour():
    BEHAVIOUR.clear()
    yield
    BEHAVIOUR.clear()


# ---------------------------------------------------------------------------
# fan-in
# ---------------------------------------------------------------------------

def _join(keys):
    base = {'class': 'x/start', 'task_id': 7, 'pool': 'p', 'payload': {}}
    return semaphore.wait('x/result', keys, child(base, 'x/join'))


def _result(key, value=None, task_id=7):
    return semaphore.signal({'class': 'x/result', 'task_id': task_id}, key, value)


def test_semaphore_resolves_once_every_key_has_arrived():
    join = _join([('a', 0), ('a', 1), ('b', 0)])
    assert join['depends'] == ['x/result']
    assert not semaphore.resolve(join, _result(('a', 0)))
    assert not semaphore.resolve(join, _result(('b', 0)))
    assert semaphore.resolve(join, _result(('a', 1), value={'t': 2}))
    assert semaphore.get_aggregate_results(join) == {('a', 0): None, ('b', 0): None, ('a', 1): {'t': 2}}


def test_semaphore_duplicate_delivery_does_not_resolve_early():
    join = _join([1, 2])
    assert not semaphore.resolve(join, _result(1))
    assert not semaphore.resolve(join, _result(1))
    assert semaphore.resolve(join, _result(2))


def test_semaphore_ignores_other_classes_tasks_and_keyless_messages():
    join = _join([1])
    assert not semaphore.resolve(join, {'class': 'x/other', 'task_id': 7, 'signal_key': 1})
    assert not semaphore.resolve(join, _result(1, task_id=8))
    assert not semaphore.resolve(join, {'class': 'x/result', 'task_id': 7})
    assert semaphore.resolve(join, _result(1))


def test_semaphore_ignores_signals_from_an_older_attempt():
    join = _join([1])
    join['attempt'] = 2
    old = _result(1)
    old['attempt'] = 1
    assert not semaphore.resolve(join, old)
    new = _result(1)
    new['attempt'] = 2
    assert semaphore.resolve(join, new)


def test_semaphore_rejects_keys_that_cannot_be_aggregated():
    with pytest.raises(TypeError):
        _join([{'iface': 'a'}])
    with pytest.raises(TypeError):
        semaphore.signal({'class': 'x/result'}, key=('a', {'b': 1}))


def test_semaphore_with_no_keys_is_not_blocked():
    join = _join([])
    assert 'depends' not in join and semaphore.get_aggregate_results(join) == {}


def test_semaphore_survives_json_roundtrip():
    import json
    join = json.loads(json.dumps(_join([('a', 0)])))
    assert semaphore.resolve(join, json.loads(json.dumps(_result(('a', 0), value=[1]))))
    assert semaphore.get_aggregate_results(join) == {('a', 0): [1]}


def test_message_helpers():
    msg = {'class': 'a', 'task_id': 3, 'pool': 'gfx950', 'subpool': None, 'category': ['toy', None],
           'attempt': 2, 'payload': {'k': 1}, 'junk': 1}
    assert task_header(msg) == {k: msg[k] for k in ('task_id', 'pool', 'subpool', 'category', 'attempt', 'payload')}
    c = child(msg, 'b', extra=5)
    assert c['class'] == 'b' and c['extra'] == 5 and 'junk' not in c and c['payload'] == {'k': 1}
    assert ack(msg) == {'class': DAG_ACK, 'task_id': 3}
    assert ack(msg, negative=True, error='x') == {'class': DAG_ACK, 'task_id': 3, 'negative': True, 'error': 'x'}


# ---------------------------------------------------------------------------
# declarations
# ---------------------------------------------------------------------------

def test_handler_set_includes_the_engine_handlers():
    hs = HandlerSet(TOY)
    assert TASK_FAILED in hs.by_class and GRACEFUL_CANCEL in hs.by_class
    assert hs.queue_of('toy/work') == 'gpu' and hs.queue_of(TASK_FAILED) == 'cpu'
    assert hs.priorities()['toy/join'] == 4 and hs.priorities()['toy/start'] == 0
    assert {h.CLASS for h in hs.for_queue('gpu')} == {'toy/start', 'toy/work'}


def test_emits_must_lead_somewhere():
    class Lost(MessageHandler):
        CLASS, QUEUE, EMITS = 'toy/lost', 'gpu', ('toy/nowhere',)
    with pytest.raises(ValueError, match='toy/nowhere'):
        HandlerSet(TOY + [Lost])


def test_declaration_errors():
    class NoQueue(MessageHandler):
        CLASS = 'toy/x'
    with pytest.raises(ValueError):
        check_declaration(NoQueue)

    class NoClass(MessageHandler):
        QUEUE = 'gpu'
    with pytest.raises(ValueError):
        check_declaration(NoClass)

    class Dup(MessageHandler):
        CLASS, QUEUE = 'toy/work', 'cpu'
    with pytest.raises(ValueError, match='two handlers'):
        HandlerSet(TOY + [Dup])


def test_zero_gpu_queue_set_is_valid():
    assert HandlerSet(TOY).for_queue('cpu')


# ---------------------------------------------------------------------------
# GenericWorker: EMITS checking and the failure message
# ---------------------------------------------------------------------------

def _worker(queue='gpu'):
    hs = HandlerSet(TOY)
    ctx = NodeContext(app='toy', role=f'{queue}_worker', worker_id='w')
    return GenericWorker('w', queue, hs, ctx, broker_socket='/nonexistent'), hs


def test_prepare_results_fills_target_queue_from_the_target_handler():
    w, _ = _worker()
    out = w.prepare_results(ToyStart(), [{'class': 'toy/work'}, {'class': 'toy/join'}])
    assert [m['target_queue'] for m in out] == ['gpu', 'cpu']


def test_prepare_results_normalises_none_single_and_falsy():
    w, _ = _worker()
    assert w.prepare_results(ToyResult(), None) == []
    assert w.prepare_results(ToyWork(), {'class': 'toy/result'})[0]['target_queue'] == 'cpu'
    assert w.prepare_results(ToyStart(), [None, {'class': 'toy/work'}]) == [
        {'class': 'toy/work', 'target_queue': 'gpu'}]


def test_prepare_results_rejects_an_undeclared_message():
    w, _ = _worker()
    with pytest.raises(EmitsViolation, match='EMITS'):
        w.prepare_results(ToyWork(), {'class': 'toy/join'})
    with pytest.raises(EmitsViolation):
        w.prepare_results(ToyWork(), {'class': 'toy/result', 'target_queue': 'gpu'})   # wrong queue


def test_prepare_results_allows_the_ack_without_a_queue():
    w, _ = _worker('cpu')
    out = w.prepare_results(ToyJoin(), ack({'task_id': 1}))
    assert out == [{'class': DAG_ACK, 'task_id': 1}]


def test_report_failure_builds_the_engine_message(monkeypatch):
    w, _ = _worker()
    sent = []
    monkeypatch.setattr(w, '_forward_message', sent.append)
    msg = {'class': 'toy/work', 'task_id': 4, 'pool': 'gfx950', 'category': ['toy', None],
           'attempt': 1, 'payload': {'n': 1}, 'tmpdir': '/x'}
    w._report_failure(msg, Permanent('bad'))
    w._report_failure(msg, ValueError('flaky'))
    (perm, transient) = sent
    assert perm['class'] == TASK_FAILED and perm['target_queue'] == 'cpu' and perm['permanent'] is True
    assert transient['permanent'] is False and 'ValueError: flaky' in transient['error']
    assert perm['task_id'] == 4 and perm['pool'] == 'gfx950' and perm['origin']['tmpdir'] == '/x'


def test_failure_of_the_failure_path_only_releases_the_reader(monkeypatch):
    w, _ = _worker('cpu')
    sent = []
    monkeypatch.setattr(w, '_forward_message', sent.append)
    w._report_failure({'class': TASK_FAILED, 'task_id': 4, 'pool': 'p'}, RuntimeError('db down'))
    assert sent == [{'class': DAG_ACK, 'task_id': 4, 'negative': True}]


def test_a_message_without_a_task_id_is_dropped(monkeypatch):
    w, _ = _worker()
    sent = []
    monkeypatch.setattr(w, '_forward_message', sent.append)
    w._report_failure({'class': 'toy/work'}, RuntimeError('x'))
    assert sent == []


def test_graph_render_lists_every_edge():
    text = graph.render(TOY, ToyStart)
    assert text.splitlines()[0] == 'toy/start  [gpu]'
    for cls in ('toy/work', 'toy/result', 'toy/join', DAG_ACK):
        assert cls in text
    assert 'not reachable' not in text
    assert 'not reachable from the start handler: toy/orphan' in graph.render(
        TOY + [type('Orphan', (MessageHandler,), {'CLASS': 'toy/orphan', 'QUEUE': 'cpu'})], ToyStart)


# ---------------------------------------------------------------------------
# A node, end to end (threads, real sockets)
# ---------------------------------------------------------------------------

@pytest.fixture
def db(monkeypatch):
    fake = FakeDB()
    fake.install(monkeypatch)
    return fake


@pytest.fixture
def run_node(tmp_path, db):
    nodes = []

    def make(**kw):
        node = Node(tmp_path, TOY, ToyStart, db, **kw)
        nodes.append(node)
        return node

    yield make
    for node in nodes:
        node.stop()


def test_a_task_runs_through_the_whole_graph(run_node, db):
    tid = db.add(payload={'n': 4})
    run_node(n_gpu=2)
    assert wait_for(lambda: db.status(tid) == 'done')
    assert sorted(BEHAVIOUR['results']) == [(tid, i) for i in range(4)]
    assert BEHAVIOUR['joined'] == [(tid, [0, 1, 2, 3])]


def test_the_reader_goes_on_to_the_next_task_after_the_ack(run_node, db):
    ids = [db.add(payload={'n': 2}) for _ in range(3)]
    run_node()
    assert wait_for(lambda: all(db.status(i) == 'done' for i in ids))


def test_a_join_with_no_work_completes_immediately(run_node, db):
    tid = db.add(payload={'n': 0})
    run_node()
    assert wait_for(lambda: db.status(tid) == 'done')
    assert BEHAVIOUR['joined'] == [(tid, [])]


def test_by_default_a_handler_exception_fails_the_task_without_retry(run_node, db):
    # DEFAULT_MAX_ATTEMPTS is 1: retry exists but is unused unless a task opts in.
    BEHAVIOUR['fail_start_on_attempt'] = 1
    tid = db.add(payload={'n': 1})
    run_node()
    assert wait_for(lambda: db.status(tid) == 'failed')
    row = db.rows[tid]
    assert row.attempt == 1 and row.errors[0][1] is False


def test_a_transient_handler_exception_is_retried_when_the_task_opts_in(run_node, db):
    BEHAVIOUR['fail_start_on_attempt'] = 1
    tid = db.add(payload={'n': 1}, max_attempts=2)
    run_node()
    assert wait_for(lambda: db.status(tid) == 'done')
    row = db.rows[tid]
    assert row.attempt == 2 and row.errors[0][1] is False and 'transient start failure' in row.errors[0][0]


def test_a_permanent_exception_fails_the_task_without_retry(run_node, db):
    BEHAVIOUR['work_raises'] = Permanent('unsupported entry')
    tid = db.add(payload={'n': 2})
    run_node()
    assert wait_for(lambda: db.status(tid) == 'failed')
    row = db.rows[tid]
    assert row.attempt == 1 and row.errors[0][1] is True
    assert tid in BEHAVIOUR['cleaned']              # the start handler's cleanup ran


def test_exhausted_attempts_fail_the_task_and_never_leak_the_reader_slot(run_node, db):
    BEHAVIOUR['work_raises'] = RuntimeError('boom')
    bad = db.add(payload={'n': 1}, max_attempts=2)
    run_node()
    assert wait_for(lambda: db.status(bad) == 'failed')
    assert db.rows[bad].attempt == 2
    # The single reader is free again: a good task still runs.
    BEHAVIOUR['work_raises'] = None
    good = db.add(payload={'n': 1})
    assert wait_for(lambda: db.status(good) == 'done')


def test_an_undeclared_emit_fails_the_task(run_node, db):
    BEHAVIOUR['emit_wrong'] = True
    tid = db.add(payload={'n': 1}, max_attempts=1)
    run_node()
    assert wait_for(lambda: db.status(tid) == 'failed')
    assert 'EMITS' in db.rows[tid].errors[0][0]


def test_a_zero_gpu_node_fails_the_task_instead_of_hanging(run_node, db):
    tid = db.add(payload={'n': 1})
    run_node(queues=('cpu',))
    assert wait_for(lambda: db.status(tid) == 'failed')
    error, permanent = db.rows[tid].errors[0]
    assert permanent and "queue 'gpu'" in error and "['cpu']" in error
    # and the reader is free for the next one
    other = db.add(payload={'n': 1})
    assert wait_for(lambda: db.status(other) == 'failed')


def test_a_zero_cpu_node_fails_the_task_instead_of_hanging(run_node, db):
    tid = db.add(payload={'n': 1})
    run_node(queues=('gpu',))
    assert wait_for(lambda: db.status(tid) == 'failed')
    assert "queue 'cpu'" in db.rows[tid].errors[0][0]


def test_the_reader_renews_the_lease_while_the_task_runs(run_node, db):
    BEHAVIOUR['work_sleep'] = 1.0
    tid = db.add(payload={'n': 1})
    run_node(lease_s=0.6)                        # renews about every 0.2 s
    assert wait_for(lambda: db.status(tid) == 'done', timeout=15)
    assert db.rows[tid].renewals >= 3


def test_a_lost_lease_stops_the_renewals_but_not_the_wait(run_node, db):
    BEHAVIOUR['work_sleep'] = 1.0
    tid = db.add(payload={'n': 1})
    db.rows[tid].lease_lost = True
    run_node(lease_s=0.6)
    assert wait_for(lambda: db.status(tid) == 'done', timeout=15)
    assert db.rows[tid].renewals == 1            # the first renewal reported the loss; no more
