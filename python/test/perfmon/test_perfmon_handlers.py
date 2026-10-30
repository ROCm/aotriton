# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""perfmon's handler graph on the localq test harness (the production broker,
workers and PG reader in threads, an in-memory fake of the pq calls), with a
scripted runner in place of the native one. CPU-only."""

import json

import pytest

from aotriton.pgdef.exaid import ExaidSubprocessNotOK
from aotriton.pgdef.localq import graph
from aotriton.pgdef.localq.handlers import DAG_ACK, HandlerSet, NodeContext
from aotriton.pgdef.pq import tasks as pgtasks
from aotriton.perfmon import exaid_perfmon, handlers
from aotriton.perfmon.registry import make_perf_desc

from localq_harness import FakeDB, Node, wait_for

GiB = 1 << 30
SUBJECT = 'aotriton-0.13b+rocm10.1.0'
BACKENDS = {'op_attn_fwd': ['triton', 'aiter'], 'op_attn_bwd': ['triton_split', 'triton_fuse']}
STATS = {'timing_method': 'hipgraph_ev100', 'n': 100, 'l2_flush': True, 'seed': 7, 'median': 12.5,
         'mean': 12.6, 'stddev': 0.1, 'min': 12.0, 'p05': 12.1, 'p95': 13.0, 'thermal': {'gate': False}}
PLATFORM = {'pmon_abi_version': 2, 'gcn_arch': 'gfx950', 'vram_total_bytes': 288 * GiB,
            'pci_device_id': '0x75a0', 'pci_bdf': '0000:05:00.0', 'core_arches': ['gfx942', 'gfx950'],
            'harness': 'abc1234', 'thermal_gate': 'off'}


class ScriptedRunner:
    """Stands in for ExaidPerfmonRunner; the test sets SCRIPT before the node runs."""
    SCRIPT: dict = {}
    LOG: list = []

    def __init__(self, module, gpu_id):
        self.module, self.gpu_id = module, gpu_id

    def use_subject(self, subject, arch=None):
        self.LOG.append(('use_subject', subject, arch))

    def platform(self):
        return {**PLATFORM, **self.SCRIPT.get('platform', {})}

    def enumerate(self, pon, iface):
        self.LOG.append(('enumerate', iface))
        return list(BACKENDS[iface])

    def measure(self, pon, iface, backend):
        self.LOG.append(('measure', iface, backend))
        how = self.SCRIPT.get((iface, backend), 'ok')
        if how == 'unavailable':
            raise exaid_perfmon.BackendUnavailable('no such kernel on this arch')
        if how == 'notok':
            raise ExaidSubprocessNotOK('ERROR bad launch', 'stderr text')
        if how == 'crash':
            raise OSError(5, 'runner died')
        return dict(STATS)

    def exit(self):
        pass


@pytest.fixture(autouse=True)
def _reset():
    ScriptedRunner.SCRIPT = {}
    ScriptedRunner.LOG = []
    yield


@pytest.fixture
def saved(monkeypatch):
    """Reports written by the result handler, and the cookie appends."""
    out = {'reports': [], 'cookie': []}

    def save_measurement(conn, ref, report, *, attempt):
        out['reports'].append((ref, report, attempt))

    def append_cookie(conn, ref, key, values):
        out['cookie'].append((ref, key, list(values)))

    monkeypatch.setattr(handlers, 'save_measurement', save_measurement)
    monkeypatch.setattr(pgtasks, 'append_cookie', append_cookie)
    return out


@pytest.fixture
def db(monkeypatch):
    fake = FakeDB()
    fake.install(monkeypatch)
    return fake


@pytest.fixture
def run_node(tmp_path, db):
    nodes = []

    def make(**kw):
        node = Node(tmp_path,
                    handlers.HANDLERS,
                    handlers.START,
                    db,
                    runners={'perfmon': ScriptedRunner},
                    category=('perfmon', None),
                    **kw)
        nodes.append(node)
        return node

    yield make
    for node in nodes:
        node.stop()


def _entry_dict(**patch):
    desc = make_perf_desc('flash')
    axes = {**desc.entry_sets()['prime'], 'hdim': [64], 'causal': [0], 'seqlen': [1024],
            'T_io': ['bf16'], 'dropout_p': [0.0]}
    entry = list(desc.entries(axes))[0]
    return {**desc.entry_to_dict(entry), **patch}


def _add(db, **extra):
    payload = {'module': 'flash', 'subject': SUBJECT, 'entry': _entry_dict(), **extra}
    return db.add(pool='gfx950', payload=payload, category=('perfmon', None))


def _by_pair(saved):
    return {(r['iface'], r['backend']): r for _, r, _ in saved['reports']}


ALL_PAIRS = {(i, b) for i, names in BACKENDS.items() for b in [None] + names}


# -- the declaration ---------------------------------------------------------------------------

def test_the_declaration_is_valid_and_the_graph_is_the_documented_one():
    HandlerSet(handlers.HANDLERS)
    text = graph.render(handlers.HANDLERS, handlers.START)
    assert text.splitlines()[0] == 'perfmon/start  [gpu]'
    for cls in (handlers.PROBE, handlers.MEASURE, handlers.RESULT, handlers.FINALIZE, DAG_ACK):
        assert cls in text
    assert 'not reachable' not in text
    assert handlers.START is handlers.PerfmonStartHandler
    queues = {h.CLASS: h.QUEUE for h in handlers.HANDLERS}
    assert queues == {'perfmon/start': 'gpu', 'perfmon/probe': 'gpu', 'perfmon/measure': 'gpu',
                      'perfmon/result': 'cpu', 'perfmon/finalize': 'cpu'}


# -- the whole graph --------------------------------------------------------------------------------

def test_every_backend_of_every_iface_is_measured_and_reported(run_node, db, saved):
    tid = _add(db)
    run_node()
    assert wait_for(lambda: db.status(tid) == 'done')
    reports = _by_pair(saved)
    assert set(reports) == ALL_PAIRS
    assert {r['status'] for r in reports.values()} == {'OK'}
    ok = reports[('op_attn_bwd', 'triton_fuse')]
    assert ok['payload']['stats']['median'] == 12.5
    assert ok['payload']['subject'] == SUBJECT and ok['payload']['backend_forced'] is True
    assert reports[('op_attn_fwd', None)]['payload']['backend_forced'] is False
    assert ok['payload']['harness'] == {'commit': 'abc1234', 'arches': ['gfx942', 'gfx950']}
    assert ok['payload']['platform']['arch'] == 'gfx950'
    assert ok['payload']['platform']['variant'] == 'any'
    assert ok['payload']['platform']['pci_device_id'] == '0x75a0'
    assert ok['payload']['l2_flush'] == 1
    # every pair is recorded done in the appcookie
    assert {tuple(v[0]) for _, _, v in saved['cookie']} == ALL_PAIRS
    assert len(saved['cookie']) == len(ALL_PAIRS)
    assert all(key == 'done' for _, key, _ in saved['cookie'])
    json.dumps([r for _, r, _ in saved['reports']])     # reports are JSON


def test_notok_crash_and_unavailable_are_results_not_task_failures(run_node, db, saved):
    ScriptedRunner.SCRIPT = {('op_attn_fwd', 'aiter'): 'unavailable',
                             ('op_attn_bwd', 'triton_fuse'): 'notok',
                             ('op_attn_bwd', 'triton_split'): 'crash'}
    tid = _add(db)
    run_node()
    assert wait_for(lambda: db.status(tid) == 'done')
    reports = _by_pair(saved)
    assert reports[('op_attn_fwd', 'aiter')]['status'] == 'excluded'
    assert 'no such kernel' in reports[('op_attn_fwd', 'aiter')]['error']['reason']
    assert reports[('op_attn_bwd', 'triton_fuse')]['status'] == 'NotOK'
    assert reports[('op_attn_bwd', 'triton_fuse')]['error']['stderr'] == 'stderr text'
    assert reports[('op_attn_bwd', 'triton_split')]['status'] == 'crash'
    assert reports[('op_attn_bwd', 'triton_split')]['payload'].get('stats') is None
    assert reports[('op_attn_fwd', None)]['status'] == 'OK'
    assert db.rows[tid].errors == []


def test_pairs_in_the_appcookie_are_skipped(run_node, db, saved, monkeypatch):
    done = [['op_attn_fwd', None], ['op_attn_fwd', 'triton'], ['op_attn_bwd', None]]
    claim = db.claim

    def claim_with_cookie(*a, **k):
        out = claim(*a, **k)
        for task in out:
            task.appcookie = {'done': done}
        return out

    monkeypatch.setattr(pgtasks, 'claim', claim_with_cookie)
    tid = _add(db)
    run_node()
    assert wait_for(lambda: db.status(tid) == 'done')
    assert set(_by_pair(saved)) == ALL_PAIRS - {(i, b) for i, b in map(tuple, done)}
    measured = {x[1:] for x in ScriptedRunner.LOG if x[0] == 'measure'}
    assert measured == set(_by_pair(saved))


def test_an_entry_that_does_not_fit_the_gpu_is_excluded_and_nothing_runs(run_node, db, saved):
    ScriptedRunner.SCRIPT = {'platform': {'vram_total_bytes': 4 * GiB}}
    tid = _add(db)
    run_node()
    assert wait_for(lambda: db.status(tid) == 'done')
    reports = _by_pair(saved)
    assert set(reports) == ALL_PAIRS
    assert {r['status'] for r in reports.values()} == {'excluded'}
    assert 'GiB' in reports[('op_attn_fwd', None)]['error']['reason']
    assert not [x for x in ScriptedRunner.LOG if x[0] == 'measure']


def test_an_llm_case_measured_forward_only_has_no_backward_pairs(run_node, db, saved):
    tid = _add(db, ifaces=['op_attn_fwd'])
    run_node()
    assert wait_for(lambda: db.status(tid) == 'done')
    assert {i for i, _ in _by_pair(saved)} == {'op_attn_fwd'}
    assert not [x for x in ScriptedRunner.LOG if x == ('enumerate', 'op_attn_bwd')]


def test_the_node_subpool_is_recorded_in_every_report(run_node, db, saved):
    tid = _add(db)
    run_node(enrich=lambda conn, task: {'node_subpool': 'pool_a'})
    assert wait_for(lambda: db.status(tid) == 'done')
    assert {r['payload']['platform']['node_subpool'] for _, r, _ in saved['reports']} == {'pool_a'}


def test_a_missing_subject_fails_the_task_not_a_measurement(run_node, db, saved, monkeypatch):
    def use_subject(self, subject, arch=None):
        raise FileNotFoundError(f'subject {subject} is not installed')

    monkeypatch.setattr(ScriptedRunner, 'use_subject', use_subject)
    tid = _add(db)
    run_node()
    assert wait_for(lambda: db.status(tid) == 'failed')
    assert saved['reports'] == []


def test_a_bad_entry_is_a_permanent_failure(run_node, db, saved):
    tid = db.add(pool='gfx950',
                 payload={'module': 'flash', 'subject': SUBJECT, 'entry': {'T_io': 'bf16'}},
                 category=('perfmon', None))
    run_node()
    assert wait_for(lambda: db.status(tid) == 'failed')
    assert db.rows[tid].errors[0][1] is True


# -- the runner ---------------------------------------------------------------------------------------

class FakeProxy:
    instances = []

    def __init__(self, spawn_argv, *a, **k):
        self.spawn_argv, self.writes, self.replies, self.shut = spawn_argv, [], [], 0
        FakeProxy.instances.append(self)

    def write(self, *words):
        self.writes.append(' '.join(str(w) for w in words))

    def readinfo(self, timeout=None):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def shutdown(self):
        self.shut += 1


@pytest.fixture
def runner(monkeypatch, tmp_path):
    FakeProxy.instances = []
    monkeypatch.setattr(exaid_perfmon, 'ExaidProxy', FakeProxy)
    launcher = tmp_path / 'launch_runner.sh'
    monkeypatch.setattr(exaid_perfmon, 'launcher', lambda: launcher)
    return exaid_perfmon.ExaidPerfmonRunner('flash', 3)


def test_the_runner_spawns_the_launcher_for_the_current_subject(runner):
    runner.use_subject('aotriton-0.13b+rocm10.1.0')
    argv = runner.proxy.spawn_argv()
    assert argv[1:] == ['--subject', 'aotriton-0.13b+rocm10.1.0', '--module', 'flash', '--gpu', '3']
    assert '--arch' not in argv         # no arch given: the launcher resolves it


def test_the_runner_passes_the_arch_to_the_launcher(runner, monkeypatch, tmp_path):
    monkeypatch.setenv('AOTRITON_WORKDIR', str(tmp_path))
    (tmp_path / 'installed' / 'perfmon' / 'gfx950' / 'aotriton-0.13b+rocm10.1.0').mkdir(parents=True)
    runner.use_subject('aotriton-0.13b+rocm10.1.0', arch='gfx950')
    argv = runner.proxy.spawn_argv()
    assert argv[-2:] == ['--arch', 'gfx950']


def test_the_runner_keeps_two_processes_and_evicts_the_least_recently_used(runner):
    a, b, c = 'aotriton-0.13b+rocm10.1.0', 'aotriton-0.14.2b+rocm10.1.0', 'aotriton-0.15b+rocm10.1.0'
    runner.use_subject(a)
    pa = runner.proxy
    pa.replies = [json.dumps({'vram_total_bytes': 1})]
    assert runner.platform() == {'vram_total_bytes': 1}
    runner.platform()                                           # cached: no second reply needed
    assert pa.writes == ['platform']
    runner.use_subject(a)                                       # same subject: nothing happens
    assert pa.shut == 0
    runner.use_subject(b)
    pb = runner.proxy
    assert pb is not pa and pa.shut == 0                        # a stays alive
    runner.use_subject(a)                                       # switching back is free
    assert runner.proxy is pa and pa.shut == 0 and pb.shut == 0
    assert runner.platform() == {'vram_total_bytes': 1}         # platform kept too
    runner.use_subject(c)                                       # b is the least recently used
    assert pb.shut == 1 and pa.shut == 0
    assert runner.proxy.spawn_argv()[2] == c
    runner.exit()
    assert pa.shut == 1 and runner.proxy.shut == 1


def test_the_same_subject_on_two_arches_is_two_runners(runner, monkeypatch, tmp_path):
    monkeypatch.setenv('AOTRITON_WORKDIR', str(tmp_path))
    s = 'aotriton-0.13b+rocm10.1.0'
    for arch in ('gfx942', 'gfx950'):
        (tmp_path / 'installed' / 'perfmon' / arch / s).mkdir(parents=True)
    runner.use_subject(s, arch='gfx942')
    p1 = runner.proxy
    runner.use_subject(s, arch='gfx950')
    assert runner.proxy is not p1 and runner.proxy.spawn_argv()[-1] == 'gfx950'


def test_the_runner_without_a_subject_refuses_to_launch(runner):
    with pytest.raises(RuntimeError):
        runner.proxy.spawn_argv()


def test_measure_wire_form_and_unavailable_reply(runner):
    runner.use_subject('aotriton-0.13b+rocm10.1.0')
    proxy = runner.proxy
    proxy.replies = [json.dumps({'median': 1.0}), ExaidSubprocessNotOK('UNAVAILABLE no aiter here', ''),
                     ExaidSubprocessNotOK('ERROR boom', '')]
    assert runner.measure('PON', 'op_attn_fwd', None) == {'median': 1.0}
    with pytest.raises(exaid_perfmon.BackendUnavailable, match='no aiter'):
        runner.measure('PON', 'op_attn_fwd', 'aiter')
    with pytest.raises(ExaidSubprocessNotOK):
        runner.measure('PON', 'op_attn_bwd', 'triton_fuse')
    assert proxy.writes == ['measure PON op_attn_fwd -', 'measure PON op_attn_fwd aiter',
                            'measure PON op_attn_bwd triton_fuse']


def test_enumerate_returns_the_backend_names(runner):
    runner.use_subject('aotriton-0.13b+rocm10.1.0')
    runner.proxy.replies = [json.dumps({'backends': ['triton', 'aiter']}), json.dumps({'backends': []})]
    assert runner.enumerate('PON', 'op_attn_fwd') == ['triton', 'aiter']
    assert runner.enumerate('PON', 'op_attn_bwd') == []


def test_use_subject_checks_the_install_when_given_an_arch(runner, monkeypatch, tmp_path):
    from aotriton.pgdef import paths
    monkeypatch.setenv('AOTRITON_WORKDIR', str(tmp_path))
    with pytest.raises(FileNotFoundError):
        runner.use_subject('aotriton-0.13b+rocm10.1.0', arch='gfx950')
    (tmp_path / 'installed' / 'perfmon' / 'gfx950' / 'aotriton-0.13b+rocm10.1.0').mkdir(parents=True)
    runner.use_subject('aotriton-0.13b+rocm10.1.0', arch='gfx950')
    assert runner.subject == 'aotriton-0.13b+rocm10.1.0'


def test_the_node_gives_the_runner_the_module_only(monkeypatch):
    """One runner per (name, module): never per subject."""
    ctx = NodeContext(app='perfmon',
                      role='gpu_worker',
                      worker_id='g0',
                      gpu_id=2,
                      runner_types={'perfmon': ScriptedRunner})
    a, b = ctx.runner('perfmon', 'flash'), ctx.runner('perfmon', 'flash')
    assert a is b and (a.module, a.gpu_id) == ('flash', 2)
