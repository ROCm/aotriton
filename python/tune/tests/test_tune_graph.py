# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The tune application's handler graph, run end to end on an in-process node
with a scripted fake `tuner` runner (no GPU, no PostgreSQL, no torch).

The expected report sets are what the pre-pgdef handlers produced for the same
scripted runner replies (per (iface, impl): OK / crash / NotOK, op-level
single-backend skipping, kernel-only max_hsaco); the same scenarios were also
run through the old handlers once, by hand, when the graph was ported.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'test' / 'pgdef'))   # localq_harness, pgfake

pytest.importorskip('psycopg')

from aotriton.pgdef.exaid import ExaidSubprocessNotOK
from aotriton.pgdef.localq import graph
from aotriton.tune import handlers as tune_handlers
from aotriton.tune.handlers import HANDLERS, START

from localq_harness import FakeDB, Node, wait_for


class ScriptedRunner:
    """Stands in for ExaidTunerWorker. `script` steers it per scenario."""
    script: dict = {}
    instances: list = []

    def __init__(self, module_name, gpu_id):
        self.module, self.gpu_id = module_name, gpu_id
        self.calls = []
        ScriptedRunner.instances.append(self)

    def get_tmpfs_for(self, entry):
        return Path(self.script['tmp']) / f"{entry['hdim']}-{entry['seq']}"

    def prepare_data(self, entry, workdir, extra_im_texts=()):
        self.script.setdefault('prepared', []).append((entry, list(extra_im_texts)))
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / 'entry.json').write_text('{}')
        err = self.script.get('prepare_error')
        if err:
            raise err

    def probe(self, workdir, arch=None, tuning_level=None):
        self.script.setdefault('probed', []).append((arch, tuning_level))
        return self.script['probe'][tuning_level]

    def benchmark(self, workdir, impl_selector):
        outcome = self.script.get('bench', {}).get((impl_selector.iface_name, impl_selector.impl_index), 'ok')
        if outcome == 'crash':
            raise OSError(11, 'segfault')
        if outcome == 'notok':
            raise ExaidSubprocessNotOK('Error', 'bad impl')
        return {'impl_selection': {'iface_name': impl_selector.iface_name,
                                   'impl_index': impl_selector.impl_index}, 'times': [1.0]}

    def exit(self):
        pass


@pytest.fixture
def scenario(tmp_path, monkeypatch):
    ScriptedRunner.script = {'tmp': str(tmp_path / 'tmpfs')}
    ScriptedRunner.instances = []
    db = FakeDB()
    db.install(monkeypatch)
    saved = []
    monkeypatch.setattr(tune_handlers, 'save_tuning_result',
                        lambda conn, ref, report, *, attempt: saved.append((ref, report, attempt)))
    nodes = []

    def make(**kw):
        node = Node(tmp_path, HANDLERS, START, db, runners={'tuner': ScriptedRunner},
                    category=('tune', kw.pop('level', 'kernel')), **kw)
        nodes.append(node)
        return node

    yield db, saved, make
    for n in nodes:
        n.stop()


def _add(db, level, max_hsaco=None, hdim=64, seq=128):
    payload = {'module': 'flash', 'entry': {'hdim': hdim, 'seq': seq}}
    if max_hsaco is not None:
        payload['max_hsaco'] = {'*': max_hsaco}
    return db.add(payload=payload, category=('tune', level))


def _reports(saved):
    return {(r['iface_name'], r['impl_index']): r for _, r, _ in saved}


KERNEL_PROBE = {'attn_fwd': [{}, {}, {}], 'bwd_kernel_dq': [{}, {}]}
OP_PROBE = {'op.attn_fwd': [{}, {}, {}], 'op.bwd': [{}]}


def test_kernel_level_reports_every_hsaco(scenario):
    db, saved, make = scenario
    ScriptedRunner.script['probe'] = {'kernel': KERNEL_PROBE}
    tid = _add(db, 'kernel')
    make(n_gpu=2, level='kernel')
    assert wait_for(lambda: db.status(tid) == 'done')
    reports = _reports(saved)
    assert sorted(reports) == [('attn_fwd', 0), ('attn_fwd', 1), ('attn_fwd', 2),
                               ('bwd_kernel_dq', 0), ('bwd_kernel_dq', 1)]
    assert {r['result'] for r in reports.values()} == {'OK'}
    assert all(r['tuning_level'] == 'kernel' and r['complete_on_gpu'] in (0, 1) for r in reports.values())
    assert ScriptedRunner.script['probed'] == [('gfx950', 'kernel')]


def test_max_hsaco_caps_the_kernel_sweep(scenario):
    db, saved, make = scenario
    ScriptedRunner.script['probe'] = {'kernel': KERNEL_PROBE}
    tid = _add(db, 'kernel', max_hsaco=2)
    make(level='kernel')
    assert wait_for(lambda: db.status(tid) == 'done')
    assert sorted(_reports(saved)) == [('attn_fwd', 0), ('attn_fwd', 1),
                                       ('bwd_kernel_dq', 0), ('bwd_kernel_dq', 1)]


def test_op_level_skips_single_backend_interfaces_and_ignores_max_hsaco(scenario):
    db, saved, make = scenario
    ScriptedRunner.script['probe'] = {'op': OP_PROBE}
    tid = _add(db, 'op', max_hsaco=1)
    make(level='op')
    assert wait_for(lambda: db.status(tid) == 'done')
    assert sorted(_reports(saved)) == [('attn_fwd', 0), ('attn_fwd', 1), ('attn_fwd', 2)]
    assert {r['tuning_level'] for _, r, _ in saved} == {'op'}


def test_a_crashed_or_rejected_benchmark_is_a_result_not_a_failure(scenario):
    db, saved, make = scenario
    ScriptedRunner.script['probe'] = {'kernel': {'attn_fwd': [{}, {}, {}]}}
    ScriptedRunner.script['bench'] = {('attn_fwd', 1): 'crash', ('attn_fwd', 2): 'notok'}
    tid = _add(db, 'kernel')
    make(level='kernel')
    assert wait_for(lambda: db.status(tid) == 'done')
    r = _reports(saved)
    assert r[('attn_fwd', 0)]['result'] == 'OK' and r[('attn_fwd', 0)]['error'] is None
    assert (r[('attn_fwd', 1)]['result'], r[('attn_fwd', 1)]['error']) == ('crash', {'errno': 11, 'stderr': 'segfault'})
    assert (r[('attn_fwd', 2)]['result'], r[('attn_fwd', 2)]['error']) == (
        'NotOK', {'stdout': 'Error', 'stderr': 'bad impl'})
    assert r[('attn_fwd', 1)]['result_data'] is None


def test_the_tmpdir_is_removed_when_the_task_is_done(scenario, tmp_path):
    db, saved, make = scenario
    ScriptedRunner.script['probe'] = {'kernel': {'attn_fwd': [{}]}}
    tid = _add(db, 'kernel', hdim=32, seq=7)
    make(level='kernel')
    assert wait_for(lambda: db.status(tid) == 'done')
    assert wait_for(lambda: not (tmp_path / 'tmpfs' / '32-7').exists())


def test_extra_unit_tests_reach_prepare_data(scenario):
    db, saved, make = scenario
    ScriptedRunner.script['probe'] = {'kernel': {'attn_fwd': [{}]}}
    tid = _add(db, 'kernel')
    make(level='kernel', enrich=lambda conn, task: {'extra_im_texts': ['a=1', 'b=2']})
    assert wait_for(lambda: db.status(tid) == 'done')
    assert ScriptedRunner.script['prepared'][0][1] == ['a=1', 'b=2']


def test_a_rejected_preprocess_fails_permanently_and_frees_the_tmpdir(scenario, tmp_path):
    db, saved, make = scenario
    ScriptedRunner.script['prepare_error'] = ExaidSubprocessNotOK('Error', 'cannot parse entry')
    tid = _add(db, 'kernel', hdim=16, seq=3)
    make(level='kernel')
    assert wait_for(lambda: db.status(tid) == 'failed')
    row = db.rows[tid]
    assert row.attempt == 1 and row.errors[0][1] is True and 'Preprocess failed' in row.errors[0][0]
    assert not (tmp_path / 'tmpfs' / '16-3').exists()
    assert saved == []


def test_a_crashed_runner_in_preprocess_is_retried(scenario):
    db, saved, make = scenario
    ScriptedRunner.script['prepare_error'] = OSError(32, 'runner died')
    tid = _add(db, 'kernel')
    make(level='kernel')
    assert wait_for(lambda: db.status(tid) == 'failed')
    assert db.rows[tid].attempt == db.rows[tid].max_attempts
    assert all(not permanent for _, permanent in db.rows[tid].errors)


def test_a_probe_with_nothing_to_tune_completes_the_task(scenario):
    db, saved, make = scenario
    ScriptedRunner.script['probe'] = {'op': {'op.attn_fwd': [{}]}}        # one backend: skipped
    tid = _add(db, 'op')
    make(level='op')
    assert wait_for(lambda: db.status(tid) == 'done')
    assert saved == []


def test_each_level_is_probed_with_its_own_filter(scenario):
    db, saved, make = scenario
    ScriptedRunner.script['probe'] = {'kernel': {'attn_fwd': [{}]}, 'op': {'op.attn_fwd': [{}, {}]}}
    k = _add(db, 'kernel')
    make(level='kernel')
    assert wait_for(lambda: db.status(k) == 'done')
    assert ScriptedRunner.script['probed'] == [('gfx950', 'kernel')]


def test_the_graph_is_declared_and_printable():
    text = graph.render(HANDLERS, START)
    for cls in ('tune/start', 'tune/preprocess', 'tune/probe', 'tune/impl', 'tune/impl_result',
                'tune/postprocess', 'pgdef/dag_ack'):
        assert cls in text
    assert 'not reachable' not in text
    # the same listing, from the command line entry point's loader
    assert graph.load_application('tune') == (HANDLERS, START)
