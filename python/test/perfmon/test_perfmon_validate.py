# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The pure parts of the phase 1 driver (validate.py) and its job plans. No
GPU, runner or database."""

import dataclasses
import json
from pathlib import Path

import pytest

from aotriton.perfmon import validate, validate_plans
from aotriton.perfmon.registry import make_perf_desc


@pytest.fixture(scope='module')
def desc():
    return make_perf_desc('flash')


def test_parse_reply_ok():
    r = validate.parse_reply('OK {"median": 12.5, "seed": 7}\n')
    assert r == {'status': 'ok', 'data': {'median': 12.5, 'seed': 7}}


def test_parse_reply_unavailable_and_error():
    assert validate.parse_reply('UNAVAILABLE no gqa here') == {'status': 'unavailable', 'message': 'no gqa here'}
    assert validate.parse_reply('boom')['status'] == 'error'
    assert validate.parse_reply('OK {bad')['status'] == 'error'


def test_read_jobs_validates(tmp_path):
    good = {'subject': 's', 'iface': 'op_attn_fwd', 'backend': '-', 'entry_pon': "a=1"}
    jobs = validate.read_jobs(['# c\n', '\n', json.dumps(good) + '\n'])
    assert jobs == [good]
    with pytest.raises(ValueError, match='missing "iface"'):
        validate.read_jobs([json.dumps({'subject': 's', 'backend': '-', 'entry_pon': 'a=1'})])
    with pytest.raises(ValueError, match='entry'):
        validate.read_jobs([json.dumps({'subject': 's', 'iface': 'i', 'backend': '-'})])


def test_group_by_subject_keeps_order():
    jobs = [{'subject': 'a', 'n': 1}, {'subject': 'a', 'n': 2}, {'subject': 'b', 'n': 3}, {'subject': 'a', 'n': 4}]
    groups = validate.group_by_subject(jobs)
    assert [(s, [j['n'] for j in g]) for s, g in groups] == [('a', [1, 2]), ('b', [3]), ('a', [4])]


def test_summarize():
    s = validate.summarize([100.0, 101.0, 99.0])
    assert s['n'] == 3 and s['min'] == 99.0 and s['max'] == 101.0
    assert s['spread_pct'] == pytest.approx(2 / 99 * 100)
    assert validate.summarize([]) == {}
    assert validate.summarize([5.0])['spread_pct'] == 0.0


class FakeRunner:
    def __init__(self, replies):
        self.replies = replies
        self.sent = []
        self.keepalives = 0

    def command(self, text, timeout):
        self.sent.append(text)
        return self.replies.pop(0)


def test_run_job_all_expands_backends():
    runner = FakeRunner([{'status': 'ok', 'data': {'backends': ['x', 'y']}},
                         {'status': 'ok', 'data': {'median': 1.0}},
                         {'status': 'unavailable', 'message': 'nope'},
                         {'status': 'error', 'message': 'bad'}])
    job = {'subject': 's', 'iface': 'op_attn_fwd', 'backend': 'all', 'entry_pon': 'a=1', 'label': 'L'}
    records = validate.run_job(runner, job, 5)
    assert runner.sent == ['enumerate a=1 op_attn_fwd', 'measure a=1 op_attn_fwd -',
                           'measure a=1 op_attn_fwd x', 'measure a=1 op_attn_fwd y']
    assert [(r['backend'], r['status']) for r in records] == [('-', 'ok'), ('x', 'unavailable'), ('y', 'error')]
    assert records[0]['stats'] == {'median': 1.0} and records[0]['label'] == 'L'
    assert records[1]['message'] == 'nope' and 'entry_pon' not in records[0]


def test_run_job_stops_on_dead_runner():
    runner = FakeRunner([{'status': 'error', 'message': 'runner exited with 1'}])
    job = {'subject': 's', 'iface': 'op_attn_fwd', 'backend': '-', 'entry_pon': 'a=1'}
    assert len(validate.run_job(runner, job, 5)) == 1


def test_noise_plan_entries(desc):
    jobs = validate_plans.noise_jobs(desc, ['s'])
    assert len(jobs) == len(validate_plans.NOISE_SEQLENS) * 2
    for job in jobs:
        assert job['entry']['T_io'] == 'bf16' and job['entry']['hdim_qk'] == 128
    # the job's entry round-trips into the runner's PON
    assert validate.job_pon(jobs[0]).startswith("T_io='bf16';hdim_qk=128")


def test_feature_plan_covers_every_feature(desc):
    jobs = validate_plans.feature_jobs(desc, ['s'])
    labels = {j['label'] for j in jobs}
    assert labels == {f'feature:{n}' for n in validate_plans.FEATURES}
    by = {j['label']: j['entry'] for j in jobs}
    assert by['feature:bias_b1qk']['bias_shape'] == 'b1qk' and by['feature:bias_b1qk']['bias_type'] == 1
    assert by['feature:gqa4']['gqa_ratio'] == 4
    assert by['feature:dropout']['dropout_p'] == 0.1
    assert all(j['entry']['batch'] <= validate_plans.FEATURE_MAX_BATCH for j in jobs)


def test_llm_plan_skips_pending(desc):
    jobs = validate_plans.llm_jobs(desc, ['s'])
    cases = [c for c in desc.llm_cases() if not c.pending]
    assert len(jobs) == sum(len(desc.phase_ifaces(c.phase)) for c in cases)


def test_size_plan_has_small_and_large(desc):
    jobs = validate_plans.size_jobs(desc, ['s'], hdims=(64,))
    seqlens = {j['entry']['seqlen_q'] for j in jobs}
    assert seqlens == {16, 16384}


def test_aiter_plan_forces_aiter(desc):
    jobs = validate_plans.aiter_jobs(desc, ['s'], seqlens=(64,))
    assert {j['backend'] for j in jobs} == {'-', 'aiter'}
    assert {j['entry']['causal'] for j in jobs} == {0, 1}


class RestartableRunner(FakeRunner):
    restarted = 0

    def restart(self):
        self.restarted += 1


def test_run_job_probe_counts_timeout_as_started_and_restarts():
    timeout = {'status': 'timeout', 'message': 'timeout after 5s', 'dead': True}
    runner = RestartableRunner([{'status': 'ok', 'data': {'backends': ['x']}},
                                dict(timeout),
                                dict(timeout)])
    job = {'subject': 's', 'iface': 'op_attn_fwd', 'backend': 'all', 'entry_pon': 'a=1'}
    records = validate.run_job(runner, job, 5, probe=True)
    assert [(r['backend'], r['status']) for r in records] == [('-', 'started'), ('x', 'started')]
    assert runner.restarted == 2


def test_run_job_timeout_without_probe_is_an_error():
    runner = RestartableRunner([{'status': 'timeout', 'message': 'timeout after 5s', 'dead': True}])
    job = {'subject': 's', 'iface': 'op_attn_fwd', 'backend': '-', 'entry_pon': 'a=1'}
    assert [r['status'] for r in validate.run_job(runner, job, 5)] == ['error']
    assert runner.restarted == 0
