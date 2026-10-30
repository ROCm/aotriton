# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The perfmon application's handler graph.

A task is one entry of one subject (payload {module, subject, entry}, plus
`ifaces` for an LLM case measured forward only) on one arch (the pool). Its
category is ('perfmon', None). `python -m aotriton.pgdef.localq.graph perfmon`
prints the graph from the declarations below:

    perfmon/start (gpu) --> perfmon/probe (gpu) --+--> perfmon/measure x N (gpu) --> perfmon/result (cpu)
                                                  |
                                                  +--> perfmon/finalize (cpu) --> dag_ack
                                                       (fan-in over perfmon/result)

The probe also sends perfmon/result directly for the points it excludes
without measuring (the entry does not fit this GPU).

A measurement that fails is a result, not a task failure: `NotOK` (the runner
rejected it), `crash` (the runner died or timed out) or `excluded` (the
backend cannot run here). Only infrastructure errors (a subject missing on
the node, a runner that cannot start or enumerate) fail the task, through the
engine's single failure path.

perfmon selects and observes: each iface is measured unforced (backend None)
and forced to each backend the subject lists. Which backend the unforced
operator picked is never recorded.
"""

import logging
from functools import lru_cache

from aotriton.pgdef.exaid import ExaidSubprocessNotOK
from aotriton.pgdef.localq import semaphore
from aotriton.pgdef.localq.handlers import (DAG_ACK, MessageHandler, Permanent, StartHandler, ack,
                                            child, owner_of, task_ref)
from aotriton.pgdef.pq import tasks as pgtasks

from .exaid_perfmon import BackendUnavailable, host_facts
from .pq.measurements import ANY_VARIANT, save_measurement
from .registry import make_perf_desc

logger = logging.getLogger(__name__)

START_CLASS = 'perfmon/start'
PROBE = 'perfmon/probe'
MEASURE = 'perfmon/measure'
RESULT = 'perfmon/result'
FINALIZE = 'perfmon/finalize'

RUNNER = 'perfmon'

#: The appcookie key listing the [iface, backend] pairs already reported.
DONE = 'done'


@lru_cache(maxsize=None)
def perf_desc(module: str):
    return make_perf_desc(module)


def _runner_failure(what: str, e: Exception) -> Exception:
    """The runner rejecting a request is permanent; the runner dying, timing
    out or missing is worth a retry."""
    text = f'{what} failed: {type(e).__name__}: {e}'
    if isinstance(e, ExaidSubprocessNotOK):
        return Permanent(text)
    return RuntimeError(text)


def _entry(msg):
    """(description, entry, entry PON) of the task."""
    payload = msg['payload']
    desc = perf_desc(payload['module'])
    try:
        entry = desc.entry_from_dict(payload['entry'])
    except (TypeError, ValueError, KeyError) as e:
        raise Permanent(f'bad entry {payload.get("entry")!r}: {e}') from e
    return desc, entry, desc.entry_pon(entry)


def _ifaces(msg, desc) -> tuple[str, ...]:
    wanted = msg['payload'].get('ifaces')
    if wanted is None:
        return desc.ifaces()
    unknown = set(wanted) - set(desc.ifaces())
    if unknown:
        raise Permanent(f'payload names unknown iface(s) {sorted(unknown)}')
    return tuple(i for i in desc.ifaces() if i in wanted)


def _use_subject(msg, ctx, what: str):
    """The GPU's runner, switched to the task's subject, and its platform."""
    payload = msg['payload']
    runner = ctx.runner(RUNNER, payload['module'])
    try:
        runner.use_subject(payload['subject'], arch=msg['pool'])
        return runner, runner.platform()
    except (OSError, ExaidSubprocessNotOK) as e:
        logger.error(f"{what}: runner unavailable for task_id={msg['task_id']}: {e}")
        raise _runner_failure(what, e) from e


def make_report(msg, ctx, iface: str, backend: str | None, status: str, platform: dict,
                data: dict | None = None, error: dict | None = None) -> dict:
    """The report of one (iface, backend), as save_measurement() takes it."""
    payload = msg['payload']
    measured = {
        'subject': payload['subject'],
        'module': payload['module'],
        'entry': payload['entry'],
        'backend_forced': backend is not None,
        'harness': {'commit': platform.get('harness'), 'arches': platform.get('core_arches')},
        'platform': {
            **{k: v for k, v in platform.items() if k not in ('harness', 'core_arches')},
            **host_facts(),
            'arch': msg['pool'],
            'variant': msg.get('subpool') or ANY_VARIANT,
            'node_subpool': msg.get('node_subpool'),
        },
        'worker': ctx.worker_id,
        'gpu': ctx.gpu_id,
    }
    if data is not None:
        measured.update({
            'timing_method': data.get('timing_method'),
            'n': data.get('n'),
            'l2_flush': int(bool(data.get('l2_flush'))),
            'seed': data.get('seed'),
            'stats': {k: data.get(k) for k in ('median', 'mean', 'stddev', 'min', 'p05', 'p95')},
            'thermal': data.get('thermal'),
        })
    else:
        measured['thermal'] = {'gate': platform.get('thermal_gate')}
    return {'iface': iface,
            'backend': backend,
            'status': status,
            'payload': measured,
            'error': error,
            'gpu': ctx.gpu_id}


def _result(msg, report: dict) -> dict:
    out = child(msg, RESULT, report=report)
    return semaphore.signal(out, key=(report['iface'], report['backend']), value=report['status'])


class PerfmonStartHandler(StartHandler):
    """Starts the graph for a claimed task.

    Input: the PG reader's start message (payload, appcookie, and `enrich`,
    which carries the node's own subpool).
    Output: a probe message, carrying the pairs already done.
    """
    CLASS = START_CLASS
    QUEUE = 'gpu'
    EMITS = (PROBE,)

    def handle(self, msg, ctx):
        done = (msg.get('appcookie') or {}).get(DONE, [])
        return child(msg,
                     PROBE,
                     done=done,
                     node_subpool=(msg.get('enrich') or {}).get('node_subpool'))


class ProbeHandler(MessageHandler):
    """Lists what to measure and fans out.

    Switches the GPU's runner to the task's subject, reads its platform, and
    lists the backends of each iface (the subject's own list, by canonical
    name), plus the unforced operator. Pairs in the appcookie's `done` list
    are skipped. If the entry does not fit this GPU's VRAM on an iface, every
    pair of that iface is reported excluded and nothing runs. A backend that
    cannot run this entry is found out by measure, not here.

    Input: probe message.
    Output: the join (finalize, blocked until every pair has reported), one
    result per excluded pair, and one measure message per remaining pair.
    """
    CLASS = PROBE
    QUEUE = 'gpu'
    EMITS = (MEASURE, RESULT, FINALIZE)
    PRIORITY = 3

    def handle(self, msg, ctx):
        desc, entry, pon = _entry(msg)
        ifaces = _ifaces(msg, desc)
        runner, platform = _use_subject(msg, ctx, 'Probe')
        pairs = []
        try:
            for iface in ifaces:
                pairs.append((iface, None))
                pairs.extend((iface, backend) for backend in runner.enumerate(pon, iface))
        except (OSError, ExaidSubprocessNotOK) as e:
            logger.error(f"Probe failed for task_id={msg['task_id']}: {e}")
            raise _runner_failure('Probe', e) from e

        done = {(iface, backend) for iface, backend in msg.get('done', [])}
        pairs = [p for p in pairs if p not in done]

        vram = platform.get('vram_total_bytes')
        fit = {}
        for iface in ifaces:
            if vram:
                fit[iface] = desc.fits_vram(entry, iface, int(vram))
            else:
                logger.warning('runner reports no VRAM size; skipping the fit check')
                fit[iface] = (True, '')

        excluded, measures = [], []
        for iface, backend in pairs:
            ok, reason = fit[iface]
            if ok:
                measures.append(child(msg,
                                      MEASURE,
                                      iface=iface,
                                      backend=backend,
                                      node_subpool=msg.get('node_subpool')))
            else:
                report = make_report(msg,
                                     ctx,
                                     iface,
                                     backend,
                                     'excluded',
                                     platform,
                                     error={'reason': reason})
                excluded.append(_result(msg, report))

        join = semaphore.wait(RESULT, pairs, child(msg, FINALIZE))
        logger.info(f"Probed task_id={msg['task_id']}: {len(measures)} to measure, "
                    f"{len(excluded)} excluded, {len(done)} already done")
        # The join goes first: it is parked in the broker before any result
        # can arrive.
        return [join] + excluded + measures


class MeasureHandler(MessageHandler):
    """Measures one (iface, backend) of the entry.

    Input: measure message.
    Output: a result message, signalled with key (iface, backend).
    """
    CLASS = MEASURE
    QUEUE = 'gpu'
    EMITS = (RESULT,)
    PRIORITY = 2

    def handle(self, msg, ctx):
        desc, entry, pon = _entry(msg)
        iface, backend = msg['iface'], msg['backend']
        runner, platform = _use_subject(msg, ctx, 'Measure')
        data, error = None, None
        try:
            data = runner.measure(pon, iface, backend)
            status = 'OK'
        except BackendUnavailable as e:
            status, error = 'excluded', {'reason': str(e)}
        except OSError as e:
            logger.error(f'Measure crashed for {iface}/{backend}: {e}')
            status, error = 'crash', {'errno': e.errno, 'stderr': e.strerror}
        except ExaidSubprocessNotOK as e:
            logger.error(f'Measure NotOK for {iface}/{backend}: {e}')
            status, error = 'NotOK', {'stdout': e.stdout, 'stderr': e.stderr}
        report = make_report(msg,
                             ctx,
                             iface,
                             backend,
                             status,
                             platform,
                             data=data,
                             error=error)
        return _result(msg, report)


class ResultHandler(MessageHandler):
    """Writes one (iface, backend) as a pgdef.report row and records it done
    in the appcookie.

    Input: result message.
    Output: none. (The broker also offers the message to the join.)
    """
    CLASS = RESULT
    QUEUE = 'cpu'
    EMITS = ()
    PRIORITY = 0

    def handle(self, msg, ctx):
        report = msg['report']
        ref = task_ref(msg)
        save_measurement(ctx.db_conn, ref, report, attempt=msg.get('attempt', 1))
        pgtasks.append_cookie(ctx.db_conn, ref, DONE, [[report['iface'], report['backend']]])
        return None


class FinalizeHandler(MessageHandler):
    """Completes the task once every pair has reported.

    Input: finalize message (released by the fan-in).
    Output: dag_ack, which releases the PG reader. If the broker shuts down
    first, the default teardown gives the task back.
    """
    CLASS = FINALIZE
    QUEUE = 'cpu'
    EMITS = (DAG_ACK,)
    PRIORITY = 4

    def handle(self, msg, ctx):
        statuses = semaphore.get_aggregate_results(msg)
        if not pgtasks.complete(ctx.db_conn, task_ref(msg), owner=owner_of(msg)):
            logger.warning(f"task_id={msg['task_id']} was not completed: it is no longer running "
                           f"under this claim (reaped, reset or cancelled)")
        counts: dict[str, int] = {}
        for status in statuses.values():
            counts[status] = counts.get(status, 0) + 1
        logger.info(f"Finalized task_id={msg['task_id']}: {counts}")
        return ack(msg)


START = PerfmonStartHandler

HANDLERS = [
    PerfmonStartHandler,
    ProbeHandler,
    MeasureHandler,
    ResultHandler,
    FinalizeHandler,
]
