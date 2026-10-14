# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The tune application's handler graph.

A task is one tuning entry (payload {module, entry, max_hsaco?}); its category
is ('tune', 'kernel') or ('tune', 'op'). The graph is implicit -- each
handler's return value is the next message -- and each handler declares where
it runs and what it may emit; `python -m aotriton.pgdef.localq.graph tune`
prints it from those declarations.

    tune/start (gpu) ---> tune/preprocess (gpu) ---> tune/probe (gpu) --+--> tune/impl x N (gpu) --> tune/impl_result (cpu)
                                                                         |
                                                                         +--> tune/postprocess (cpu) --> dag_ack
                                                                              (fan-in over tune/impl_result)

Any exception from any handler is turned into the engine's `task_failed` by
GenericWorker; a handler raises `Permanent` for a failure that retrying cannot
fix. Nothing here builds a failure message.

Runner access: GPU handlers use `ctx.runner('tuner', module)` (an
ExaidTunerWorker). CPU handlers use `ctx.db_conn`.
"""

import logging
import shutil
from pathlib import Path

from aotriton.pgdef.localq import semaphore
from aotriton.pgdef.localq.handlers import (DAG_ACK, MessageHandler, Permanent, StartHandler, ack,
                                            child, task_ref)
from aotriton.pgdef.exaid import ExaidSubprocessNotOK
from aotriton.pgdef.pq import tasks as pgtasks

from .pq.results import save_tuning_result
from .tdesc import ImplSelector

logger = logging.getLogger(__name__)

START_CLASS = 'tune/start'
PREPROCESS = 'tune/preprocess'
PROBE = 'tune/probe'
IMPL = 'tune/impl'
IMPL_RESULT = 'tune/impl_result'
POSTPROCESS = 'tune/postprocess'


def level_of(msg: dict) -> str:
    """The tune level ('kernel' | 'op'): the task category's second level.

    No default: a missing level means the task did not come through the PG
    reader, and guessing 'kernel' would silently run an op task at the wrong
    level.
    """
    return msg['category'][1]


def _rmtree(path) -> None:
    if path:
        shutil.rmtree(path, ignore_errors=True)


def _runner_failure(what: str, e: Exception) -> Exception:
    """Classify a runner error: the runner rejecting the request (an `Error`
    reply, e.g. an entry it cannot parse) is permanent; the runner dying or
    timing out (OSError) is worth a retry."""
    text = f'{what} failed: {type(e).__name__}: {e}'
    if isinstance(e, ExaidSubprocessNotOK):
        return Permanent(text)
    return RuntimeError(text)


class TuneStartHandler(StartHandler):
    """Starts the graph for a claimed task.

    Input: the PG reader's start message (task columns, payload, appcookie, and
    `enrich`, which carries the task's extra unit tests).
    Output: a preprocess message.
    """
    CLASS = START_CLASS
    QUEUE = 'gpu'
    EMITS = (PREPROCESS,)

    def handle(self, msg, ctx):
        return child(msg, PREPROCESS,
                     extra_im_texts=(msg.get('enrich') or {}).get('extra_im_texts', []))

    @classmethod
    def cleanup(cls, msg, ctx):
        """The task failed or was given back: free its prepared data."""
        _rmtree(msg.get('tmpdir'))


class PreprocessHandler(MessageHandler):
    """Prepares the test data for the entry in the node's tmpfs.

    Input: preprocess message.
    Output: probe message.
    """
    CLASS = PREPROCESS
    QUEUE = 'gpu'
    EMITS = (PROBE,)
    PRIORITY = 1

    def handle(self, msg, ctx):
        payload = msg['payload']
        runner = ctx.runner('tuner', payload['module'])
        if 'tmpdir' in payload:
            tmpdir = Path(payload['tmpdir'])      # explicit override, e.g. for debugging
        else:
            tmpdir = runner.get_tmpfs_for(payload['entry'])
        try:
            runner.prepare_data(payload['entry'], tmpdir,
                                extra_im_texts=msg.get('extra_im_texts', []))
        except (OSError, ExaidSubprocessNotOK) as e:
            logger.error(f"Preprocess failed for task_id={msg['task_id']}: {e}")
            _rmtree(tmpdir)
            raise _runner_failure('Preprocess', e) from e
        return child(msg, PROBE, tmpdir=tmpdir.as_posix())


class ProbeHandler(MessageHandler):
    """Discovers impl variants (HSACO indices for kernel level, backend indices
    for op level) and fans out one impl message per variant, plus the join.

    Input: probe message.
    Output: N impl messages and one postprocess message, which is blocked in the
    broker until every impl has reported (fan-in over tune/impl_result).

    The flash/op-level provider modules' `enumerate_variants()` return
    `list[dict]` for both levels (one dict per candidate variant, its position
    in the list being the impl_index), so `probe()`'s {iface_name: [dict, ...]}
    shape is uniform across levels -- no separate kernel/op fan-out branch.

    The level is passed to the runner as a per-call filter, so a container only
    probes what its library can serve; filtering happens at the call, not by
    discarding results afterwards (which would mean the container had already
    attempted, and failed, to import the wrong-library provider module).
    """
    CLASS = PROBE
    QUEUE = 'gpu'
    EMITS = (IMPL, POSTPROCESS)
    PRIORITY = 3

    def handle(self, msg, ctx):
        payload = msg['payload']
        level = level_of(msg)
        runner = ctx.runner('tuner', payload['module'])
        try:
            impl_dict = runner.probe(Path(msg['tmpdir']), msg['pool'], tuning_level=level)
        except (OSError, ExaidSubprocessNotOK) as e:
            logger.error(f"Probe failed for task_id={msg['task_id']}: {e}")
            raise _runner_failure('Probe', e) from e
        return self._build_fanout(impl_dict, msg, payload, level)

    def _build_fanout(self, impl_dict: dict, msg: dict, payload: dict, level: str) -> list[dict]:
        # impl_dict: {dsl_name: [variant_dict, ...], ...} -- keys are
        # DSL-spelled (e.g. 'attn_fwd' or 'op.attn_fwd', as returned by
        # probe()); variant_dict's contents are level-specific (psels/copts for
        # kernel, backend_name for op) but unused here, only its position
        # (impl_index) matters. Storage keeps bare iface_name + the tuning level, so
        # the DSL prefix (surface syntax only) is stripped back off here via
        # ImplSelector.split_dsl_name().
        max_hsaco_dict = payload.get('max_hsaco', {})
        max_hsaco_global = max_hsaco_dict.get('*', None)
        impl_msgs = []
        keys = []

        for dsl_name, variants in impl_dict.items():
            _, iface_name = ImplSelector.split_dsl_name(dsl_name)
            if isinstance(variants, dict) and 'error' in variants:
                # Defensive only: with the per-call level filter, every name
                # returned should already belong to this task's own level and
                # resolve cleanly. Surfacing rather than crashing keeps a
                # misconfigured/legacy filter from taking down the whole task.
                logger.error(f"Probe reported an unresolvable impl {dsl_name!r} for "
                             f"task_id={msg['task_id']}: {variants['error']}")
                continue
            # A single candidate is only skippable at op level, where one
            # backend means there is genuinely nothing to choose between.
            # Kernel level must still benchmark a lone HSACO: its row is what
            # feeds most_accurate_tuning_results and best_tuning_results, so
            # skipping it would drop that interface out of the accuracy and
            # best-results pipeline entirely.
            if level == 'op' and len(variants) <= 1:
                logger.info(f"Skipping iface_name={iface_name} for task_id={msg['task_id']}: "
                            f"only {len(variants)} backend(s), no tuning needed")
                continue
            # max_hsaco caps the HSACO sweep and is kernel-level only. Applying
            # it to op backends would drop real backends from the fanout -- with
            # --max_hsaco 1 every op interface truncates to one variant and is
            # then dropped entirely by the skip above, so op tuning would
            # silently do nothing.
            max_h = max_hsaco_dict.get(iface_name, max_hsaco_global) if level == 'kernel' else None
            limited = variants[:max_h] if max_h else variants
            for impl_index in range(len(limited)):
                keys.append((iface_name, impl_index))
                impl_msgs.append(child(msg, IMPL, tmpdir=msg['tmpdir'],
                                       iface_name=iface_name, impl_index=impl_index))

        join = semaphore.wait(IMPL_RESULT, keys, child(msg, POSTPROCESS, tmpdir=msg['tmpdir']))
        logger.info(f"Probed {len(keys)} impl variant(s) for task_id={msg['task_id']} (tuning_level={level})")
        # The join goes FIRST: it is blocked in the broker before any impl can
        # possibly have reported, so no impl_result can arrive ahead of it.
        return [join] + impl_msgs


class TuneImplHandler(MessageHandler):
    """Benchmarks a single impl variant (HSACO index for kernel level, backend
    index for op level). ImplSelector's iface_name/impl_index plus the level
    address either.

    A benchmark that crashes or is rejected is a RESULT (`crash`, `NotOK`), not
    a failure of the task: it is reported and the sweep continues.

    Input: impl message.
    Output: impl_result message.
    """
    CLASS = IMPL
    QUEUE = 'gpu'
    EMITS = (IMPL_RESULT,)
    PRIORITY = 2

    def handle(self, msg, ctx):
        payload = msg['payload']
        iface_name, impl_index = msg['iface_name'], msg['impl_index']
        level = level_of(msg)
        runner = ctx.runner('tuner', payload['module'])

        impl_selector = ImplSelector(tuning_level=level, iface_name=iface_name, impl_index=impl_index)
        report = {'tuning_level': level, 'iface_name': iface_name, 'impl_index': impl_index,
                  'complete_on_gpu': ctx.gpu_id}
        try:
            report['result_data'] = runner.benchmark(Path(msg['tmpdir']), impl_selector)
            report['result'] = 'OK'
            report['error'] = None
        except OSError as e:
            logger.error(f"Benchmark crashed for {iface_name}[{impl_index}]: {e}")
            report['result'] = 'crash'
            report['result_data'] = None
            report['error'] = {'errno': e.errno, 'stderr': e.strerror}
        except ExaidSubprocessNotOK as e:
            logger.error(f"Benchmark NotOK for {iface_name}[{impl_index}]: {e}")
            report['result'] = 'NotOK'
            report['result_data'] = None
            report['error'] = {'stdout': e.stdout, 'stderr': e.stderr}

        result = child(msg, IMPL_RESULT, iface_name=iface_name, impl_index=impl_index, report=report)
        return semaphore.signal(result, key=(iface_name, impl_index), value=None)


class WriteImplResultHandler(MessageHandler):
    """Writes an impl benchmark result as a pgdef.report row.

    Input: impl_result message.
    Output: none. (The broker also offers the message to the join, the fan-in,
    that is waiting for it.)
    """
    CLASS = IMPL_RESULT
    QUEUE = 'cpu'
    EMITS = ()
    PRIORITY = 0

    def handle(self, msg, ctx):
        report = msg['report']
        save_tuning_result(ctx.db_conn, task_ref(msg), report, attempt=msg.get('attempt', 1))
        logger.debug(f"Wrote impl result for task_id={msg['task_id']} "
                     f"{report['iface_name']}[{report['impl_index']}] (tuning_level={report['tuning_level']})")
        return None


class PostprocessHandler(MessageHandler):
    """Completes the task and frees its prepared data, once every impl has
    reported.

    Input: postprocess message (released by the fan-in).
    Output: dag_ack, which releases the PG reader.
    """
    CLASS = POSTPROCESS
    QUEUE = 'cpu'
    EMITS = (DAG_ACK,)
    PRIORITY = 4

    def handle(self, msg, ctx):
        task_id = msg['task_id']
        pgtasks.complete(ctx.db_conn, task_ref(msg))
        logger.info(f"Postprocess completed for task_id={task_id}")
        _rmtree(msg.get('tmpdir'))
        return ack(msg)

    @classmethod
    def teardown_with_unmet_dependency(cls, msg):
        """Graceful shutdown with impls still outstanding: free the prepared
        data, then give the task back (the default)."""
        _rmtree(msg.get('tmpdir'))
        return super().teardown_with_unmet_dependency(msg)


START = TuneStartHandler

HANDLERS = [
    TuneStartHandler,
    PreprocessHandler,
    ProbeHandler,
    TuneImplHandler,
    WriteImplResultHandler,
    PostprocessHandler,
]
