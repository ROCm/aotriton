# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Message handlers for local queue DAG workflow.
"""

import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List
import psycopg
from psycopg.types.json import Jsonb

from ..exaid import exaid_create, ExaidSubprocessNotOK
from ..tdesc import ImplSelector
from ..pq.queue import TaskQueue
from ..pq.results import save_tuning_result

logger = logging.getLogger(__name__)


class MessageHandler:
    """Base class for message handlers"""

    @classmethod
    def get_class_name(cls) -> str:
        """Get message class this handler processes"""
        raise NotImplementedError

    def handle(self, message: dict) -> dict | List[dict] | None:
        """
        Process message and return result message(s) (or None).

        Result message is automatically forwarded to its target_queue.

        Args:
            message: Input message

        Returns:
            Result message, list of result messages, or None
        """
        raise NotImplementedError

    def resolve_dependency(self, blocked_msg: dict, incoming_msg: dict) -> bool:
        """
        Called when incoming_msg arrives that might resolve blocked_msg's dependency.

        Args:
            blocked_msg: Message waiting for dependencies
            incoming_msg: Newly arrived message

        Returns:
            True if dependency is resolved (unblock message)
        """
        return False

    def teardown_with_unmet_dependency(self, message: dict) -> dict | None:
        """
        Called during graceful shutdown when message has unmet dependencies.

        Default implementation returns None (no action needed).
        Override in subclasses if teardown requires specific actions.

        Args:
            message: Blocked message being torn down

        Returns:
            Result message to enqueue (or None)
        """
        return None


class TuneKernelHandler(MessageHandler):
    """
    Starts the DAG by creating initial preprocess message.

    Input: tune_kernel message from PG reader
    Output: preprocess message
    """

    @classmethod
    def get_class_name(cls) -> str:
        return "tune_kernel"

    def handle(self, message: dict) -> dict:
        return {
            'class': 'preprocess',
            'target_queue': 'gpu_queue',
            'task_id': message['task_id'],
            'task_config': message['task_config'],
        }


class PreprocessHandler(MessageHandler):
    """
    Prepares test data.

    Input: preprocess message
    Output: probe message or mark_task_failed message
    """

    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id

    @classmethod
    def get_class_name(cls) -> str:
        return "preprocess"

    def handle(self, message: dict) -> dict | None:
        task_config = message['task_config']
        task_id = message['task_id']

        # Execute preprocessing
        module = task_config["module"]
        exaid = exaid_create(module, self.gpu_id)

        if 'tmpdir' in task_config:
            tmpdir = Path(task_config['tmpdir'])
        else:
            tmpdir = exaid.get_tmpfs_for(task_config["entry"])

        extra_im_texts = task_config.get('extra_im_texts', [])

        try:
            exaid.prepare_data(task_config["entry"], tmpdir, extra_im_texts=extra_im_texts)
            task_config['tmpdir'] = tmpdir.as_posix()
        except (OSError, ExaidSubprocessNotOK) as e:
            logger.error(f"Preprocess failed for task_id={task_id}: {e}")
            # Return message to CPU worker to mark task as failed
            arch = task_config.get('arch')
            error_msg = f"Preprocess failed: {type(e).__name__}: {str(e)}"
            return {
                'class': 'mark_task_failed',
                'target_queue': 'cpu_queue',
                'task_id': task_id,
                'arch': arch,
                'error': error_msg,
                'tmpdir': tmpdir.as_posix(),
            }

        # Return probe message
        return {
            'class': 'probe',
            'target_queue': 'gpu_queue',
            'task_id': message['task_id'],
            'task_config': task_config
        }


class ProbeHandler(MessageHandler):
    """
    Discovers impl variants (HSACO indices for kernel level, backend indices
    for op level) and creates tune_impl + postprocess messages.

    Input: probe message
    Output: Multiple tune_impl messages + one postprocess message (with dependencies), or mark_task_failed message

    Unified (modular-tune.md §4.1-§4.3): the flash/op-level provider modules'
    `enumerate_variants()` return `list[dict]` for both levels (one dict per
    candidate variant, its position in the list being the impl_index), so
    `exaid.probe()`'s {iface_name: [dict, ...]} shape is already uniform
    across levels -- no separate kernel/op fan-out branch is needed.

    Revision note 3: this handler passes `task_config['tuning_level']` into
    `exaid.probe()` as a per-call filter, so a container only probes what its
    library can serve -- `testrun`/`exaid` hold no level state of their own.
    Filtering happens at the call, not by post-hoc filtering of `probe()`'s
    return value here: filtering after the fact would mean the container had
    already attempted (and failed) to import the wrong-library provider
    module before this handler got a chance to discard the result.
    """

    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id

    @classmethod
    def get_class_name(cls) -> str:
        return "probe"

    def handle(self, message: dict) -> List[dict] | dict | None:
        task_config = message['task_config']
        task_id = message['task_id']
        module = task_config['module']
        level = task_config.get('tuning_level', 'kernel')

        exaid = exaid_create(module, self.gpu_id)
        tmpdir = Path(task_config['tmpdir'])
        arch = task_config.get('arch')

        try:
            impl_dict = exaid.probe(tmpdir, arch, tuning_level=level)
        except (OSError, ExaidSubprocessNotOK) as e:
            logger.error(f"Probe failed for task_id={task_id}: {e}")
            return {
                'class': 'mark_task_failed',
                'target_queue': 'cpu_queue',
                'task_id': task_id,
                'arch': task_config.get('arch'),
                'error': f"Probe failed: {type(e).__name__}: {e}",
                'tmpdir': tmpdir.as_posix(),
            }

        return self._build_fanout(impl_dict, task_id, task_config, level)

    def _build_fanout(self, impl_dict: dict, task_id: int,
                      task_config: dict, level: str) -> List[dict]:
        # impl_dict: {dsl_name: [variant_dict, ...], ...} -- keys are
        # DSL-spelled (e.g. 'attn_fwd' or 'op.attn_fwd', as returned by
        # exaid.probe()); variant_dict's contents are level-specific
        # (psels/copts for kernel, backend_index for op) but unused here,
        # only its position (impl_index) matters. Storage stays bare
        # iface_name + tuning_level (no schema change), so the DSL prefix
        # (surface syntax only) is stripped back off here via
        # ImplSelector.split_dsl_name() before it reaches tune_impl messages.
        max_hsaco_dict = task_config.get('max_hsaco', {})
        max_hsaco_global = max_hsaco_dict.get('*', None)
        results = []
        impl_tasks = []

        for dsl_name, variants in impl_dict.items():
            _, iface_name = ImplSelector.split_dsl_name(dsl_name)
            if isinstance(variants, dict) and 'error' in variants:
                # Defensive only: with the per-call tuning_level filter this
                # handler passes into exaid.probe(), every name returned
                # should already belong to this task's own level and resolve
                # cleanly. Surfacing rather than crashing keeps a
                # misconfigured/legacy filter from taking down the whole task.
                logger.error(f"Probe reported an unresolvable impl {dsl_name!r} for "
                            f"task_id={task_id}: {variants['error']}")
                continue
            if len(variants) <= 1:
                logger.info(f"Skipping iface_name={iface_name} for task_id={task_id}: "
                           f"only {len(variants)} variant(s), no tuning needed")
                continue
            max_h = max_hsaco_dict.get(iface_name, max_hsaco_global)
            limited = variants[:max_h] if max_h else variants
            for impl_index in range(len(limited)):
                impl_tasks.append((iface_name, impl_index))
                results.append({
                    'class': 'tune_impl',
                    'target_queue': 'gpu_queue',
                    'task_id': task_id,
                    'task_config': task_config,
                    'iface_name': iface_name,
                    'impl_index': impl_index,
                })

        expected_impls = {}
        for name, index in impl_tasks:
            expected_impls.setdefault(name, []).append(index)

        results.append({
            'class': 'postprocess',
            'target_queue': 'cpu_queue',
            'task_id': task_id,
            'task_config': task_config,
            'depends': ['impl_result'],
            'expected_impls': expected_impls,
            'received_impls': defaultdict(dict),
        })
        logger.info(f"Probed {len(impl_tasks)} impl variant(s) for task_id={task_id} (tuning_level={level})")
        return results


class TuneImplHandler(MessageHandler):
    """
    Benchmarks a single impl variant (HSACO index for kernel level, backend
    index for op level).

    Unified (modular-tune.md §4.1-§4.3): replaces the former
    TuneHsacoHandler/TuneBackendHandler pair -- ImplSelector's iface_name/
    impl_index (plus tuning_level, read from task_config) are enough to
    address either level.

    Input: tune_impl message
    Output: impl_result message
    """

    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id

    @classmethod
    def get_class_name(cls) -> str:
        return "tune_impl"

    def handle(self, message: dict) -> dict:
        task_config = message['task_config']
        task_id = message['task_id']
        iface_name = message['iface_name']
        impl_index = message['impl_index']

        module = task_config['module']
        level = task_config.get('tuning_level', 'kernel')
        exaid = exaid_create(module, self.gpu_id)
        tmpdir = Path(task_config['tmpdir'])

        impl_selector = ImplSelector(tuning_level=level, iface_name=iface_name, impl_index=impl_index)
        report = {'tuning_level': level, 'iface_name': iface_name, 'impl_index': impl_index}
        try:
            result_data = exaid.benchmark(tmpdir, impl_selector)
            report['result'] = 'OK'
            report['result_data'] = result_data
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

        return {
            'class': 'impl_result',
            'target_queue': 'cpu_queue',
            'task_id': task_id,
            'iface_name': iface_name,
            'impl_index': impl_index,
            'report': report,
        }


class WriteImplResultHandler(MessageHandler):
    """
    Writes an impl benchmark result to the unified tuning_results table.

    Unified (modular-tune.md §4.1-§4.3): replaces the former
    WriteHsacoResultHandler/WriteBackendResultHandler pair, both of which
    wrote to two separate tables (tuning_results/optune_results); now a
    single save_tuning_result() call, distinguished by report['tuning_level'].

    Input: impl_result message
    Output: None (triggers dependency resolution for postprocess)
    """

    def __init__(self, db_conn):
        self.db_conn = db_conn

    @classmethod
    def get_class_name(cls) -> str:
        return "impl_result"

    def handle(self, message: dict) -> None:
        task_id = message['task_id']
        report = message['report']

        save_tuning_result(task_id, report, self.db_conn)

        logger.debug(f"Wrote impl result for task_id={task_id} "
                    f"{report['iface_name']}[{report['impl_index']}] (tuning_level={report['tuning_level']})")
        return None


class PostprocessHandler(MessageHandler):
    """
    Aggregates all hsaco results and cleans up.

    Input: postprocess message (after dependencies resolved)
    Output: tune_kernel_ack message (triggers PG reader to continue)

    DESIGN NOTE: This class has dual-context usage:
    1. Broker context: Instantiated with db_conn=None, only resolve_dependency() is called
    2. CPU worker context: Instantiated with valid db_conn, handle() is called

    The broker tracks postprocess message dependencies using resolve_dependency(),
    while the CPU worker executes the actual postprocessing using handle().

    This means there are two "copies" of the postprocess message state:
    - One in the broker's blocked_messages dict (tracking received_impls)
    - One in the CPU worker's handler (executing final aggregation)

    TODO: Consider splitting into BrokerPostprocessTracker + WorkerPostprocessHandler
    """

    def __init__(self, db_conn):
        self.db_conn = db_conn

    @classmethod
    def get_class_name(cls) -> str:
        return "postprocess"

    def resolve_dependency(self, blocked_msg: dict, incoming_msg: dict) -> bool:
        """
        Called when an impl result arrives (hsaco_result or backend_result).
        Accumulates reports and checks if all expected impls completed.

        IMPORTANT: This method is called in the BROKER context, not the CPU worker context.
        Do NOT access self.db_conn here — it will be None.
        """
        if blocked_msg['class'] != 'postprocess':
            return False

        if incoming_msg['class'] not in blocked_msg['depends']:
            return False

        if blocked_msg['task_id'] != incoming_msg['task_id']:
            return False

        impl_name = incoming_msg['iface_name']
        impl_index = incoming_msg['impl_index']
        blocked_msg['received_impls'].setdefault(impl_name, {})[impl_index] = incoming_msg['report']

        expected = blocked_msg['expected_impls']
        received = blocked_msg['received_impls']
        for name, indices in expected.items():
            if name not in received:
                return False
            for idx in indices:
                if idx not in received[name]:
                    return False

        logger.info(f"All impls received for task_id={blocked_msg['task_id']}, "
                   f"unblocking postprocess")
        return True

    def handle(self, message: dict) -> dict:
        task_id = message['task_id']
        task_config = message['task_config']

        arch = task_config.get('arch')
        logger.info(f"PostprocessHandler: Marking task_id={task_id} as completed (arch={arch})")
        TaskQueue(self.db_conn).mark_completed(task_id, arch)

        logger.info(f"Postprocess completed for task_id={task_id}")

        tmpdir = Path(task_config['tmpdir'])
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
            logger.debug(f"Cleaned up tmpdir: {tmpdir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup tmpdir {tmpdir}: {e}")

        logger.info(f"Postprocess returning ack message for task_id={task_id}")
        return {
            'class': 'tune_kernel_ack',
            'task_id': task_id,
        }

    def teardown_with_unmet_dependency(self, message: dict) -> dict:
        """
        Called during graceful shutdown when postprocess message has unmet dependencies.

        This happens when GPU workers are stopped before completing all tune_hsaco tasks.
        We need to cancel the running task by moving it back to pending state.

        Args:
            message: Postprocess message with unmet dependencies

        Returns:
            GracefulCancelRunningTask message to move task back to pending
        """
        task_id = message['task_id']
        task_config = message.get('task_config', {})
        arch = task_config.get('arch')

        logger.info(f"PostprocessHandler teardown: task_id={task_id} has unmet dependencies, "
                   f"creating cancel message")

        # Cleanup tmpdir if it exists
        if 'tmpdir' in task_config:
            tmpdir = Path(task_config['tmpdir'])
            try:
                shutil.rmtree(tmpdir, ignore_errors=True)
                logger.debug(f"Cleaned up tmpdir during teardown: {tmpdir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup tmpdir {tmpdir} during teardown: {e}")

        # Return message to cancel the running task (move it back to pending)
        return {
            'class': 'graceful_cancel_running_task',
            'target_queue': 'cpu_queue',
            'task_id': task_id,
            'arch': arch
        }


class GracefulCancelRunningTaskHandler(MessageHandler):
    """
    Moves task state back to pending when gracefully cancelled.

    This handler is used during graceful shutdown to cancel running tasks
    that have unmet dependencies (incomplete tune_hsaco work).
    """

    def __init__(self, db_conn):
        self.db_conn = db_conn

    @classmethod
    def get_class_name(cls) -> str:
        return "graceful_cancel_running_task"

    def handle(self, message: dict) -> None:
        task_id = message['task_id']
        arch = message['arch']

        logger.info(f"Gracefully cancelling task_id={task_id}, moving back to pending")

        # Move task back to pending state
        task_queue = TaskQueue(self.db_conn)
        task_queue.mark_pending(task_id, arch)

        logger.info(f"Task {task_id} moved back to pending state")

        # No result message
        return None


class MarkTaskFailedHandler(MessageHandler):
    """
    Marks task as failed in database.

    This handler is used when GPU workers encounter exceptions during
    preprocess or probe stages. GPU workers don't have DB access, so they
    send this message to CPU workers to write the failure to the database.
    """

    def __init__(self, db_conn):
        self.db_conn = db_conn

    @classmethod
    def get_class_name(cls) -> str:
        return "mark_task_failed"

    def handle(self, message: dict) -> dict:
        task_id = message['task_id']
        arch = message['arch']
        error = message['error']

        logger.info(f"Marking task_id={task_id} as failed: {error}")

        # Mark task as failed in database
        task_queue = TaskQueue(self.db_conn)
        task_queue.mark_failed(task_id, arch=arch, error_message=error)

        logger.info(f"Task {task_id} marked as failed in database")

        # Remove prepared data from tmpfs to free space
        tmpdir = message.get('tmpdir')
        if tmpdir:
            tmpdir_path = Path(tmpdir)
            if tmpdir_path.exists():
                try:
                    shutil.rmtree(tmpdir_path)
                    logger.info(f"Removed tmpdir {tmpdir_path} for failed task {task_id}")
                except OSError as e:
                    logger.warning(f"Failed to remove tmpdir {tmpdir_path}: {e}")

        # Return nak (negative ack) message to unblock PG reader
        return {
            'class': 'tune_kernel_ack',
            'task_id': task_id,
            'negative': True
        }
