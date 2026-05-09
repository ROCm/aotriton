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

from v3python.tune.exaid import exaid_create, ExaidSubprocessNotOK
from ..pq.queue import TaskQueue
from ..pq.results import save_tuning_result, save_optune_result

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
    Discovers hsaco kernels and creates tune_hsaco + postprocess messages.

    Input: probe message
    Output: Multiple tune_hsaco messages + one postprocess message (with dependencies), or mark_task_failed message
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
        is_op = module.endswith('_op')

        exaid = exaid_create(module, self.gpu_id)
        tmpdir = Path(task_config['tmpdir'])

        try:
            impl_dict = exaid.probe(tmpdir)
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

        if is_op:
            return self._build_fanout_op(impl_dict, task_id, task_config)
        else:
            return self._build_fanout_kernel(impl_dict, task_id, task_config)

    def _build_fanout_kernel(self, impl_dict: dict, task_id: int,
                             task_config: dict) -> List[dict]:
        max_hsaco_dict = task_config.get('max_hsaco', {})
        max_hsaco_global = max_hsaco_dict.get('*', None)
        results = []
        impl_tasks = []

        for kname, hsaco_list in impl_dict.items():
            max_h = max_hsaco_dict.get(kname, max_hsaco_global)
            limited = hsaco_list[:max_h] if max_h else hsaco_list
            for hsaco_index in range(len(limited)):
                impl_tasks.append((kname, hsaco_index))
                results.append({
                    'class': 'tune_hsaco',
                    'target_queue': 'gpu_queue',
                    'task_id': task_id,
                    'task_config': task_config,
                    'kname': kname,
                    'hsaco_index': hsaco_index,
                })

        expected_impls = {}
        for name, index in impl_tasks:
            expected_impls.setdefault(name, []).append(index)

        results.append({
            'class': 'postprocess',
            'target_queue': 'cpu_queue',
            'task_id': task_id,
            'task_config': task_config,
            'depends': ['hsaco_result'],
            'name_key': 'kname',
            'index_key': 'hsaco_index',
            'expected_impls': expected_impls,
            'received_impls': defaultdict(dict),
        })
        logger.info(f"Probed {len(impl_tasks)} hsaco kernels for task_id={task_id}")
        return results

    def _build_fanout_op(self, impl_dict: dict, task_id: int,
                         task_config: dict) -> List[dict]:
        # impl_dict: {op_name: [{'backend_index': i}, ...], ...}
        results = []
        impl_tasks = []

        for op_name, backend_list in impl_dict.items():
            for entry in backend_list:
                backend_index = entry['backend_index']
                impl_tasks.append((op_name, backend_index))
                results.append({
                    'class': 'tune_backend',
                    'target_queue': 'gpu_queue',
                    'task_id': task_id,
                    'task_config': task_config,
                    'op_name': op_name,
                    'backend_index': backend_index,
                })

        expected_impls = {}
        for name, index in impl_tasks:
            expected_impls.setdefault(name, []).append(index)

        results.append({
            'class': 'postprocess',
            'target_queue': 'cpu_queue',
            'task_id': task_id,
            'task_config': task_config,
            'depends': ['backend_result'],
            'name_key': 'op_name',
            'index_key': 'backend_index',
            'expected_impls': expected_impls,
            'received_impls': defaultdict(dict),
        })
        logger.info(f"Probed {len(impl_tasks)} backends for task_id={task_id}")
        return results


class TuneHsacoHandler(MessageHandler):
    """
    Benchmarks single hsaco kernel.

    Input: tune_hsaco message
    Output: hsaco_result message
    """

    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id

    @classmethod
    def get_class_name(cls) -> str:
        return "tune_hsaco"

    def _impl_keys(self, message: dict) -> tuple[str, int]:
        return message['kname'], message['hsaco_index']

    def _result_class(self) -> str:
        return "hsaco_result"

    def _report_id_fields(self, impl_name: str, impl_index: int) -> dict:
        return {'kernel_name': impl_name, 'hsaco_index': impl_index}

    def _result_id_fields(self, impl_name: str, impl_index: int) -> dict:
        return {'kname': impl_name, 'hsaco_index': impl_index}

    def handle(self, message: dict) -> dict:
        task_config = message['task_config']
        task_id = message['task_id']
        impl_name, impl_index = self._impl_keys(message)

        module = task_config['module']
        exaid = exaid_create(module, self.gpu_id)
        tmpdir = Path(task_config['tmpdir'])

        report = self._report_id_fields(impl_name, impl_index)
        try:
            result_data = exaid.benchmark(tmpdir, impl_name, impl_index)
            report['result'] = 'OK'
            report['result_data'] = result_data
            report['error'] = None
        except OSError as e:
            logger.error(f"Benchmark crashed for {impl_name}[{impl_index}]: {e}")
            report['result'] = 'crash'
            report['result_data'] = None
            report['error'] = {'errno': e.errno, 'stderr': e.strerror}
        except ExaidSubprocessNotOK as e:
            logger.error(f"Benchmark NotOK for {impl_name}[{impl_index}]: {e}")
            report['result'] = 'NotOK'
            report['result_data'] = None
            report['error'] = {'stdout': e.stdout, 'stderr': e.stderr}

        return {
            'class': self._result_class(),
            'target_queue': 'cpu_queue',
            'task_id': task_id,
            **self._result_id_fields(impl_name, impl_index),
            'report': report,
        }


class TuneBackendHandler(TuneHsacoHandler):
    """
    Benchmarks single op backend.

    Input: tune_backend message
    Output: backend_result message
    """

    @classmethod
    def get_class_name(cls) -> str:
        return "tune_backend"

    def _impl_keys(self, message: dict) -> tuple[str, int]:
        return message['op_name'], message['backend_index']

    def _result_class(self) -> str:
        return "backend_result"

    def _report_id_fields(self, impl_name: str, impl_index: int) -> dict:
        return {'op_name': impl_name, 'backend_index': impl_index}

    def _result_id_fields(self, impl_name: str, impl_index: int) -> dict:
        return {'op_name': impl_name, 'backend_index': impl_index}


class WriteHsacoResultHandler(MessageHandler):
    """
    Writes hsaco result to database.

    Input: hsaco_result message
    Output: None (triggers dependency resolution for postprocess)
    """

    def __init__(self, db_conn):
        self.db_conn = db_conn

    @classmethod
    def get_class_name(cls) -> str:
        return "hsaco_result"

    def handle(self, message: dict) -> None:
        task_id = message['task_id']
        report = message['report']

        save_tuning_result(task_id, report, self.db_conn)

        logger.debug(f"Wrote hsaco result for task_id={task_id} "
                    f"{report['kernel_name']}[{report['hsaco_index']}]")
        return None


class WriteBackendResultHandler(MessageHandler):
    """
    Writes op backend result to optune_results.

    Input: backend_result message
    Output: None (triggers dependency resolution for postprocess)
    """

    def __init__(self, db_conn):
        self.db_conn = db_conn

    @classmethod
    def get_class_name(cls) -> str:
        return "backend_result"

    def handle(self, message: dict) -> None:
        task_id = message['task_id']
        report = message['report']

        save_optune_result(task_id, report, self.db_conn)

        logger.debug(f"Wrote backend result for task_id={task_id} "
                    f"{report['op_name']}[{report['backend_index']}]")
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
    - One in the broker's blocked_messages dict (tracking received_hsacos)
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

        name_key = blocked_msg['name_key']
        index_key = blocked_msg['index_key']
        impl_name = incoming_msg[name_key]
        impl_index = incoming_msg[index_key]
        blocked_msg['received_impls'][impl_name][impl_index] = incoming_msg['report']

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
