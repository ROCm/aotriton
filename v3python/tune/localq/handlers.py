# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Message handlers for local queue DAG workflow.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List
import psycopg
from psycopg.types.json import Jsonb

from v3python.tune.exaid import exaid_create, ExaidSubprocessNotOK
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
        # Just forward task_config to preprocess
        return {
            'class': 'preprocess',
            'target_queue': 'gpu_queue',
            'task_id': message['task_id'],
            'task_config': message['task_config']
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

        try:
            exaid.prepare_data(task_config["entry"], tmpdir)
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
                'error': error_msg
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

        # Execute probe
        module = task_config["module"]
        exaid = exaid_create(module, self.gpu_id)
        tmpdir = Path(task_config['tmpdir'])

        try:
            kernel_dict = exaid.probe(tmpdir)
        except (OSError, ExaidSubprocessNotOK) as e:
            logger.error(f"Probe failed for task_id={task_id}: {e}")
            # Return message to CPU worker to mark task as failed
            arch = task_config.get('arch')
            error_msg = f"Probe failed: {type(e).__name__}: {str(e)}"
            return {
                'class': 'mark_task_failed',
                'target_queue': 'cpu_queue',
                'task_id': task_id,
                'arch': arch,
                'error': error_msg
            }

        # Apply max_hsaco filtering
        max_hsaco_dict = task_config.get("max_hsaco", {})
        max_hsaco_global = max_hsaco_dict.get("*", None)

        # Generate tune_hsaco messages
        results = []
        hsaco_tasks = []  # List of (kname, hsaco_index) for tracking

        for kname, hsaco_list in kernel_dict.items():
            max_hsaco = max_hsaco_dict.get(kname, max_hsaco_global)
            limited_hsaco = hsaco_list[:max_hsaco] if max_hsaco else hsaco_list

            for hsaco_index in range(len(limited_hsaco)):
                hsaco_tasks.append((kname, hsaco_index))

                results.append({
                    'class': 'tune_hsaco',
                    'target_queue': 'gpu_queue',
                    'task_id': task_id,
                    'task_config': task_config,
                    'kname': kname,
                    'hsaco_index': hsaco_index
                })

        # Generate postprocess message (depends on all tune_hsaco)
        # Build expected_hsacos dict for tracking: {kname: [hsaco_index, ...]}
        expected_hsacos = {}
        for kname, hsaco_index in hsaco_tasks:
            if kname not in expected_hsacos:
                expected_hsacos[kname] = []
            expected_hsacos[kname].append(hsaco_index)

        postprocess_msg = {
            'class': 'postprocess',
            'target_queue': 'cpu_queue',
            'task_id': task_id,
            'task_config': task_config,
            'depends': ['hsaco_result'],  # Wait for all hsaco_result messages
            'expected_hsacos': expected_hsacos,  # Track which kernels expected
            'received_hsacos': {},  # Will accumulate: {kname: {hsaco_index: report}}
        }

        results.append(postprocess_msg)

        logger.info(f"Probed {len(hsaco_tasks)} hsaco kernels for task_id={task_id}")
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

    def handle(self, message: dict) -> dict:
        task_config = message['task_config']
        kname = message['kname']
        hsaco_index = message['hsaco_index']
        task_id = message['task_id']

        # Execute benchmarking
        module = task_config["module"]
        exaid = exaid_create(module, self.gpu_id)
        tmpdir = Path(task_config['tmpdir'])

        report = {
            "kernel_name": kname,
            "hsaco_index": hsaco_index,
        }

        try:
            result_data = exaid.benchmark(tmpdir, kname, hsaco_index)
            report['result'] = "OK"
            report['result_data'] = result_data
            report['error'] = None
        except OSError as e:
            logger.error(f"Benchmark crashed for {kname}[{hsaco_index}]: {e}")
            report['result'] = "crash"
            report['result_data'] = None
            report['error'] = {
                "errno": e.errno,
                "stderr": e.strerror
            }
        except ExaidSubprocessNotOK as e:
            logger.error(f"Benchmark NotOK for {kname}[{hsaco_index}]: {e}")
            report['result'] = "NotOK"
            report['result_data'] = None
            report['error'] = {
                "stdout": e.stdout,
                "stderr": e.stderr,
            }

        # Return hsaco_result message
        return {
            'class': 'hsaco_result',
            'target_queue': 'cpu_queue',
            'task_id': task_id,
            'kname': kname,
            'hsaco_index': hsaco_index,
            'report': report
        }


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

        # Write to tuning_results table using pq function
        save_tuning_result(task_id, report, self.db_conn)

        logger.debug(f"Wrote hsaco result for task_id={task_id} "
                    f"{report['kernel_name']}[{report['hsaco_index']}]")

        # No result message - this triggers dependency resolution in broker
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
        Called when hsaco_result arrives.
        Accumulate reports and check if all hsacos completed.

        IMPORTANT: This method is called in the BROKER context, not the CPU worker context.
        The broker instantiates PostprocessHandler with db_conn=None just to call this method.
        Do NOT access self.db_conn here - it will be None. This method only manipulates
        message dictionaries to track dependency resolution.

        The actual handle() method runs in the CPU worker context and has a valid db_conn.

        TODO: Consider splitting this class into BrokerPostprocessTracker and WorkerPostprocessHandler
        to make the dual-context usage more explicit.
        """
        if blocked_msg['class'] != 'postprocess':
            return False

        if incoming_msg['class'] != 'hsaco_result':
            return False

        if blocked_msg['task_id'] != incoming_msg['task_id']:
            return False

        # Accumulate report
        kname = incoming_msg['kname']
        hsaco_index = incoming_msg['hsaco_index']
        report = incoming_msg['report']

        if kname not in blocked_msg['received_hsacos']:
            blocked_msg['received_hsacos'][kname] = {}

        blocked_msg['received_hsacos'][kname][hsaco_index] = report

        # Check if all expected hsacos received
        expected = blocked_msg['expected_hsacos']
        received = blocked_msg['received_hsacos']

        # Check each kernel
        for kname, expected_indices in expected.items():
            if kname not in received:
                return False  # Kernel not started yet

            for hsaco_index in expected_indices:
                if hsaco_index not in received[kname]:
                    return False  # Missing hsaco index

        # All hsacos received
        logger.info(f"All hsacos received for task_id={blocked_msg['task_id']}, "
                   f"unblocking postprocess")
        return True

    def handle(self, message: dict) -> dict:
        """
        Called after all dependencies resolved.
        """
        task_id = message['task_id']
        task_config = message['task_config']
        received_hsacos = message['received_hsacos']

        # Aggregate results into brief format
        # brief = {kname: {hsaco_index: result, ...}, ...}
        brief = {}
        for kname, hsaco_dict in received_hsacos.items():
            brief[kname] = {}
            for hsaco_index, report in hsaco_dict.items():
                brief[kname][hsaco_index] = report['result']

        aggregation = {
            "brief": brief,
        }

        # Update task_queue with completed status
        # Extract arch from task_config for partition routing
        arch = task_config.get('arch')
        logger.info(f"PostprocessHandler: Marking task_id={task_id} as completed (arch={arch})")
        task_queue = TaskQueue(self.db_conn)
        task_queue.mark_completed(task_id, arch)

        logger.info(f"Postprocess completed for task_id={task_id}")

        # Cleanup tmpdir
        tmpdir = Path(task_config['tmpdir'])
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
            logger.debug(f"Cleaned up tmpdir: {tmpdir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup tmpdir {tmpdir}: {e}")

        # Return ack message to unblock PG reader
        ack_msg = {
            'class': 'tune_kernel_ack',
            'task_id': task_id
        }
        logger.info(f"Postprocess returning ack message for task_id={task_id}")
        return ack_msg

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

        # Return nak (negative ack) message to unblock PG reader
        return {
            'class': 'tune_kernel_ack',
            'task_id': task_id,
            'negative': True
        }
