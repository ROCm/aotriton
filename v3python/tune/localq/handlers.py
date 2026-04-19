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
    Output: probe message
    """

    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id

    @classmethod
    def get_class_name(cls) -> str:
        return "preprocess"

    def handle(self, message: dict) -> dict:
        task_config = message['task_config']

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
            logger.error(f"Preprocess failed: {e}")
            raise

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
    Output: Multiple tune_hsaco messages + one postprocess message (with dependencies)
    """

    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id

    @classmethod
    def get_class_name(cls) -> str:
        return "probe"

    def handle(self, message: dict) -> List[dict]:
        task_config = message['task_config']
        task_id = message['task_id']

        # Execute probe
        module = task_config["module"]
        exaid = exaid_create(module, self.gpu_id)
        tmpdir = Path(task_config['tmpdir'])

        try:
            kernel_dict = exaid.probe(tmpdir)
        except (OSError, ExaidSubprocessNotOK) as e:
            logger.error(f"Probe failed: {e}")
            raise

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

    def __init__(self, conn_params: dict):
        self.conn_params = conn_params

    @classmethod
    def get_class_name(cls) -> str:
        return "hsaco_result"

    def handle(self, message: dict) -> None:
        task_id = message['task_id']
        report = message['report']

        # Write to tuning_results table
        with psycopg.connect(**self.conn_params) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO tuning_results
                    (task_id, kernel_name, hsaco_index, result, result_data, error, gpu_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    task_id,
                    report['kernel_name'],
                    report['hsaco_index'],
                    report['result'],
                    Jsonb(report.get('result_data')),
                    Jsonb(report.get('error')),
                    message.get('gpu_id')
                ))
                conn.commit()

        logger.debug(f"Wrote hsaco result for task_id={task_id} "
                    f"{report['kernel_name']}[{report['hsaco_index']}]")

        # No result message - this triggers dependency resolution in broker
        return None


class PostprocessHandler(MessageHandler):
    """
    Aggregates all hsaco results and cleans up.

    Input: postprocess message (after dependencies resolved)
    Output: tune_kernel_ack message (triggers PG reader to continue)
    """

    def __init__(self, conn_params: dict):
        self.conn_params = conn_params

    @classmethod
    def get_class_name(cls) -> str:
        return "postprocess"

    def resolve_dependency(self, blocked_msg: dict, incoming_msg: dict) -> bool:
        """
        Called when hsaco_result arrives.
        Accumulate reports and check if all hsacos completed.
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
        with psycopg.connect(**self.conn_params) as conn:
            with conn.cursor() as cur:
                # Note: We don't write tmpdir to database per feedback
                cur.execute("""
                    UPDATE task_queue
                    SET status = 'completed',
                        completed_at = NOW()
                    WHERE id = %s
                """, (task_id,))
                conn.commit()

        logger.info(f"Postprocess completed for task_id={task_id}")

        # Cleanup tmpdir
        tmpdir = Path(task_config['tmpdir'])
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
            logger.debug(f"Cleaned up tmpdir: {tmpdir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup tmpdir {tmpdir}: {e}")

        # Return ack message to unblock PG reader
        return {
            'class': 'tune_kernel_ack',
            'task_id': task_id
        }
