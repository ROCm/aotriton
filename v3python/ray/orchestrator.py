# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
DAG Orchestrator for Ray-based task execution

Coordinates the full tuning workflow:
1. Preprocess (GPU 0)
2. Probe (GPU 0)
3. Distribute tune_hsaco tasks across all GPUs
4. Write results to PostgreSQL via CPU tasks
5. Postprocess and cleanup

Ensures GPU exclusivity via Ray actors while maximizing parallelism.
"""

import ray
import os
import logging
from typing import Dict, Any

from .worker_pool import get_gpu_worker_pool
from .cpu_tasks import db_writer_task, postprocess_task

logger = logging.getLogger(__name__)


def init_ray(num_gpus: int = None, address: str = 'local') -> None:
    """
    Initialize Ray runtime.

    Args:
        num_gpus: Number of GPUs (default: from NUM_GPUS env var or 4)
        address: Ray cluster address ('local' for single-node)
    """
    if ray.is_initialized():
        print('[Ray] Already initialized')
        return

    if num_gpus is None:
        num_gpus = int(os.environ.get('NUM_GPUS', 4))

    print(f'[Ray] Initializing with num_gpus={num_gpus}, address={address}')

    ray.init(
        address=address,
        num_gpus=num_gpus,
        ignore_reinit_error=True,
        logging_level='warning'  # Reduce Ray verbosity
    )

    print(f'[Ray] Initialized successfully')


def execute_tuning_dag(task_id: str, task_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute full tuning DAG with proper GPU exclusivity.

    Workflow:
    - GPU tasks (preprocess, probe, tune_hsaco) run serially per GPU
    - Different GPUs run in parallel
    - CPU tasks (db_writer, postprocess) run without consuming GPUs
    - Ray handles scheduling and dependencies

    Args:
        task_id: Task ID for tracking and database association
        task_config: Task configuration dictionary with keys:
            - module: Tuning module name
            - entry: Entry configuration
            - arch: GPU architecture
            - max_hsaco: Optional hsaco limits per kernel

    Returns:
        Aggregation dictionary with:
            - task_config: Original config
            - brief: Summary of all kernel results

    Raises:
        Exception: Any task execution failures
    """
    # Ensure Ray is initialized
    if not ray.is_initialized():
        init_ray()

    module = task_config["module"]
    num_gpus = int(os.environ.get('NUM_GPUS', 4))

    logger.info(f'Starting DAG execution for task_id={task_id}, module={module}')

    # Get GPU worker pool for this module
    gpu_workers = get_gpu_worker_pool(module, num_gpus)

    # Use GPU 0 as coordinator for preprocess/probe
    # (they need consistent GPU for tmpdir setup)
    coordinator_worker = gpu_workers[0]

    # ========================================================================
    # Step 1: Preprocess on GPU 0 (exclusive)
    # ========================================================================
    logger.debug(f'Step 1: Preprocess on GPU 0')
    task_config_ref = coordinator_worker.preprocess.remote(task_config)

    # ========================================================================
    # Step 2: Probe on GPU 0 (exclusive, depends on preprocess)
    # ========================================================================
    logger.debug(f'Step 2: Probe on GPU 0')
    hsaco_tasks_ref = coordinator_worker.probe.remote(task_config_ref)

    # Wait for probe to complete before distributing GPU tasks
    task_config_updated = ray.get(task_config_ref)
    hsaco_tasks = ray.get(hsaco_tasks_ref)

    logger.info(f'Probed {len(hsaco_tasks)} hsaco kernels')

    if not hsaco_tasks:
        logger.warning('No hsaco kernels to tune')
        # Still run postprocess to cleanup
        aggregation_ref = postprocess_task.remote(task_id, [], task_config_updated)
        return ray.get(aggregation_ref)

    # ========================================================================
    # Step 3: Distribute tune_hsaco tasks across ALL GPUs (round-robin)
    # Each GPU task is immediately chained with a db_writer task
    # ========================================================================
    logger.debug(f'Step 3: Distributing {len(hsaco_tasks)} GPU tasks across {num_gpus} GPUs')
    gpu_futures = []
    db_futures = []

    for idx, (kname, hsaco_index) in enumerate(hsaco_tasks):
        # Round-robin across all GPUs
        worker = gpu_workers[idx % num_gpus]
        gpu_future = worker.tune_hsaco.remote(
            task_config_updated,
            kname,
            hsaco_index,
            task_id
        )
        gpu_futures.append(gpu_future)

        # Chain db_writer task - starts as soon as GPU task completes
        db_future = db_writer_task.remote(task_id, gpu_future)
        db_futures.append(db_future)

    # ========================================================================
    # Step 4: Wait for all GPU tasks and database writes to complete
    # ========================================================================
    logger.debug(f'Step 4: Waiting for {len(gpu_futures)} GPU tasks and DB writes...')
    reports = ray.get(gpu_futures)
    ray.get(db_futures)  # Ensure all writes complete
    logger.debug(f'All GPU tasks and database writes completed')

    # ========================================================================
    # Step 5: Postprocess on CPU (aggregate, write aggregation to DB, and cleanup)
    # ========================================================================
    logger.debug(f'Step 5: Postprocessing...')
    aggregation_ref = postprocess_task.remote(task_id, reports, task_config_updated)
    final_result = ray.get(aggregation_ref)

    logger.info(f'DAG execution completed for task_id={task_id}')
    return final_result
