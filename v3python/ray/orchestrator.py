# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
DAG Orchestrator for Ray-based task execution

Coordinates the full tuning workflow:
1. Preprocess (any GPU, load balanced)
2. Probe (any GPU, load balanced)
3. Distribute tune_hsaco tasks across all GPUs
4. Write results to PostgreSQL via CPU tasks
5. Postprocess and cleanup

Ensures GPU exclusivity via Ray actors while maximizing parallelism.
"""

import ray
import os
import logging
from typing import Dict, Any
from ray.util import ActorPool

from .worker_pool import get_gpu_worker_pool
from .cpu_tasks import db_writer_task, postprocess_task

logger = logging.getLogger(__name__)


def init_ray(address: str = 'auto') -> None:
    """
    Connect to existing Ray cluster.

    Workers should connect to a shared Ray cluster started by rayctl.
    Multiple worker_main.py instances share the same GPU worker pool.

    Args:
        address: Ray cluster address ('auto' to auto-discover)

    Connection mechanism:
        - rayctl starts Ray with --temp-dir=$WORKDIR/run/ray --node-ip-address=127.0.0.1
        - worker_service.sh exports RAY_TMPDIR=$WORKDIR/run/ray before starting worker_main.py
        - ray.init(address='auto') reads $RAY_TMPDIR and finds cluster at 127.0.0.1:6379
        - All workers share same Ray cluster (single node, localhost-only)
    """
    if ray.is_initialized():
        logger.debug('Already connected to Ray cluster')
        return

    logger.info(f'Connecting to Ray cluster at {address}')

    # Ray uses RAY_TMPDIR env var (set by worker_service.sh) to locate cluster
    ray.init(
        address=address,
        ignore_reinit_error=True,
        logging_level='warning'  # Reduce Ray verbosity
    )

    logger.info(f'Connected to Ray cluster successfully')


class TuningOrchestrator:
    """
    Orchestrates tuning tasks using Ray framework.

    Initialize once and reuse for multiple tasks to avoid overhead of
    creating ActorPool instances repeatedly.
    """

    def __init__(self, num_gpus: int = None):
        """
        Initialize orchestrator with GPU worker pool.

        Args:
            num_gpus: Number of GPUs (default: read from NUM_GPUS env var)
        """
        if num_gpus is None:
            num_gpus = int(os.environ.get('NUM_GPUS', 4))

        self.num_gpus = num_gpus
        self.gpu_workers = get_gpu_worker_pool(num_gpus)
        self.actor_pool = ActorPool(self.gpu_workers)

        logger.info(f'TuningOrchestrator initialized with {num_gpus} GPUs')

    def execute_tuning_dag(self, task_id: str, task_config: Dict[str, Any]) -> Dict[str, Any]:
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
        module = task_config["module"]

        logger.info(f'Starting DAG execution for task_id={task_id}, module={module}')

        # ========================================================================
        # Step 1 & 2: Preprocess and Probe - dispatch to any available GPU
        # ========================================================================
        # ActorPool automatically selects least-loaded worker

        # Submit preprocess task
        self.actor_pool.submit(lambda worker, cfg: worker.preprocess.remote(cfg), task_config)
        task_config_updated = self.actor_pool.get_next()

        logger.debug(f'Step 1: Preprocess completed (ActorPool auto-selected GPU)')

        # Submit probe task
        self.actor_pool.submit(lambda worker, cfg: worker.probe.remote(cfg), task_config_updated)
        hsaco_tasks = self.actor_pool.get_next()

        logger.debug(f'Step 2: Probe completed (ActorPool auto-selected GPU)')

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
        logger.debug(f'Step 3: Distributing {len(hsaco_tasks)} GPU tasks across {self.num_gpus} GPUs')
        gpu_futures = []
        db_futures = []

        for idx, (kname, hsaco_index) in enumerate(hsaco_tasks):
            # Round-robin across all GPUs
            worker = self.gpu_workers[idx % self.num_gpus]
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
