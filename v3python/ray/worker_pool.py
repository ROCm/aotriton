# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
GPU Worker Pool Management

Manages a single persistent GPU worker pool with 1:1 GPU mapping.
Each worker can handle any tuning module (exaid is cached by module, gpu_id).
"""

from typing import List
from .gpu_worker import GPUWorker


# Global worker pool: single pool with 1 worker per GPU
GPU_WORKER_POOL = None


def get_gpu_worker_pool(num_gpus: int = 4) -> List:
    """
    Get or create the GPU worker pool.

    Worker pool is persistent and reused across all tasks and modules.
    Each GPU gets exactly one worker (1:1 mapping).

    Args:
        num_gpus: Number of GPUs (workers in pool)

    Returns:
        List of GPUWorker actor handles
    """
    global GPU_WORKER_POOL

    if GPU_WORKER_POOL is None:
        print(f'[WorkerPool] Creating pool with {num_gpus} GPUs (1:1 mapping)')

        GPU_WORKER_POOL = [
            GPUWorker.remote(gpu_id)
            for gpu_id in range(num_gpus)
        ]

    return GPU_WORKER_POOL


def shutdown_worker_pool() -> None:
    """
    Shutdown the GPU worker pool.
    """
    import ray
    global GPU_WORKER_POOL

    if GPU_WORKER_POOL is not None:
        print(f'[WorkerPool] Shutting down pool with {len(GPU_WORKER_POOL)} workers')
        for worker in GPU_WORKER_POOL:
            ray.kill(worker)

        GPU_WORKER_POOL = None


def get_worker_pool_stats() -> dict:
    """
    Get statistics about the active worker pool.

    Returns:
        Dictionary with pool statistics
    """
    return {
        'num_workers': len(GPU_WORKER_POOL) if GPU_WORKER_POOL else 0,
        'active': GPU_WORKER_POOL is not None
    }
