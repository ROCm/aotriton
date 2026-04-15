# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
GPU Worker Pool Management

Manages persistent GPU worker pools per module.
Each pool contains one worker per GPU.
"""

from typing import List
from .gpu_worker import GPUWorker


# Global worker pools: module -> [GPUWorker actors]
GPU_WORKER_POOLS = {}


def get_gpu_worker_pool(module: str, num_gpus: int = 8) -> List:
    """
    Get or create GPU worker pool for a module.

    Worker pools are persistent and reused across tasks. Each module
    gets its own pool because workers maintain module-specific exaid instances.

    Args:
        module: Tuning module name (e.g., 'attn_fwd')
        num_gpus: Number of GPUs (workers per pool)

    Returns:
        List of GPUWorker actor handles
    """
    if module not in GPU_WORKER_POOLS:
        print(f'[WorkerPool] Creating pool for module={module} with {num_gpus} GPUs')

        GPU_WORKER_POOLS[module] = [
            GPUWorker.remote(gpu_id, module)
            for gpu_id in range(num_gpus)
        ]

    return GPU_WORKER_POOLS[module]


def shutdown_worker_pool(module: str = None) -> None:
    """
    Shutdown worker pool(s).

    Args:
        module: Optional module name. If None, shutdown all pools.
    """
    import ray

    if module is None:
        # Shutdown all pools
        for mod, workers in GPU_WORKER_POOLS.items():
            print(f'[WorkerPool] Shutting down pool for module={mod}')
            for worker in workers:
                ray.kill(worker)

        GPU_WORKER_POOLS.clear()

    elif module in GPU_WORKER_POOLS:
        # Shutdown specific pool
        print(f'[WorkerPool] Shutting down pool for module={module}')
        workers = GPU_WORKER_POOLS.pop(module)

        for worker in workers:
            ray.kill(worker)


def get_worker_pool_stats() -> dict:
    """
    Get statistics about active worker pools.

    Returns:
        Dictionary with pool statistics
    """
    return {
        'num_pools': len(GPU_WORKER_POOLS),
        'modules': list(GPU_WORKER_POOLS.keys()),
        'total_workers': sum(len(workers) for workers in GPU_WORKER_POOLS.values())
    }
