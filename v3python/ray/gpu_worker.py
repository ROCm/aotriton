# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
GPU Worker Actor for Ray-based task execution

Handles ALL GPU task types (preprocess, probe, tune_hsaco) with exclusive access.
Ray ensures actor methods run serially - one task at a time per GPU.
"""

import ray
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple


@ray.remote(num_gpus=1)
class GPUWorker:
    """
    Persistent GPU worker that handles any tuning module.
    Owns one GPU exclusively (1:1 mapping).

    Ray guarantees:
    - Methods on same actor run serially (exclusive GPU access)
    - num_gpus=1 reserves GPU resources
    - Different actors (different GPUs) run in parallel

    exaid instances are cached by (module, gpu_id), so we call
    exaid_create() in each method rather than storing a reference.
    """

    def __init__(self, gpu_id: int):
        """
        Initialize GPU worker.

        Args:
            gpu_id: GPU device ID (0-based)
        """
        self.gpu_id = gpu_id

    def preprocess(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocessing: prepare test data (needs GPU).
        Runs exclusively on this GPU.

        Args:
            task_config: Task configuration dictionary with 'module' key

        Returns:
            Updated task_config with tmpdir path

        Raises:
            OSError: Filesystem errors
            ExaidSubprocessNotOK: Subprocess execution failures
        """
        from v3python.tune.exaid import exaid_create, ExaidSubprocessNotOK

        module = task_config['module']
        exaid = exaid_create(module, self.gpu_id)

        if 'tmpdir' in task_config:
            tmpdir = Path(task_config['tmpdir'])
        else:
            tmpdir = exaid.get_tmpfs_for(task_config["entry"])

        try:
            exaid.prepare_data(task_config["entry"], tmpdir)
            task_config['tmpdir'] = tmpdir.as_posix()
            print(f'[exaid][GPU{self.gpu_id}][preprocess][{module}] Data prepared in {tmpdir}')
            return task_config

        except (OSError, ExaidSubprocessNotOK) as e:
            print(f'[exaid][GPU{self.gpu_id}][preprocess][{module}] ERROR: {e}')
            raise

    def probe(self, task_config: Dict[str, Any]) -> List[Tuple[str, int]]:
        """
        Probing: discover hsaco kernels (needs GPU).
        Runs exclusively on this GPU.

        Args:
            task_config: Task configuration with tmpdir and 'module' key

        Returns:
            List of (kernel_name, hsaco_index) tuples

        Raises:
            OSError: Filesystem errors
            ExaidSubprocessNotOK: Subprocess execution failures
        """
        from v3python.tune.exaid import exaid_create, ExaidSubprocessNotOK

        module = task_config['module']
        exaid = exaid_create(module, self.gpu_id)
        tmpdir = Path(task_config['tmpdir'])

        try:
            kernel_dict = exaid.probe(tmpdir)

            # Flatten to list of (kname, hsaco_index) tuples
            hsaco_tasks = []
            max_hsaco_dict = task_config.get("max_hsaco", {})
            max_hsaco_global = max_hsaco_dict.get("*", None)

            for kname, hsaco_list in kernel_dict.items():
                max_hsaco = max_hsaco_dict.get(kname, max_hsaco_global)
                # Limit number of hsaco variants if specified
                limited_hsaco = hsaco_list[:max_hsaco] if max_hsaco else hsaco_list

                for hsaco_index in range(len(limited_hsaco)):
                    hsaco_tasks.append((kname, hsaco_index))

            print(f'[exaid][GPU{self.gpu_id}][probe][{module}] Found {len(hsaco_tasks)} hsaco kernels')
            return hsaco_tasks

        except (OSError, ExaidSubprocessNotOK) as e:
            print(f'[exaid][GPU{self.gpu_id}][probe][{module}] ERROR: {e}')
            raise

    def tune_hsaco(
        self,
        task_config: Dict[str, Any],
        kname: str,
        hsaco_index: int,
        task_id: str
    ) -> Dict[str, Any]:
        """
        Benchmark one hsaco (needs GPU).
        Runs exclusively on this GPU.
        Returns report WITHOUT writing to DB (offloaded to CPU task).

        Args:
            task_config: Task configuration with 'module' key
            kname: Kernel name
            hsaco_index: HSACO variant index
            task_id: Task ID for tracking

        Returns:
            Report dictionary with benchmark results or error
        """
        from v3python.tune.exaid import exaid_create, ExaidSubprocessNotOK

        module = task_config['module']
        exaid = exaid_create(module, self.gpu_id)
        tmpdir = Path(task_config['tmpdir'])

        report = {
            "task_config": task_config,
            "complete_on_gpu": self.gpu_id,
            "kernel_name": kname,
            "hsaco_index": hsaco_index,
        }

        try:
            result_data = exaid.benchmark(tmpdir, kname, hsaco_index)
            report['result'] = "OK"
            report['result_data'] = result_data
            print(f'[exaid][GPU{self.gpu_id}][tune_hsaco][{module}] {kname}[{hsaco_index}] OK')

        except OSError as e:
            print(f'[exaid][GPU{self.gpu_id}][tune_hsaco][{module}] {kname}[{hsaco_index}] OSError')
            report['result'] = "crash"
            report['error'] = {"errno": e.errno, "stderr": e.strerror}

        except ExaidSubprocessNotOK as e:
            print(f'[exaid][GPU{self.gpu_id}][tune_hsaco][{module}] {kname}[{hsaco_index}] NotOK')
            report['result'] = "NotOK"
            report['error'] = {"stdout": e.stdout, "stderr": e.stderr}

        except Exception as e:
            print(f'[exaid][GPU{self.gpu_id}][tune_hsaco][{module}] {kname}[{hsaco_index}] Unexpected error: {e}')
            report['result'] = "ERROR"
            report['error'] = str(e)

        return report
