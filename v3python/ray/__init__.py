# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Ray-based node-local task execution framework for Tuner v3.5

Provides GPU-exclusive task execution with proper DAG orchestration.
Replaces local Celery queues (CPUQ, GPUQ) with Ray actors.
"""

from .orchestrator import execute_tuning_dag, init_ray

__all__ = ['execute_tuning_dag', 'init_ray']

__version__ = '3.5.0'
