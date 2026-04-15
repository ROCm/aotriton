# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Ray-based node-local task execution framework for Tuner v3.5

Provides GPU-exclusive task execution with proper DAG orchestration.
Replaces local Celery queues (CPUQ, GPUQ) with Ray actors.
"""

from .orchestrator import init_ray, TuningOrchestrator

__all__ = ['init_ray', 'TuningOrchestrator']

__version__ = '3.5.0'
