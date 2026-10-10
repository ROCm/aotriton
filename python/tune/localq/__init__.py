# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Unix socket-based node-local task queue for Tuner v3.5

Provides GPU-exclusive task execution with proper DAG orchestration.
Uses Unix domain sockets for IPC instead of Ray.
"""

__all__ = []

__version__ = '3.5.0'
