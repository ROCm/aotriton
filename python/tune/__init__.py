# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Tuner v3.5: the "tune" application on pgdef (aotriton.pgdef).

Entry points: python -m aotriton.tune.{node,dispatch,manifest}. The handler graph is
handlers.py; the PostgreSQL layer is pq/ (schema `tune`).
"""

__version__ = '3.5.0'
