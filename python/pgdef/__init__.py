# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
pgdef: the PostgreSQL Distributed Execution Framework.

An application-independent engine: a task queue in PostgreSQL (`pq`), a
per-node handler runtime (`localq`) and supervised subprocess runners
(`exaid`). It is a library -- applications such as `aotriton.tune` call it,
and it never imports them, torch or triton. See
docs/Distributed Execution Framework.md.
"""
