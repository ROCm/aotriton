# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
tune's PostgreSQL layer: schema `tune` and the queries over it.

SQL for tune's tables lives here, and nowhere else. The engine's tables
(pgdef.task, pgdef.report, ...) are written only through aotriton.pgdef.pq;
this package may read them directly.

Storage mapping: tune stores its application name as `cat_level0 = 'tune'` and
its tuning level ('kernel'|'op') as `cat_level1` in pgdef (pgdef.task,
pgdef.report.rkey) and in tune's own tables. Outside this package tune code
says `tuning_level`.
"""
