# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
perfmon's PostgreSQL layer: schema `perfmon` (views only) and the queries
over it. SQL for perfmon lives here, and nowhere else. The engine's tables
(pgdef.task, pgdef.report) are written only through aotriton.pgdef.pq; this
package may read them directly.

Storage mapping: perfmon stores its application name as `cat_level0 =
'perfmon'`, with `cat_level1` NULL. The arch is the task's `pool`, the variant
its `subpool` ('any' when NULL). A measurement is a `pgdef.report` row with
rkey {"iface", "backend"}. Outside this package perfmon code says arch,
variant, subject, iface and backend.
"""

from .measurements import (ANY_VARIANT, STATUSES, count_superseded, iter_latest,  # noqa: F401
                           list_selections, save_measurement, subject_progress)
