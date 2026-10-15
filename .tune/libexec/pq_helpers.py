#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Shared PostgreSQL helpers for retry_missing_entries and reset_broken_to_pending.

Functions here operate on an open psycopg connection (dict_row factory assumed).
Connection management is the caller's responsibility.
"""

import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

try:
    import aotriton  # noqa: F401
except ImportError:
    sys.exit(
        "Error: the 'aotriton' package is not importable by python3.\n"
        f"  Install it with: pip install -e '{_REPO_ROOT}'"
    )

from aotriton.tune.pq import tasks as tune_tasks
from aotriton.tune.registry import load_flash_entry_module

FlashEntry = load_flash_entry_module().FlashEntry


def fetch_matches(conn, entries: list[tuple[str, FlashEntry]],
                  tuning_level: str) -> list[dict]:
    """Query pgdef.task for matching rows at one level (tuning_level).

    tuning_level is required: entries match on arch and payload fields, which
    both levels share, so an unfiltered query returns each entry's kernel and
    op rows together. Callers use the result to report a count and to write
    scratch/retry_task_ids.txt, so double-counting would inflate the
    confirmation prompt and feed the other level's ids to the recompute.
    """
    rows: list[dict] = []
    for arch, entry in entries:
        rows.extend(tune_tasks.find_by_entry(conn, entry, arch=arch, tuning_level=tuning_level))
    return rows


def fetch_matches_by_ids(conn, task_ids: list[int]) -> list[dict]:
    """Query pgdef.task for specific task_ids, return list of row dicts
    ({id, arch, module, tuning_level, status})."""
    return tune_tasks.find_by_ids(conn, task_ids)


def print_summary(label: str, count: int, matches: list[dict]) -> None:
    """Print a summary of matched pgdef.task rows by arch and status."""
    by_arch: Counter = Counter()
    by_status: Counter = Counter()
    for r in matches:
        by_arch[r['arch']] += 1
        by_status[r['status']] += 1

    print(f'\n{label} (de-duplicated): {count}')
    print(f'Matching pgdef.task rows:        {len(matches)}')

    if not matches:
        return

    print('\nBy arch:')
    for arch, n in sorted(by_arch.items()):
        print(f'  {arch}: {n}')

    print('\nBy current status:')
    for status, n in sorted(by_status.items()):
        print(f'  {status}: {n}')


def find_resettable(conn, what: str, *,
                    timeout_seconds: int | None = None) -> list[dict]:
    """Stale or failed tune tasks. Thin wrapper over
    aotriton.tune.pq.tasks.find_resettable, which owns the SQL."""
    return tune_tasks.find_resettable(conn, what, timeout_seconds=timeout_seconds)


def find_resettable_by_ids(conn, task_ids: list[int]) -> list[dict]:
    """Tune tasks for explicitly named ids, shaped like find_resettable.
    Thin wrapper over aotriton.tune.pq.tasks.find_resettable_by_ids."""
    return tune_tasks.find_resettable_by_ids(conn, task_ids)


def reset_to_pending(conn, row_ids: list[int], tuning_level: str, *,
                     delete_results: bool) -> int:
    """Reset the given task ids to pending. Returns affected row count.

    Thin wrapper over aotriton.tune.pq.tasks.reset, which owns the SQL.
    delete_results stays keyword-only and required here too -- forwarding it
    with a default would hide from this layer's callers that it can drop their
    reports and most_accurate_tuning_results rows.
    """
    return tune_tasks.reset(conn, row_ids, tuning_level, delete_results=delete_results)
