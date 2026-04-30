#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Populate the most_accurate_tuning_results plain table.

Full mode (no task_ids):
    DROP TABLE + CREATE TABLE AS SELECT (CTAS) — CTAS is parallel-safe
    unlike INSERT...SELECT, which PostgreSQL serializes because INSERT is
    parallel-restricted. After CTAS, rebuild the unique index.

Incremental mode (task_ids given):
    DELETE rows for the given task_ids, then INSERT only those task_ids.
    INSERT is acceptable here since the row count is small.

Usage:
    python3 -m v3python.tune.pq.populate_most_accurate <workdir>
    python3 -m v3python.tune.pq.populate_most_accurate <workdir> --task_ids_file /path/to/ids.txt
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root without install
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import psycopg

from v3python.tune.utils import get_db_connection_params

_SELECT_SQL = """
SELECT
    tr.task_id,
    tq.arch,
    tq.task_config,
    tr.kernel_name,
    test_case.key                               AS test_case,
    tensor.key                                  AS tensor_name,
    MIN((tensor.value->>0)::float)              AS target_fudge_factor,
    MIN((tensor.value->>1)::float)              AS absolute_error
FROM tuning_results tr
JOIN task_queue tq ON tq.id = tr.task_id
CROSS JOIN LATERAL jsonb_each(tr.result_data->'adiffs') AS test_case(key, value)
CROSS JOIN LATERAL jsonb_each(test_case.value)           AS tensor(key, value)
WHERE tr.result_data IS NOT NULL {filter}
GROUP BY tr.task_id, tq.arch, tq.task_config, tr.kernel_name, test_case.key, tensor.key
"""

_CTAS_SQL = 'CREATE TABLE most_accurate_tuning_results AS' + _SELECT_SQL

_INSERT_SQL = """INSERT INTO most_accurate_tuning_results
    (task_id, arch, task_config, kernel_name, test_case, tensor_name,
     target_fudge_factor, absolute_error)
""" + _SELECT_SQL

_INDEX_SQL = """
CREATE UNIQUE INDEX idx_most_accurate_tuning_results_lookup
    ON most_accurate_tuning_results (task_id, kernel_name, test_case, tensor_name)
"""


def populate(conn, task_ids: list[int] | None = None) -> int:
    """
    Populate most_accurate_tuning_results.

    Args:
        conn:     psycopg connection. autocommit state is managed internally.
        task_ids: If None, full DROP + CTAS (parallel).
                  If given, DELETE + INSERT for those task_ids only (small, serial ok).

    Returns:
        Number of rows produced (rowcount after CTAS or INSERT).
    """
    if task_ids is None:
        # Full mode: DROP + CREATE TABLE AS SELECT.
        # CTAS is parallel-safe; INSERT...SELECT is not (PostgreSQL serializes
        # it because INSERT is parallel-restricted).
        # Set GUCs at session level outside any transaction so the planner
        # sees them unconditionally.
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute('SET max_parallel_workers_per_gather = 8')
            cur.execute('SET max_parallel_workers = 16')
            # Force parallel plan — without these the planner may decide
            # serial is cheaper due to poor JSONB lateral cardinality estimates.
            cur.execute('SET parallel_setup_cost = 0')
            cur.execute('SET min_parallel_table_scan_size = 0')
            # Avoid disk sort spills: EXPLAIN ANALYZE showed each of 8 workers
            # spilling ~26MB to disk for the GROUP BY incremental sort.
            cur.execute("SET work_mem = '64MB'")
            # Skip JIT — for a single large CTAS it adds ~1900ms overhead
            # (inlining + optimization + emission) with no amortization benefit.
            cur.execute('SET jit = off')
        with conn.cursor() as cur:
            cur.execute('DROP TABLE IF EXISTS most_accurate_tuning_results')
            cur.execute('DROP INDEX IF EXISTS idx_most_accurate_tuning_results_lookup')
            cur.execute(_CTAS_SQL.format(filter=''))
            row_count = cur.rowcount
            cur.execute(_INDEX_SQL)
    else:
        # Incremental mode: row count is small, parallel not needed.
        # DELETE in one transaction, INSERT in a fresh one.
        conn.autocommit = False
        with conn.cursor() as cur:
            cur.execute(
                'DELETE FROM most_accurate_tuning_results WHERE task_id = ANY(%s)',
                (task_ids,),
            )
        conn.commit()
        with conn.cursor() as cur:
            cur.execute(_INSERT_SQL.format(filter='AND tr.task_id = ANY(%s)'), (task_ids,))
            row_count = cur.rowcount
        conn.commit()

    return row_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('workdir', help='Project workdir containing config.rc')
    parser.add_argument(
        '--task_ids_file',
        default=None,
        help='Path to file with one task_id per line (incremental mode)',
    )
    args = parser.parse_args()

    workdir = Path(args.workdir)
    if not workdir.is_dir():
        sys.exit(f'Error: workdir does not exist: {workdir}')

    task_ids: list[int] | None = None

    if args.task_ids_file is not None:
        ids_path = Path(args.task_ids_file)
        if not ids_path.is_file():
            sys.exit(f'Error: task_ids_file not found: {ids_path}')
        lines = ids_path.read_text().splitlines()
        task_ids = [int(line) for line in lines if line.strip()]
        if not task_ids:
            print('No task_ids. Nothing to do.')
            return

    conn_params = get_db_connection_params(workdir)

    if task_ids is None:
        print('Full populate: TRUNCATE + INSERT all rows...')
    else:
        print(f'Incremental populate: {len(task_ids)} task_id(s)...')

    with psycopg.connect(**conn_params, autocommit=False) as conn:
        row_count = populate(conn, task_ids)

    print(f'Done: {row_count} rows inserted into most_accurate_tuning_results.')


if __name__ == '__main__':
    main()
