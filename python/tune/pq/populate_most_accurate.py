#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Populate the most_accurate_tuning_results plain table.

One most_accurate_tuning_results table serves both tuning levels. Rows are
keyed by iface_name, and tuning_level ('kernel' | 'op', selected by
--tuning_mode) is a filter column on tuning_results rather than a table-name
switch.

Full mode (no task_ids):
    CREATE TEMP TABLE AS SELECT (CTAS) into a scratch table scoped to this
    session — CTAS is parallel-safe unlike INSERT...SELECT, which PostgreSQL
    serializes because INSERT is parallel-restricted. Then, in one
    transaction: DELETE this run's tuning_level slice from the real table
    (NOT DROP TABLE -- most_accurate_tuning_results is shared by both tuning
    levels, so a full 'kernel' run must not discard 'op' rows and vice
    versa) and INSERT...SELECT from the temp table (cheap: a plain scan, not
    the JSONB lateral-join aggregation, so the INSERT parallel-restriction
    doesn't matter here).

Incremental mode (task_ids given):
    DELETE rows for the given task_ids, then INSERT only those task_ids.
    INSERT is acceptable here since the row count is small.

Usage:
    python3 -m aotriton.tune.pq.populate_most_accurate <workdir>
    python3 -m aotriton.tune.pq.populate_most_accurate <workdir> --task_ids_file /path/to/ids.txt
    query_broken_entries ~/wkdir | python3 -m aotriton.tune.pq.populate_most_accurate <workdir> --task_ids_file -
"""

import argparse
import sys
from pathlib import Path

import psycopg

from ..utils import get_db_connection_params

class SqlStatements:
    """Table and column names are fixed; only tuning_level varies with
    --tuning_mode. Every query against tuning_results /
    most_accurate_tuning_results must filter on it, because iface_name
    collides across levels (e.g. 'attn_fwd' is valid at both)."""

    table_name = 'most_accurate_tuning_results'
    key_col = 'iface_name'

    def __init__(self, tuning_mode: str):
        self.tuning_level = tuning_mode  # 'kernel' | 'op'

    @property
    def temp_table_name(self) -> str:
        return f'most_accurate_tuning_results_tmp_{self.tuning_level}'

    @property
    def _select_sql(self) -> str:
        return f"""
SELECT
    tr.task_id,
    tq.arch,
    tq.task_config,
    tr.iface_name,
    test_case.key                                   AS test_case,
    tensor.key                                      AS tensor_name,
    MIN((tensor.value->>0)::float)                  AS target_fudge_factor,
    MIN((tensor.value->>1)::float)                  AS absolute_error
FROM tuning_results tr
JOIN task_queue tq ON tq.id = tr.task_id
CROSS JOIN LATERAL jsonb_each(tr.result_data->'adiffs') AS test_case(key, value)
CROSS JOIN LATERAL jsonb_each(test_case.value)           AS tensor(key, value)
WHERE tr.result_data IS NOT NULL
  AND tr.tuning_level = '{self.tuning_level}'
  AND (tensor.value IS NULL OR (tensor.value->>1)::float >= 0.0) {{filter}}
GROUP BY tr.task_id, tq.arch, tq.task_config, tr.iface_name, test_case.key, tensor.key
"""

    @property
    def ctas_temp_sql(self) -> str:
        return f'CREATE TEMP TABLE {self.temp_table_name} AS' + self._select_sql

    @property
    def swap_insert_sql(self) -> str:
        return f"""INSERT INTO {self.table_name}
    (task_id, arch, tuning_level, task_config, {self.key_col}, test_case, tensor_name,
     target_fudge_factor, absolute_error)
SELECT task_id, arch, '{self.tuning_level}', task_config, {self.key_col}, test_case,
       tensor_name, target_fudge_factor, absolute_error
FROM {self.temp_table_name}
"""

    @property
    def insert_sql(self) -> str:
        return f"""INSERT INTO {self.table_name}
    (task_id, arch, tuning_level, task_config, {self.key_col}, test_case, tensor_name,
     target_fudge_factor, absolute_error)
SELECT task_id, arch, '{self.tuning_level}', task_config, {self.key_col}, test_case,
       tensor_name, target_fudge_factor, absolute_error
FROM (""" + self._select_sql + ") sub"


def populate(conn, task_ids: list[int] | None = None, tuning_mode: str = 'kernel') -> int:
    """
    Populate most_accurate_tuning_results for one tuning_level.

    Args:
        conn:         psycopg connection. autocommit state is managed internally.
        task_ids:     If None, full CTAS-into-temp-table then swap (parallel).
                      If given, DELETE + INSERT for those task_ids only (small, serial ok).
        tuning_mode:  'kernel' | 'op' -- selects the tuning_level filter applied
                      to tuning_results.

    Returns:
        Number of rows produced (rowcount after the swap-INSERT or plain INSERT).
    """
    sql = SqlStatements(tuning_mode)

    if task_ids is None:
        # Full mode: CREATE TEMP TABLE AS SELECT (CTAS is parallel-safe;
        # INSERT...SELECT is not, PostgreSQL serializes it because INSERT is
        # parallel-restricted), then swap into the real table inside one
        # transaction. NOT DROP TABLE: most_accurate_tuning_results is shared
        # by both tuning levels, so a full 'kernel' run must not discard 'op'
        # rows and vice versa -- only this run's tuning_level slice is
        # replaced.
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
            cur.execute(f'DROP TABLE IF EXISTS {sql.temp_table_name}')
            cur.execute(sql.ctas_temp_sql.format(filter=''))

        # Swap: replace only this tuning_level's rows in the real table.
        conn.autocommit = False
        with conn.cursor() as cur:
            cur.execute(
                f'DELETE FROM {sql.table_name} WHERE tuning_level = %s',
                (sql.tuning_level,),
            )
            cur.execute(sql.swap_insert_sql)
            row_count = cur.rowcount
            cur.execute(f'DROP TABLE IF EXISTS {sql.temp_table_name}')
        conn.commit()
    else:
        # Incremental mode: row count is small, parallel not needed.
        # DELETE in one transaction, INSERT in a fresh one.
        conn.autocommit = False
        with conn.cursor() as cur:
            cur.execute(
                f'DELETE FROM {sql.table_name} WHERE tuning_level = %s AND task_id = ANY(%s)',
                (sql.tuning_level, task_ids),
            )
        conn.commit()
        with conn.cursor() as cur:
            cur.execute(sql.insert_sql.format(filter='AND tr.task_id = ANY(%s)'), (task_ids,))
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
        help='Path to file with one task_id per line (incremental mode); use - to read from stdin',
    )
    parser.add_argument(
        '--tuning_mode',
        choices=['kernel', 'op'],
        default='kernel',
        help='Selects the tuning_level filter applied to tuning_results',
    )
    args = parser.parse_args()

    workdir = Path(args.workdir)
    if not workdir.is_dir():
        sys.exit(f'Error: workdir does not exist: {workdir}')

    task_ids: list[int] | None = None

    if args.task_ids_file is not None:
        if args.task_ids_file == '-':
            lines = sys.stdin.read().splitlines()
        else:
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
        print(f'Full populate ({args.tuning_mode}): replace this tuning_level\'s rows...')
    else:
        print(f'Incremental populate ({args.tuning_mode}): {len(task_ids)} task_id(s)...')

    with psycopg.connect(**conn_params, autocommit=False) as conn:
        row_count = populate(conn, task_ids, tuning_mode=args.tuning_mode)

    table = SqlStatements(args.tuning_mode).table_name
    print(f'Done: {row_count} rows inserted into {table} (tuning_level={args.tuning_mode}).')


if __name__ == '__main__':
    main()
