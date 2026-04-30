#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Populate the most_accurate_tuning_results plain table.

Full mode (no task_ids):
    TRUNCATE most_accurate_tuning_results, then INSERT all rows.

Incremental mode (task_ids given):
    DELETE rows for the given task_ids, then INSERT only those task_ids.

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

_INSERT_SQL = """
INSERT INTO most_accurate_tuning_results
    (task_id, arch, task_config, kernel_name, test_case, tensor_name,
     target_fudge_factor, absolute_error)
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


def populate(conn, task_ids: list[int] | None = None) -> int:
    """
    Populate most_accurate_tuning_results.

    Args:
        conn:     psycopg connection. autocommit state is managed internally.
        task_ids: If None, full TRUNCATE + INSERT.
                  If given, DELETE matching rows then INSERT only those task_ids.

    Returns:
        Number of rows inserted (cur.rowcount after INSERT).
    """
    # Step 1: set parallel GUCs at session level outside any transaction so
    # the planner sees them unconditionally.
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute('SET max_parallel_workers_per_gather = 8')
        cur.execute('SET max_parallel_workers = 16')

    # Step 2: clear old rows and commit. Separating this from the INSERT means
    # the INSERT runs in a fresh transaction with no prior writes — PostgreSQL
    # only parallelizes INSERT...SELECT when no earlier writes exist in the txn.
    conn.autocommit = False
    with conn.cursor() as cur:
        if task_ids is None:
            cur.execute('TRUNCATE most_accurate_tuning_results')
        else:
            cur.execute(
                'DELETE FROM most_accurate_tuning_results WHERE task_id = ANY(%s)',
                (task_ids,),
            )
    conn.commit()

    # Step 3: INSERT in a fresh transaction — parallel scan now allowed.
    with conn.cursor() as cur:
        if task_ids is None:
            cur.execute(_INSERT_SQL.format(filter=''))
        else:
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
