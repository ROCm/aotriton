#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Force a specific tuning_results/optune_results row to be the "best" result
for its (task_id, kernel_name/op_name), overriding whatever
compute_best_results picked -- without touching
most_accurate_tuning_results/most_accurate_optune_results.

Use this when an entry is reported missing/broken but a specific existing
raw result row is known-good (e.g. manually verified) and should just be
used directly, bypassing compute_best_results' automatic accuracy-threshold
selection for that one (task_id, kernel_name/op_name).

Usage:
    python -m v3python.tune.pq.manual_pick_missing_entry --workdir /path/to/workdir --id <tuning_results.id>
    python -m v3python.tune.pq.manual_pick_missing_entry --workdir /path/to/workdir --id <id> --tuning_mode op
    python -m v3python.tune.pq.manual_pick_missing_entry --workdir /path/to/workdir --id <id> --dry_run
"""

import argparse
import logging
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb

from ..utils import get_db_connection_params
from .compute_best_results import SqlStatements

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def pick(conn, sql: SqlStatements, result_id: int, *, dry_run: bool = False) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f'SELECT task_id, {sql.key_col}, {sql.index_col}, result, result_data '
            f'FROM {sql.results_table} WHERE id = %s',
            (result_id,))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f'{sql.results_table}.id={result_id} not found')
        task_id, key_name, index, result, result_data = row

        if not result_data or not result_data.get('times'):
            raise ValueError(
                f'{sql.results_table}.id={result_id} (result={result!r}) has no '
                f'result_data.times -- cannot use it as a timed best result')
        median_time = result_data['times'][0]
        impl_desc = result_data.get('impl_desc')

        cur.execute('SELECT arch, task_config FROM task_queue WHERE id = %s', (task_id,))
        task_row = cur.fetchone()
        if task_row is None:
            raise ValueError(f'task_queue row not found for task_id={task_id}')
        arch, task_config = task_row

        cur.execute(
            f'SELECT {sql.index_col}, median_time FROM {sql.best_table} '
            f'WHERE task_id = %s AND {sql.key_col} = %s',
            (task_id, key_name))
        existing = cur.fetchone()

        logger.info('%s.id=%d: task_id=%d arch=%s %s=%s result=%s index=%d median_time=%.3fms',
                    sql.results_table, result_id, task_id, arch, sql.key_col, key_name,
                    result, index, median_time)
        if existing is not None:
            old_index, old_median_time = existing
            logger.info('Overriding existing %s row: index=%d median_time=%.3fms -> index=%d median_time=%.3fms',
                        sql.best_table, old_index, old_median_time, index, median_time)
        else:
            logger.info('No existing %s row for task_id=%d %s=%s; inserting new one',
                        sql.best_table, task_id, sql.key_col, key_name)

        if dry_run:
            logger.info('--dry_run: no changes written')
            return

        cur.execute(f"""
            INSERT INTO {sql.best_table}
                (task_id, arch, task_config, {sql.key_col}, {sql.index_col}, median_time, impl_desc)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (task_id, {sql.key_col}) DO UPDATE
                SET {sql.index_col} = EXCLUDED.{sql.index_col},
                    median_time = EXCLUDED.median_time,
                    impl_desc   = EXCLUDED.impl_desc,
                    computed_at = NOW()
        """, (task_id, arch, Jsonb(task_config), key_name, index, median_time,
              Jsonb(impl_desc) if impl_desc is not None else None))
    conn.commit()
    logger.info('Done: %s.id=%d is now the picked best for task_id=%d %s=%s',
                sql.results_table, result_id, task_id, sql.key_col, key_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--workdir', required=True, type=Path)
    parser.add_argument('--id', required=True, type=int,
                        help='tuning_results.id (optune_results.id in --tuning_mode op) to pick as best')
    parser.add_argument('--tuning_mode', choices=['kernel', 'op'], default='kernel')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show what would change without writing to best_tuning_results')
    args = parser.parse_args()

    sql = SqlStatements(args.tuning_mode)
    conn_params = get_db_connection_params(args.workdir)
    with psycopg.connect(**conn_params, autocommit=False) as conn:
        pick(conn, sql, args.id, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
