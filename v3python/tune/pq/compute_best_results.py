#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Compute best_tuning_results / best_optune_results table using Python multiprocessing.

kernel mode (default):
    For each (task_id, kernel_name): find the fastest hsaco_index that passes
    the accuracy threshold — absolute_error <= 10x the minimum across all
    (test_case, tensor_name) pairs in most_accurate_tuning_results.

op mode (--tuning_mode op):
    For each (task_id, op_name): find the fastest backend_index that passes.
    Negative adiffs (early-reject sentinel) are treated as passed for that tensor.
    A backend is only considered valid if at least one test case starting with
    '00_' has a non-negative adiff (basic UT was actually exercised).

reference_error IS NULL  → tensor genuinely inapplicable → pass
absolute_error IS NULL with real reference_error → kernel/backend broken → fail

One worker process is spawned per arch (matching task_queue's arch partitions).
Each worker opens its own DB connection and streams only its arch's rows,
so no result_data is ever sent over IPC — only small result tuples are returned.

Usage:
    python -m v3python.tune.pq.compute_best_results --workdir /path/to/workdir
    python -m v3python.tune.pq.compute_best_results --workdir /path/to/workdir --incremental
    python -m v3python.tune.pq.compute_best_results --workdir /path/to/workdir --fix <pass>
    python -m v3python.tune.pq.compute_best_results --workdir /path/to/workdir --fix <hostname>:<pass>
    python -m v3python.tune.pq.compute_best_results --workdir /path/to/workdir --tuning_mode op
"""

import argparse
import json
import logging
import math
import multiprocessing
import time
from itertools import groupby
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb

from ..utils import get_db_connection_params

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

ACCURACY_MULTIPLIER = 3.0
WORKERS_PER_ARCH = 2        # sub-workers per arch, sliced by task_id
STREAM_BATCH_SIZE = 10_000  # rows fetched per server-side cursor batch
INSERT_BATCH_SIZE = 1_000   # rows per bulk INSERT

# Test cases excluded from accuracy gating (known kernel bugs whose results
# cannot be used as a correctness criterion for hsaco selection).
SKIP_TEST_CASES: frozenset[str] = frozenset({
    '01_gqa',  # GQA bias indexing bug in bwd dk/dv kernels (fixed, pending re-tune)
})


# ---------------------------------------------------------------------------
# SQL name bundle — keeps all table/column names in one place
# ---------------------------------------------------------------------------

class SqlStatements:
    def __init__(self, tuning_mode: str):
        self._tuning_mode = tuning_mode

    @property
    def results_table(self) -> str:
        return 'optune_results' if self._tuning_mode == 'op' else 'tuning_results'

    @property
    def accuracy_table(self) -> str:
        return 'most_accurate_optune_results' if self._tuning_mode == 'op' else 'most_accurate_tuning_results'

    @property
    def best_table(self) -> str:
        return 'best_optune_results' if self._tuning_mode == 'op' else 'best_tuning_results'

    @property
    def key_col(self) -> str:
        return 'op_name' if self._tuning_mode == 'op' else 'kernel_name'

    @property
    def index_col(self) -> str:
        return 'backend_index' if self._tuning_mode == 'op' else 'hsaco_index'

    @property
    def retry_ids_file(self) -> str:
        return 'retry_optune_ids.txt' if self._tuning_mode == 'op' else 'retry_task_ids.txt'

    @property
    def broken_table(self) -> str:
        return 'broken_op_entries' if self._tuning_mode == 'op' else 'broken_entries'


# ---------------------------------------------------------------------------
# Per-group computation (called inline inside each worker process)
# ---------------------------------------------------------------------------

from typing import NamedTuple

class Audit(NamedTuple):
    passes: bool
    executed: bool  # False when abs_err < 0 (early-reject, never ran on hardware)


def _find_best_candidate(task_id: int, key_name: str,
                         group_rows: list, thresholds: dict,
                         tuning_mode: str = 'kernel',
                         verbose: bool = False) -> tuple | None:
    """
    Find the fastest hsaco/backend that passes the accuracy threshold.

    group_rows:   [(index, result_data), ...]  (result_data is a dict, already parsed)
    thresholds:   {(test_case, tensor_name): min_absolute_error}
    tuning_mode:  'kernel' or 'op'
    verbose:      if True, log per-candidate and per-tensor audit details

    op mode: negative adiffs (early-reject) are treated as passed per tensor;
             backend only valid if at least one '00_' test case has non-negative adiff.

    Returns (task_id, key_name, index, median_time, impl_desc) or None.
    """
    log = logging.getLogger(__name__)
    if not thresholds:
        log.warning('task_id=%d key=%s: thresholds empty — no accuracy gating, '
                    'fastest candidate wins unconditionally', task_id, key_name)
    if tuning_mode == 'op':
        def tensor_audit(abs_err, tc, tname) -> Audit:
            if abs_err is not None and abs_err < 0:
                return Audit(passes=True, executed=False)   # early-reject sentinel
            if abs_err is None:
                return Audit(passes=False, executed=False)  # backend broken / NaN
            min_err = thresholds.get((tc, tname))
            if min_err is not None and abs_err > ACCURACY_MULTIPLIER * min_err:
                return Audit(passes=False, executed=True)
            return Audit(passes=True, executed=True)

        def update_basic_ut(acc, tc, tc_executed) -> bool:
            return acc or (tc.startswith('00_') and tc_executed)

        def final_gate(passes, has_basic_ut_pass) -> bool:
            return passes and has_basic_ut_pass
    else:
        def tensor_audit(abs_err, tc, tname) -> Audit:
            if abs_err is None:
                return Audit(passes=False, executed=False)  # kernel broken / NaN
            min_err = thresholds.get((tc, tname))
            if min_err is not None and abs_err > ACCURACY_MULTIPLIER * min_err:
                return Audit(passes=False, executed=True)
            return Audit(passes=True, executed=True)

        def update_basic_ut(acc, tc, tc_executed) -> bool:
            return acc

        def final_gate(passes, has_basic_ut_pass) -> bool:
            return passes

    best = None

    for index, rd in group_rows:
        times = rd.get('times')
        if not times:
            if verbose:
                log.debug('task_id=%d key=%s index=%d: skipped — no times', task_id, key_name, index)
            continue
        median_time = times[0]
        impl_desc = rd.get('impl_desc')

        if verbose:
            log.debug('task_id=%d key=%s index=%d: times=[%.3f, %.3f, %.3f] impl=%s',
                      task_id, key_name, index,
                      times[0], times[1], times[2],
                      impl_desc)

        passes = True
        has_basic_ut_pass = False

        for tc, tensors in rd.get('adiffs', {}).items():
            if tc in SKIP_TEST_CASES:
                if verbose:
                    log.debug('  tc=%s: skipped (in SKIP_TEST_CASES)', tc)
                continue
            tc_executed = False
            for tname, vals in tensors.items():
                if not vals:
                    if verbose:
                        log.debug('  tc=%s tname=%s: inapplicable (null vals)', tc, tname)
                    continue  # JSON null or empty array → inapplicable
                ref_err = vals[2] if len(vals) > 2 else None
                if ref_err is None:
                    if verbose:
                        log.debug('  tc=%s tname=%s: inapplicable (ref_err=None)', tc, tname)
                    continue  # tensor genuinely inapplicable → pass
                abs_err = vals[1] if len(vals) > 1 else None
                min_err = thresholds.get((tc, tname))
                a = tensor_audit(abs_err, tc, tname)
                if verbose:
                    threshold_str = f'{ACCURACY_MULTIPLIER}×{min_err:.4e}={ACCURACY_MULTIPLIER * min_err:.4e}' if min_err is not None else 'no-threshold'
                    log.debug('  tc=%s tname=%s: abs_err=%s ref_err=%.4e threshold=%s → %s',
                              tc, tname,
                              f'{abs_err:.4e}' if abs_err is not None else 'None',
                              ref_err, threshold_str,
                              'PASS' if a.passes else 'FAIL')
                if not a.passes:
                    passes = False
                    break
                tc_executed = tc_executed or a.executed
            if not passes:
                break
            has_basic_ut_pass = update_basic_ut(has_basic_ut_pass, tc, tc_executed)

        gate = final_gate(passes, has_basic_ut_pass)
        if verbose:
            log.debug('  → passes=%s has_basic_ut_pass=%s gate=%s%s',
                      passes, has_basic_ut_pass, gate,
                      ' NEW_BEST' if gate and (best is None or median_time < best[1]) else '')
        if gate and (best is None or median_time < best[1]):
            best = (index, median_time, impl_desc)

    if best is None:
        return None
    index, median_time, impl_desc = best
    return (task_id, key_name, index, median_time, impl_desc)


# ---------------------------------------------------------------------------
# Step 1: load accuracy table, keyed by arch
# ---------------------------------------------------------------------------

def load_thresholds(conn, sql: SqlStatements, task_ids: list[int] | None = None) -> dict:
    """
    Returns:
        {arch: {task_id: {key_name: {(test_case, tensor_name): min_absolute_error}}}}

    Keyed by arch so each worker process receives only its own slice.
    If task_ids is given, only those task_ids are loaded (incremental/fix mode).
    """
    if task_ids:
        logger.info('Step 1: loading %s for %d task_id(s)...', sql.accuracy_table, len(task_ids))
    else:
        logger.info('Step 1: loading %s into RAM...', sql.accuracy_table)
    t0 = time.monotonic()

    thresholds: dict = {}
    with conn.cursor() as cur:
        if task_ids:
            cur.execute(f"""
                SELECT arch, task_id, {sql.key_col}, test_case, tensor_name, absolute_error
                FROM {sql.accuracy_table}
                WHERE task_id = ANY(%s)
            """, (task_ids,))
        else:
            cur.execute(f"""
                SELECT arch, task_id, {sql.key_col}, test_case, tensor_name, absolute_error
                FROM {sql.accuracy_table}
            """)
        for arch, task_id, key_name, test_case, tensor_name, abs_err in cur:
            tk = thresholds.setdefault(arch, {}).setdefault(task_id, {}).setdefault(key_name, {})
            tk[(test_case, tensor_name)] = abs_err

    n = sum(
        len(kn)
        for at in thresholds.values()
        for tk in at.values()
        for kn in tk.values()
    )
    logger.info('Step 1 done: %d archs, %d threshold entries in %.1fs',
                len(thresholds), n, time.monotonic() - t0)
    return thresholds


def get_archs(conn, sql: SqlStatements) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(f'SELECT DISTINCT arch FROM {sql.accuracy_table} ORDER BY arch')
        return [row[0] for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Step 2+3: worker — one per arch, owns its own DB connection
# ---------------------------------------------------------------------------

def worker_process_arch(arch: str, worker_index: int,
                        task_id_lo: int, task_id_hi: int,
                        conn_params: dict, arch_thresholds: dict,
                        tuning_mode: str, verbose: bool = False,
                        filter_task_ids: list[int] | None = None) -> list:
    """
    Runs in a child process. Streams results for one (arch, task_id range),
    processes each (task_id, key_name) group inline, returns best rows.

    task_id_lo/hi:    inclusive task_id range this worker is responsible for
                      (ignored when filter_task_ids is given)
    arch_thresholds:  {task_id: {key_name: {(test_case, tensor_name): min_absolute_error}}}
                      full arch slice; worker only accesses task_ids in its range/filter
    filter_task_ids:  if given, restrict the DB query to exactly these task_ids
                      (incremental/fix/ids mode); caller sends one worker per arch

    Returns list of (task_id, key_name, index, median_time, impl_desc).
    """
    sql = SqlStatements(tuning_mode)

    # Each worker configures its own logger (basicConfig not inherited reliably).
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s [%(arch)s] %(message)s'.replace('%(arch)s', arch),
        datefmt='%H:%M:%S',
    )
    log = logging.getLogger(__name__)
    t0 = time.monotonic()
    results = []
    row_count = 0
    group_count = 0
    LOG_INTERVAL = 15.0
    # Stagger: worker i first logs at (i+1)*3s, then every 15s thereafter.
    last_log_time = t0 + (worker_index + 1) * 3 - LOG_INTERVAL

    def _execute_and_group(cur):
        # Count groups in thresholds for this worker's scope (new tasks may push
        # group_count above this estimate if tuning runs concurrently).
        if filter_task_ids is not None:
            total_groups = sum(len(kn_dict) for tid, kn_dict in arch_thresholds.items()
                               if tid in set(filter_task_ids))
            log.info('arch=%s chunk=%d [incremental %d task_id(s)]: ~%d groups to process',
                     arch, worker_index, len(filter_task_ids), total_groups)
            cur.execute(f"""
                SELECT tr.task_id, tr.{sql.key_col}, tr.{sql.index_col}, tr.result_data
                FROM {sql.results_table} tr
                JOIN task_queue tq ON tq.id = tr.task_id AND tq.arch = %s
                WHERE tr.result_data IS NOT NULL
                  AND tr.task_id = ANY(%s)
                ORDER BY tr.task_id, tr.{sql.key_col}
            """, (arch, filter_task_ids))
        else:
            total_groups = sum(len(kn_dict) for tid, kn_dict in arch_thresholds.items()
                               if task_id_lo <= tid <= task_id_hi)
            log.info('arch=%s chunk=%d [%d, %d]: ~%d groups to process',
                     arch, worker_index, task_id_lo, task_id_hi, total_groups)
            cur.execute(f"""
                SELECT tr.task_id, tr.{sql.key_col}, tr.{sql.index_col}, tr.result_data
                FROM {sql.results_table} tr
                JOIN task_queue tq ON tq.id = tr.task_id AND tq.arch = %s
                WHERE tr.result_data IS NOT NULL
                  AND tr.task_id BETWEEN %s AND %s
                ORDER BY tr.task_id, tr.{sql.key_col}
            """, (arch, task_id_lo, task_id_hi))
        return total_groups, groupby(cur, key=lambda r: (r[0], r[1]))

    with psycopg.connect(**conn_params, autocommit=True) as conn:
        with conn.transaction(), conn.cursor(name=f'worker_{arch}_{worker_index}') as cur:
            cur.itersize = STREAM_BATCH_SIZE
            total_groups, grouped_rows = _execute_and_group(cur)
            for (task_id, key_name), rows in grouped_rows:
                # Parse JSONB here in the worker — never crosses IPC boundary.
                group_rows = []
                for r in rows:
                    index, rd = r[2], r[3]
                    if isinstance(rd, str):
                        try:
                            rd = json.loads(rd)
                        except (json.JSONDecodeError, TypeError):
                            continue
                    group_rows.append((index, rd))

                row_count += len(group_rows)
                group_count += 1

                task_thresholds = arch_thresholds.get(task_id, {}).get(key_name, {})
                result = _find_best_candidate(task_id, key_name, group_rows, task_thresholds,
                                              tuning_mode=tuning_mode, verbose=verbose)
                if result is not None:
                    results.append(result)

                now = time.monotonic()
                if now - last_log_time >= LOG_INTERVAL:
                    elapsed = now - t0
                    pct = min(100.0, 100.0 * group_count / total_groups) if total_groups else 0.0
                    eta = elapsed / group_count * (total_groups - group_count) if group_count else 0.0
                    log.info('arch=%s: %d/%d groups (%.1f%%), %d rows, '
                             'elapsed %.0fs, ETA %.0fs',
                             arch, group_count, total_groups, pct,
                             row_count, elapsed, eta)
                    last_log_time = now

    log.info('arch=%s chunk=%d done: %d/%d groups, %d rows → %d best results in %.1fs',
             arch, worker_index, group_count, total_groups, row_count, len(results),
             time.monotonic() - t0)
    return results


# ---------------------------------------------------------------------------
# Step 4: write results to best table
# ---------------------------------------------------------------------------

def write_results(conn, sql: SqlStatements, arch_results: list, incremental: bool = False) -> None:
    """
    Writes rows to best_tuning_results or best_optune_results.
    Full mode: truncates first. Incremental mode: upserts only affected rows.
    """
    logger.info('Step 4: writing %d rows to %s (%s)...',
                len(arch_results), sql.best_table, 'incremental' if incremental else 'full')
    t0 = time.monotonic()

    task_ids = list({r[0] for r in arch_results})
    task_config_map: dict = {}
    with conn.cursor() as cur:
        for i in range(0, len(task_ids), 1000):
            batch = task_ids[i:i + 1000]
            cur.execute('SELECT id, task_config FROM task_queue WHERE id = ANY(%s)', (batch,))
            for row in cur.fetchall():
                task_config_map[row[0]] = row[1]

    arch_map: dict = {}
    with conn.cursor() as cur:
        for i in range(0, len(task_ids), 1000):
            batch = task_ids[i:i + 1000]
            cur.execute('SELECT id, arch FROM task_queue WHERE id = ANY(%s)', (batch,))
            for row in cur.fetchall():
                arch_map[row[0]] = row[1]

    rows_to_insert = []
    for task_id, key_name, index, median_time, impl_desc in arch_results:
        task_config = task_config_map.get(task_id)
        arch = arch_map.get(task_id)
        if task_config is None or arch is None:
            continue
        rows_to_insert.append((task_id, arch, Jsonb(task_config), key_name,
                               index, median_time,
                               Jsonb(impl_desc) if impl_desc is not None else None))

    with conn.cursor() as cur:
        if not incremental:
            cur.execute(f'TRUNCATE TABLE {sql.best_table}')
        for i in range(0, len(rows_to_insert), INSERT_BATCH_SIZE):
            batch = rows_to_insert[i:i + INSERT_BATCH_SIZE]
            cur.executemany(f"""
                INSERT INTO {sql.best_table}
                    (task_id, arch, task_config, {sql.key_col},
                     {sql.index_col}, median_time, impl_desc)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (task_id, {sql.key_col}) DO UPDATE
                    SET {sql.index_col} = EXCLUDED.{sql.index_col},
                        median_time = EXCLUDED.median_time,
                        impl_desc   = EXCLUDED.impl_desc,
                        computed_at = NOW()
            """, batch)
            if (i // INSERT_BATCH_SIZE + 1) % 100 == 0:
                logger.info('  inserted %d / %d rows', i + len(batch), len(rows_to_insert))

    conn.commit()
    logger.info('Step 4 done: %d rows written in %.1fs',
                len(rows_to_insert), time.monotonic() - t0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _resolve_task_ids_incremental(workdir: Path, sql: SqlStatements) -> list[int]:
    """Read task_ids from scratch/<retry_ids_file> (written by reset_broken_to_pending)."""
    cache = workdir / 'scratch' / sql.retry_ids_file
    if not cache.is_file():
        raise FileNotFoundError(f'--incremental: {cache} not found; run reset_broken_to_pending first')
    ids = [int(line.strip()) for line in cache.read_text().splitlines() if line.strip()]
    if not ids:
        raise ValueError(f'--incremental: {cache} is empty')
    logger.info('Incremental mode: %d task_id(s) from %s', len(ids), cache)
    return ids


def _resolve_task_ids_fix(workdir: Path, sql: SqlStatements, fix_spec: str) -> list[int]:
    """Read task_ids from broken_entries.db / broken_op_entries table for the given [hostname:]pass spec."""
    import sqlite3
    if ':' in fix_spec:
        hostname, pass_str = fix_spec.rsplit(':', 1)
    else:
        hostname, pass_str = None, fix_spec
    pass_num = int(pass_str)

    db_path = workdir / 'scratch' / 'broken_entries.db'
    if not db_path.is_file():
        raise FileNotFoundError(f'--fix: {db_path} not found')

    with sqlite3.connect(db_path) as db:
        if hostname:
            cur = db.execute(
                f'SELECT task_id FROM {sql.broken_table} WHERE pass = ? AND host = ? ORDER BY task_id',
                (pass_num, hostname),
            )
        else:
            cur = db.execute(
                f'SELECT task_id FROM {sql.broken_table} WHERE pass = ? ORDER BY task_id',
                (pass_num,),
            )
        ids = [row[0] for row in cur.fetchall()]

    if not ids:
        raise ValueError(f'--fix: no entries in {sql.broken_table} for {fix_spec}')
    logger.info('Fix mode: %d task_id(s) for %s', len(ids), fix_spec)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--workdir', help='Project workdir containing config.rc')
    src.add_argument('--host', help='PostgreSQL host (use with --port/--user/--password)')

    parser.add_argument('--port', type=int, default=5432)
    parser.add_argument('--user')
    parser.add_argument('--password')

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--incremental', action='store_true',
                      help='Only recompute task_ids listed in scratch/retry_task_ids.txt (or retry_optune_ids.txt for op mode)')
    mode.add_argument('--fix', metavar='[HOSTNAME:]PASS',
                      help='Only recompute task_ids from broken_entries.db for the given pass')
    mode.add_argument('--ids', type=int, nargs='+', metavar='TASK_ID',
                      help='Debug: compute for specific task_id(s). Runs inline (no multiprocessing). Prints results; does NOT write to DB.')

    parser.add_argument('--tuning_mode', choices=['kernel', 'op'], default='kernel',
                        help='kernel: use tuning_results/best_tuning_results; op: use optune_results/best_optune_results')
    parser.add_argument('--verbose', action='store_true',
                        help='Debug: print per-candidate per-tensor accuracy details (implies DEBUG log level)')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    sql = SqlStatements(args.tuning_mode)

    if args.workdir:
        workdir = Path(args.workdir)
        conn_params = get_db_connection_params(workdir)
    else:
        workdir = None
        conn_params = {'host': args.host, 'port': args.port}
        if args.user:
            conn_params['user'] = args.user
        if args.password:
            conn_params['password'] = args.password

    # Resolve optional task_id filter
    filter_task_ids: list[int] | None = None
    if args.incremental:
        if workdir is None:
            parser.error('--incremental requires --workdir')
        filter_task_ids = _resolve_task_ids_incremental(workdir, sql)
    elif args.fix:
        if workdir is None:
            parser.error('--fix requires --workdir')
        filter_task_ids = _resolve_task_ids_fix(workdir, sql, args.fix)
    elif args.ids:
        filter_task_ids = args.ids
        logger.info('Debug mode: %d task_id(s): %s', len(filter_task_ids), filter_task_ids)

    incremental = filter_task_ids is not None

    t_total = time.monotonic()

    with psycopg.connect(**conn_params, autocommit=True) as conn:
        thresholds = load_thresholds(conn, sql, filter_task_ids)
        archs = get_archs(conn, sql)

    # --ids: run inline in the main process for debuggability; do not write to DB.
    if args.ids:
        all_results = []
        with psycopg.connect(**conn_params, autocommit=True) as conn:
            with conn.cursor() as cur:
                placeholders = ','.join(['%s'] * len(args.ids))
                cur.execute(f"""
                    SELECT tr.task_id, tr.{sql.key_col}, tr.{sql.index_col}, tr.result_data
                    FROM {sql.results_table} tr
                    JOIN task_queue tq ON tq.id = tr.task_id
                    WHERE tr.result_data IS NOT NULL
                      AND tr.task_id IN ({placeholders})
                    ORDER BY tr.task_id, tr.{sql.key_col}
                """, args.ids)
                rows = cur.fetchall()

        for (task_id, key_name), group in groupby(rows, key=lambda r: (r[0], r[1])):
            group_rows = []
            for r in group:
                index, rd = r[2], r[3]
                if isinstance(rd, str):
                    try:
                        rd = json.loads(rd)
                    except (json.JSONDecodeError, TypeError):
                        continue
                group_rows.append((index, rd))

            arch_thresholds = next(
                (at for at in thresholds.values() if task_id in at), {}
            )
            task_thresholds = arch_thresholds.get(task_id, {}).get(key_name, {})
            logger.info('--- task_id=%d key=%s: %d candidates, %d thresholds ---',
                        task_id, key_name, len(group_rows), len(task_thresholds))
            result = _find_best_candidate(task_id, key_name, group_rows, task_thresholds,
                                          tuning_mode=args.tuning_mode, verbose=args.verbose)
            if result is not None:
                _, _, index, median_time, impl_desc = result
                logger.info('  → best: index=%d median_time=%.3fms impl=%s',
                            index, median_time, impl_desc)
                all_results.append(result)
            else:
                logger.info('  → no best candidate found')

        logger.info('--ids: %d best results found (not written to DB)', len(all_results))
        logger.info('Total time: %.1fs', time.monotonic() - t_total)
        return

    # Build work items.
    # Incremental/fix mode: one worker per arch, passing filter_task_ids directly.
    # Full mode: split each arch's task_id range into WORKERS_PER_ARCH bands.
    work_items = []  # (arch, worker_index, band_lo, band_hi)
    for arch in archs:
        if arch not in thresholds:
            continue
        task_ids_for_arch = thresholds[arch].keys()
        lo, hi = min(task_ids_for_arch), max(task_ids_for_arch)
        if filter_task_ids is not None:
            # One worker handles all filter_task_ids for this arch via = ANY(%s).
            # band_lo/hi are passed but unused in the worker's incremental branch.
            work_items.append((arch, len(work_items), lo, hi))
        else:
            step = math.ceil((hi - lo + 1) / WORKERS_PER_ARCH)
            for chunk_idx in range(WORKERS_PER_ARCH):
                band_lo = lo + chunk_idx * step
                band_hi = min(lo + (chunk_idx + 1) * step - 1, hi)
                work_items.append((arch, len(work_items), band_lo, band_hi))

    workers_per_arch = 1 if filter_task_ids is not None else WORKERS_PER_ARCH
    logger.info('Step 2+3: spawning %d worker(s) (%d arch(s) × %d) — archs: %s',
                len(work_items), len(archs), workers_per_arch, archs)

    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=len(work_items)) as pool:
        futures = [
            pool.apply_async(
                worker_process_arch,
                (arch, worker_index, band_lo, band_hi, conn_params, thresholds[arch],
                 args.tuning_mode, args.verbose, filter_task_ids)
            )
            for arch, worker_index, band_lo, band_hi in work_items
        ]
        all_results = []
        for f in futures:
            all_results.extend(f.get())

    with psycopg.connect(**conn_params, autocommit=False) as conn:
        write_results(conn, sql, all_results, incremental=incremental)

    logger.info('Total time: %.1fs', time.monotonic() - t_total)


if __name__ == '__main__':
    main()
