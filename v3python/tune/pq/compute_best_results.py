#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Compute best_tuning_results table using Python multiprocessing.

For each (task_id, kernel_name): find the fastest hsaco_index that passes
the accuracy threshold — absolute_error <= 10x the minimum across all
(test_case, tensor_name) pairs in most_accurate_tuning_results.

reference_error IS NULL  → tensor genuinely inapplicable → pass
absolute_error IS NULL with real reference_error → kernel broken → fail

One worker process is spawned per arch (matching task_queue's arch partitions).
Each worker opens its own DB connection and streams only its arch's rows,
so no result_data is ever sent over IPC — only small result tuples are returned.

Usage:
    python -m v3python.tune.pq.compute_best_results --workdir /path/to/workdir
    python -m v3python.tune.pq.compute_best_results --host localhost --port 5432 --user myuser --password secret
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

ACCURACY_MULTIPLIER = 10.0
WORKERS_PER_ARCH = 2        # sub-workers per arch, sliced by task_id
STREAM_BATCH_SIZE = 10_000  # rows fetched per server-side cursor batch
INSERT_BATCH_SIZE = 1_000   # rows per bulk INSERT

# Test cases excluded from accuracy gating (known kernel bugs whose results
# cannot be used as a correctness criterion for hsaco selection).
SKIP_TEST_CASES: frozenset[str] = frozenset({
    '01_gqa',  # GQA bias indexing bug in bwd dk/dv kernels (fixed, pending re-tune)
})


# ---------------------------------------------------------------------------
# Per-group computation (called inline inside each worker process)
# ---------------------------------------------------------------------------

def _find_best_hsaco(task_id: int, kernel_name: str,
                     group_rows: list, thresholds: dict) -> tuple | None:
    """
    Find the fastest hsaco that passes the accuracy threshold.

    group_rows:  [(hsaco_index, result_data), ...]  (result_data is a dict, already parsed)
    thresholds:  {(test_case, tensor_name): min_absolute_error}

    Returns (task_id, arch, kernel_name, hsaco_index, median_time, impl_desc) or None.
    arch is filled in by the caller.
    """
    best = None

    for hsaco_index, rd in group_rows:
        times = rd.get('times')
        if not times:
            continue
        median_time = times[0]
        impl_desc = rd.get('impl_desc')

        passes = True
        for tc, tensors in rd.get('adiffs', {}).items():
            if tc in SKIP_TEST_CASES:
                continue
            for tname, vals in tensors.items():
                if not vals:
                    continue  # JSON null or empty array → inapplicable
                ref_err = vals[2] if len(vals) > 2 else None
                if ref_err is None:
                    continue  # tensor genuinely inapplicable → pass
                abs_err = vals[1] if len(vals) > 1 else None
                if abs_err is None:
                    passes = False  # kernel broken (NaN output) → fail
                    break
                min_err = thresholds.get((tc, tname))
                if min_err is not None and abs_err > ACCURACY_MULTIPLIER * min_err:
                    passes = False
                    break
            if not passes:
                break

        if passes and (best is None or median_time < best[1]):
            best = (hsaco_index, median_time, impl_desc)

    if best is None:
        return None
    hsaco_index, median_time, impl_desc = best
    return (task_id, kernel_name, hsaco_index, median_time, impl_desc)


# ---------------------------------------------------------------------------
# Step 1: load most_accurate_tuning_results, keyed by arch
# ---------------------------------------------------------------------------

def load_thresholds(conn) -> dict:
    """
    Returns:
        {arch: {task_id: {kernel_name: {(test_case, tensor_name): min_absolute_error}}}}

    Keyed by arch so each worker process receives only its own slice.
    """
    logger.info('Step 1: loading most_accurate_tuning_results into RAM...')
    t0 = time.monotonic()

    thresholds: dict = {}
    with conn.cursor() as cur:
        cur.execute("""
            SELECT arch, task_id, kernel_name, test_case, tensor_name, absolute_error
            FROM most_accurate_tuning_results
        """)
        for arch, task_id, kernel_name, test_case, tensor_name, abs_err in cur:
            tk = thresholds.setdefault(arch, {}).setdefault(task_id, {}).setdefault(kernel_name, {})
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


def get_archs(conn) -> list[str]:
    with conn.cursor() as cur:
        cur.execute('SELECT DISTINCT arch FROM most_accurate_tuning_results ORDER BY arch')
        return [row[0] for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Step 2+3: worker — one per arch, owns its own DB connection
# ---------------------------------------------------------------------------

def worker_process_arch(arch: str, worker_index: int,
                        task_id_lo: int, task_id_hi: int,
                        conn_params: dict, arch_thresholds: dict) -> list:
    """
    Runs in a child process. Streams tuning_results for one (arch, task_id range),
    processes each (task_id, kernel_name) group inline, returns best rows.

    task_id_lo/hi:   inclusive task_id range this worker is responsible for
    arch_thresholds: {task_id: {kernel_name: {(test_case, tensor_name): min_absolute_error}}}
                     full arch slice; worker only accesses task_ids in [lo, hi]

    Returns list of (task_id, kernel_name, hsaco_index, median_time, impl_desc).
    """
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

    # Count groups in this range from thresholds (new tasks added while tuning
    # is active may push group_count above this estimate).
    total_groups = sum(len(kn_dict) for tid, kn_dict in arch_thresholds.items()
                       if task_id_lo <= tid <= task_id_hi)
    log.info('arch=%s chunk=%d [%d, %d]: ~%d groups to process',
             arch, worker_index, task_id_lo, task_id_hi, total_groups)

    with psycopg.connect(**conn_params, autocommit=True) as conn:
        with conn.transaction(), conn.cursor(name=f'worker_{arch}_{worker_index}') as cur:
            cur.itersize = STREAM_BATCH_SIZE
            # Partition pruning on arch + index seek on task_id.
            cur.execute("""
                SELECT tr.task_id, tr.kernel_name, tr.hsaco_index, tr.result_data
                FROM tuning_results tr
                JOIN task_queue tq ON tq.id = tr.task_id AND tq.arch = %s
                WHERE tr.result_data IS NOT NULL
                  AND tr.task_id BETWEEN %s AND %s
                ORDER BY tr.task_id, tr.kernel_name
            """, (arch, task_id_lo, task_id_hi))

            for (task_id, kernel_name), rows in groupby(cur, key=lambda r: (r[0], r[1])):
                # Parse JSONB here in the worker — never crosses IPC boundary.
                group_rows = []
                for r in rows:
                    hsaco_index, rd = r[2], r[3]
                    if isinstance(rd, str):
                        try:
                            rd = json.loads(rd)
                        except (json.JSONDecodeError, TypeError):
                            continue
                    group_rows.append((hsaco_index, rd))

                row_count += len(group_rows)
                group_count += 1

                task_thresholds = arch_thresholds.get(task_id, {}).get(kernel_name, {})
                result = _find_best_hsaco(task_id, kernel_name, group_rows, task_thresholds)
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
# Step 4: write results to best_tuning_results table
# ---------------------------------------------------------------------------

def write_results(conn, arch_results: list) -> None:
    """
    Truncates best_tuning_results and bulk-inserts all rows.
    Fetches task_config from task_queue for each task_id.
    arch is already known from worker results.
    """
    logger.info('Step 4: writing %d rows to best_tuning_results...', len(arch_results))
    t0 = time.monotonic()

    task_ids = list({r[0] for r in arch_results})
    task_config_map: dict = {}
    with conn.cursor() as cur:
        for i in range(0, len(task_ids), 1000):
            batch = task_ids[i:i + 1000]
            cur.execute('SELECT id, task_config FROM task_queue WHERE id = ANY(%s)', (batch,))
            for row in cur.fetchall():
                task_config_map[row[0]] = row[1]

    # Workers return (task_id, kernel_name, hsaco_index, median_time, impl_desc).
    # Fetch arch from most_accurate_tuning_results indirectly via task_queue.
    # Re-query arch per task_id to keep write_results self-contained.
    arch_map: dict = {}
    with conn.cursor() as cur:
        for i in range(0, len(task_ids), 1000):
            batch = task_ids[i:i + 1000]
            cur.execute('SELECT id, arch FROM task_queue WHERE id = ANY(%s)', (batch,))
            for row in cur.fetchall():
                arch_map[row[0]] = row[1]

    rows_to_insert = []
    for task_id, kernel_name, hsaco_index, median_time, impl_desc in arch_results:
        task_config = task_config_map.get(task_id)
        arch = arch_map.get(task_id)
        if task_config is None or arch is None:
            continue
        rows_to_insert.append((task_id, arch, Jsonb(task_config), kernel_name,
                               hsaco_index, median_time,
                               Jsonb(impl_desc) if impl_desc is not None else None))

    with conn.cursor() as cur:
        cur.execute('TRUNCATE TABLE best_tuning_results')
        for i in range(0, len(rows_to_insert), INSERT_BATCH_SIZE):
            batch = rows_to_insert[i:i + INSERT_BATCH_SIZE]
            cur.executemany("""
                INSERT INTO best_tuning_results
                    (task_id, arch, task_config, kernel_name,
                     hsaco_index, median_time, impl_desc)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (task_id, kernel_name) DO UPDATE
                    SET hsaco_index = EXCLUDED.hsaco_index,
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

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--workdir', help='Project workdir containing config.rc')
    src.add_argument('--host', help='PostgreSQL host (use with --port/--user/--password)')

    parser.add_argument('--port', type=int, default=5432)
    parser.add_argument('--user')
    parser.add_argument('--password')
    args = parser.parse_args()

    if args.workdir:
        conn_params = get_db_connection_params(Path(args.workdir))
    else:
        conn_params = {'host': args.host, 'port': args.port}
        if args.user:
            conn_params['user'] = args.user
        if args.password:
            conn_params['password'] = args.password

    t_total = time.monotonic()

    with psycopg.connect(**conn_params, autocommit=True) as conn:
        thresholds = load_thresholds(conn)
        archs = get_archs(conn)

    # Build work items: split each arch's task_id range into WORKERS_PER_ARCH bands.
    work_items = []  # (arch, worker_index, task_id_lo, task_id_hi)
    for arch in archs:
        task_ids = thresholds[arch].keys()
        lo, hi = min(task_ids), max(task_ids)
        step = math.ceil((hi - lo + 1) / WORKERS_PER_ARCH)
        for chunk_idx in range(WORKERS_PER_ARCH):
            band_lo = lo + chunk_idx * step
            band_hi = min(lo + (chunk_idx + 1) * step - 1, hi)
            work_items.append((arch, len(work_items), band_lo, band_hi))

    logger.info('Step 2+3: spawning %d worker(s) (%d arch(s) × %d) — archs: %s',
                len(work_items), len(archs), WORKERS_PER_ARCH, archs)

    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=len(work_items)) as pool:
        futures = [
            pool.apply_async(
                worker_process_arch,
                (arch, worker_index, band_lo, band_hi, conn_params, thresholds[arch])
            )
            for arch, worker_index, band_lo, band_hi in work_items
        ]
        all_results = []
        for f in futures:
            all_results.extend(f.get())

    with psycopg.connect(**conn_params, autocommit=False) as conn:
        write_results(conn, all_results)

    logger.info('Total time: %.1fs', time.monotonic() - t_total)


if __name__ == '__main__':
    main()
