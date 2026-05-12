#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Export best_tuning_results from PostgreSQL to a centralized SQLite database.

The output file matches the schema of v3python/database/tuning_database.sqlite3
and can be fed directly into database_decompose.py to produce per-arch sharded
SQLite files (and their .tar.xz archives) for the build-time tuning database.

Typical workflow:
    # 1. Export from PostgreSQL
    .tune/bin/export_best_results ~/wkdir.v3.5 /tmp/tuning_database.sqlite3

    # 2. Decompose into per-arch shards
    python3 -m v3python.database_decompose \\
        --database_file /tmp/tuning_database.sqlite3 \\
        --script_output /tmp/decompose.sh \\
        --decompose_output /tmp/decomposed
    bash /tmp/decompose.sh

    # 3. Copy shards into the source tree
    cp -r /tmp/decomposed/amd/ v3python/database/

Column mapping
--------------
gpu                  : "{arch}_mod0"  (canonical GPU identifier)
inputs$Q_dtype       : "torch.{entry.dtype}"
inputs$Max_seqlen_q  : entry.seqlen_q          (attn_fwd capitalises Max)
inputs$max_seqlen_q  : entry.seqlen_q          (bwd kernels use lowercase)
inputs$CAUSAL_TYPE   : 0 (NONE) or 3 (WINDOWED) derived from entry.causal
inputs$BLOCK_DMODEL  : entry.hdim
inputs$ENABLE_DROPOUT: 1 if entry.dropout_p > 0 else 0
inputs$PADDED_HEAD   : impl_desc.psels["PADDED_HEAD"] (0 for POT hdims)
inputs$BIAS_TYPE     : entry.bias_type
inputs$USE_ALIBI     : impl_desc.psels["USE_ALIBI"]   (0 in standard flash)
inputs$INT8          : impl_desc.psels["INT8"]         (0 in standard flash)
inputs$INT8_KV       : impl_desc.psels["INT8_KV"]      (0 in standard flash)
inputs$USE_P_SCALE   : impl_desc.psels["USE_P_SCALE"]  (0 in standard flash)
tuned_kernel$*       : impl_desc.psels[<column>]
compiler_options$*   : impl_desc.copts[<column>]

Usage:
    python -m v3python.tune.pq.export_best_results \\
        --workdir /path/to/workdir --output /path/to/tuning_database.sqlite3
    python -m v3python.tune.pq.export_best_results \\
        --host localhost --output /path/to/tuning_database.sqlite3
"""

import argparse
import logging
import sqlite3
import time
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from ..utils import get_db_connection_params

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQLite schema definitions (must match v3python/database/tuning_database.sqlite3)
# ---------------------------------------------------------------------------

# Column lists per kernel table.
# Tuples: (sqlite_col, source, key)
#   source = 'gpu'     → constant "{arch}_mod0"
#   source = 'entry'   → task_config['entry'][key]
#   source = 'psels'   → impl_desc['psels'][key]
#   source = 'copts'   → impl_desc['copts'][key]
#   source = 'derived' → special derivation (handled below)
# Note: 'id INTEGER PRIMARY KEY' is prepended automatically.

_FWD_COLS = [
    # (sqlite_col,                    sql_type,  source,    key)
    ('gpu',                          'TEXT',    'gpu',     None),
    ('inputs$Q_dtype',               'TEXT',    'derived', 'dtype'),
    ('inputs$Max_seqlen_q',          'INTEGER', 'entry',   'seqlen_q'),
    ('inputs$Max_seqlen_k',          'INTEGER', 'entry',   'seqlen_k'),
    ('inputs$CAUSAL_TYPE',           'INTEGER', 'derived', 'causal'),
    ('inputs$BLOCK_DMODEL',          'INTEGER', 'entry',   'hdim'),
    ('inputs$ENABLE_DROPOUT',        'INTEGER', 'derived', 'dropout_p'),
    ('inputs$PADDED_HEAD',           'INTEGER', 'psels',   'PADDED_HEAD'),
    ('inputs$BIAS_TYPE',             'INTEGER', 'entry',   'bias_type'),
    ('inputs$USE_ALIBI',             'INTEGER', 'psels',   'USE_ALIBI'),
    ('inputs$INT8',                  'INTEGER', 'psels',   'INT8'),
    ('inputs$INT8_KV',               'INTEGER', 'psels',   'INT8_KV'),
    ('inputs$USE_P_SCALE',           'INTEGER', 'psels',   'USE_P_SCALE'),
    ('tuned_kernel$PERSISTENT_TYPE', 'INTEGER', 'psels',   'PERSISTENT_TYPE'),
    ('tuned_kernel$GRID_CU_MULTIP',  'INTEGER', 'psels',   'GRID_CU_MULTIP'),
    ('tuned_kernel$BLOCK_M',         'INTEGER', 'psels',   'BLOCK_M'),
    ('tuned_kernel$BLOCK_N',         'INTEGER', 'psels',   'BLOCK_N'),
    ('tuned_kernel$PRE_LOAD_V',      'INTEGER', 'psels',   'PRE_LOAD_V'),
    ('compiler_options$waves_per_eu','INTEGER', 'copts',   'waves_per_eu'),
    ('compiler_options$num_warps',   'INTEGER', 'copts',   'num_warps'),
    ('compiler_options$num_stages',  'INTEGER', 'copts',   'num_stages'),
]

_FWD_UNIQUE = [
    'gpu', 'inputs$Q_dtype', 'inputs$Max_seqlen_q', 'inputs$Max_seqlen_k',
    'inputs$CAUSAL_TYPE', 'inputs$BLOCK_DMODEL', 'inputs$ENABLE_DROPOUT',
    'inputs$PADDED_HEAD', 'inputs$BIAS_TYPE', 'inputs$USE_ALIBI',
    'inputs$INT8', 'inputs$INT8_KV', 'inputs$USE_P_SCALE',
]

# bwd_kernel_dk_dv and bwd_kernel_dq share the same schema.
# Note lowercase max_seqlen vs. attn_fwd's Max_seqlen.
_BWD_SPLIT_COLS = [
    # (sqlite_col,                    sql_type,  source,    key)
    ('gpu',                          'TEXT',    'gpu',     None),
    ('inputs$Q_dtype',               'TEXT',    'derived', 'dtype'),
    ('inputs$max_seqlen_q',          'INTEGER', 'entry',   'seqlen_q'),
    ('inputs$max_seqlen_k',          'INTEGER', 'entry',   'seqlen_k'),
    ('inputs$CAUSAL_TYPE',           'INTEGER', 'derived', 'causal'),
    ('inputs$BLOCK_DMODEL',          'INTEGER', 'entry',   'hdim'),
    ('inputs$ENABLE_DROPOUT',        'INTEGER', 'derived', 'dropout_p'),
    ('inputs$PADDED_HEAD',           'INTEGER', 'psels',   'PADDED_HEAD'),
    ('inputs$BIAS_TYPE',             'INTEGER', 'entry',   'bias_type'),
    ('tuned_kernel$BLOCK_M',         'INTEGER', 'psels',   'BLOCK_M'),
    ('tuned_kernel$BLOCK_N',         'INTEGER', 'psels',   'BLOCK_N'),
    ('compiler_options$waves_per_eu','INTEGER', 'copts',   'waves_per_eu'),
    ('compiler_options$num_warps',   'INTEGER', 'copts',   'num_warps'),
    ('compiler_options$num_stages',  'INTEGER', 'copts',   'num_stages'),
]

_BWD_SPLIT_UNIQUE = [
    'gpu', 'inputs$Q_dtype', 'inputs$max_seqlen_q', 'inputs$max_seqlen_k',
    'inputs$CAUSAL_TYPE', 'inputs$BLOCK_DMODEL', 'inputs$ENABLE_DROPOUT',
    'inputs$PADDED_HEAD', 'inputs$BIAS_TYPE',
]

_BWD_FUSE_COLS = [
    # (sqlite_col,                    sql_type,  source,    key)
    ('gpu',                          'TEXT',    'gpu',     None),
    ('inputs$Q_dtype',               'TEXT',    'derived', 'dtype'),
    ('inputs$max_seqlen_q',          'INTEGER', 'entry',   'seqlen_q'),
    ('inputs$max_seqlen_k',          'INTEGER', 'entry',   'seqlen_k'),
    ('inputs$CAUSAL_TYPE',           'INTEGER', 'derived', 'causal'),
    ('inputs$BLOCK_DMODEL',          'INTEGER', 'entry',   'hdim'),
    ('inputs$ENABLE_DROPOUT',        'INTEGER', 'derived', 'dropout_p'),
    ('inputs$PADDED_HEAD',           'INTEGER', 'psels',   'PADDED_HEAD'),
    ('inputs$BIAS_TYPE',             'INTEGER', 'entry',   'bias_type'),
    ('inputs$USE_ALIBI',             'INTEGER', 'psels',   'USE_ALIBI'),
    ('inputs$INT8',                  'INTEGER', 'psels',   'INT8'),
    ('inputs$INT8_KV',               'INTEGER', 'psels',   'INT8_KV'),
    ('inputs$USE_P_SCALE',           'INTEGER', 'psels',   'USE_P_SCALE'),
    ('tuned_kernel$BLOCK_M',         'INTEGER', 'psels',   'BLOCK_M'),
    ('tuned_kernel$BLOCK_N',         'INTEGER', 'psels',   'BLOCK_N'),
    ('compiler_options$waves_per_eu','INTEGER', 'copts',   'waves_per_eu'),
    ('compiler_options$num_warps',   'INTEGER', 'copts',   'num_warps'),
    ('compiler_options$num_stages',  'INTEGER', 'copts',   'num_stages'),
]

_BWD_FUSE_UNIQUE = [
    'gpu', 'inputs$Q_dtype', 'inputs$max_seqlen_q', 'inputs$max_seqlen_k',
    'inputs$CAUSAL_TYPE', 'inputs$BLOCK_DMODEL', 'inputs$ENABLE_DROPOUT',
    'inputs$PADDED_HEAD', 'inputs$BIAS_TYPE', 'inputs$USE_ALIBI',
    'inputs$INT8', 'inputs$INT8_KV', 'inputs$USE_P_SCALE',
]

KERNEL_SCHEMAS = {
    'attn_fwd':         (_FWD_COLS,        _FWD_UNIQUE),
    'bwd_kernel_dk_dv': (_BWD_SPLIT_COLS,  _BWD_SPLIT_UNIQUE),
    'bwd_kernel_dq':    (_BWD_SPLIT_COLS,  _BWD_SPLIT_UNIQUE),
    'bwd_kernel_fuse':  (_BWD_FUSE_COLS,   _BWD_FUSE_UNIQUE),
}

# Op-mode schemas — same input columns as the kernel tables (minus tuned_kernel$/
# compiler_options$ columns) plus op$backend and op$tflops.
_OP_EXTRA_COLS = [
    ('op$backend', 'INTEGER', 'op_backend', None),
    ('op$tflops',  'REAL',    'op_tflops',  None),
]

def _input_only(cols: list) -> list:
    return [c for c in cols
            if not c[0].startswith(('tuned_kernel$', 'compiler_options$'))]

OP_SCHEMAS = {
    'op_attn_fwd': (_input_only(_FWD_COLS)      + _OP_EXTRA_COLS, _FWD_UNIQUE),
    'op_attn_bwd': (_input_only(_BWD_FUSE_COLS) + _OP_EXTRA_COLS, _BWD_FUSE_UNIQUE),
}

# ---------------------------------------------------------------------------
# Value derivations
# ---------------------------------------------------------------------------

DTYPE_MAP = {
    'float16':  'torch.float16',
    'bfloat16': 'torch.bfloat16',
    'float32':  'torch.float32',
}


def derive_dtype(entry: dict) -> str:
    raw = entry['dtype']
    mapped = DTYPE_MAP.get(raw)
    if mapped is None:
        raise ValueError(
            f'Unknown dtype {raw!r}; expected one of {list(DTYPE_MAP)}'
        )
    return mapped


def derive_causal_type(entry: dict) -> int:
    causal = entry['causal']
    if causal is False or causal == 0:
        return 0   # CausalType.NONE
    return 3       # CausalType.WINDOWED (covers causal=True and tuple windows)


def derive_enable_dropout(entry: dict) -> int:
    return 1 if entry.get('dropout_p', 0.0) > 0.0 else 0


def extract_value(col_def: tuple, gpu: str, entry: dict,
                  psels: dict, copts: dict):
    _, _sql_type, source, key = col_def
    if source == 'gpu':
        return gpu
    if source == 'entry':
        return entry[key]
    if source == 'psels':
        return psels.get(key, 0)   # absent → fixed at 0 for this kernel variant
    if source == 'copts':
        return copts.get(key, 0)
    if source == 'derived':
        if key == 'dtype':
            return derive_dtype(entry)
        if key == 'causal':
            return derive_causal_type(entry)
        if key == 'dropout_p':
            return derive_enable_dropout(entry)
    raise ValueError(f'Unknown source/key: {source!r}/{key!r}')


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _col_names(cols: list) -> list[str]:
    return [c[0] for c in cols]


def ensure_table(db: sqlite3.Connection, kernel: str,
                 cols: list, unique: list) -> None:
    """Create table with id INTEGER PRIMARY KEY (matching central DB format)."""
    table = f'FLASH${kernel}'
    col_defs = ', '.join(
        f'"{col}" {sql_type}'
        for col, sql_type, _source, _key in cols
    )
    unique_cols = ', '.join(f'"{c}"' for c in unique)
    unique_clause = f', UNIQUE({unique_cols})'
    db.execute(
        f'CREATE TABLE IF NOT EXISTS "{table}" '
        f'(id INTEGER PRIMARY KEY, {col_defs}{unique_clause})'
    )
    db.commit()


def insert_row(db: sqlite3.Connection, kernel: str,
               cols: list, values: list) -> None:
    table = f'FLASH${kernel}'
    col_str = ', '.join(f'"{c}"' for c in _col_names(cols))
    placeholders = ', '.join('?' * len(values))
    db.execute(
        f'INSERT OR REPLACE INTO "{table}" ({col_str}) VALUES ({placeholders})',
        values,
    )


# ---------------------------------------------------------------------------
# Main export logic
# ---------------------------------------------------------------------------

def export(conn_params: dict, output_path: Path) -> None:
    t0 = time.monotonic()

    logger.info('Querying best_tuning_results...')
    with psycopg.connect(**conn_params, autocommit=True,
                         row_factory=dict_row) as pg:
        with pg.cursor() as cur:
            cur.execute("""
                SELECT b.task_id, b.arch, b.kernel_name, b.task_config, b.impl_desc
                FROM best_tuning_results b
                JOIN task_queue t ON t.id = b.task_id AND t.arch = b.arch
                WHERE t.status != 'cancelled'
                ORDER BY b.arch, b.kernel_name
            """)
            rows = cur.fetchall()

    logger.info('Fetched %d rows from best_tuning_results', len(rows))

    counts: dict[str, int] = {}
    skipped = 0

    with sqlite3.connect(output_path) as db:
        # Ensure all tables exist up front.
        for kernel, (cols, unique) in KERNEL_SCHEMAS.items():
            ensure_table(db, kernel, cols, unique)

        for row in rows:
            task_id     = row['task_id']
            arch        = row['arch']
            kernel_name = row['kernel_name']
            task_config = row['task_config']
            impl_desc   = row['impl_desc']

            if impl_desc is None:
                logger.warning(
                    'Skipping task_id=%s arch=%s kernel=%s: impl_desc is NULL '
                    '(compute_best_results may not have run for this entry)',
                    task_id, arch, kernel_name,
                )
                skipped += 1
                continue

            if kernel_name not in KERNEL_SCHEMAS:
                logger.warning('Skipping unknown kernel %s (task_id=%s)', kernel_name, task_id)
                skipped += 1
                continue

            entry  = task_config['entry']
            psels  = impl_desc.get('psels') or {}
            copts  = impl_desc.get('copts') or {}
            gpu    = f'{arch}_mod0'

            cols, unique = KERNEL_SCHEMAS[kernel_name]

            try:
                values = [
                    extract_value(col_def, gpu, entry, psels, copts)
                    for col_def in cols
                ]
            except Exception as exc:
                logger.warning(
                    'Skipping task_id=%s arch=%s kernel=%s entry=%s: %s',
                    task_id, arch, kernel_name, entry, exc,
                )
                skipped += 1
                continue

            insert_row(db, kernel_name, cols, values)
            counts[kernel_name] = counts.get(kernel_name, 0) + 1

        db.commit()

    for kernel_name, n in sorted(counts.items()):
        logger.info('  %-20s: %d rows', kernel_name, n)

    total = sum(counts.values())
    logger.info(
        'Done: %d rows exported to %s, %d skipped in %.1fs',
        total, output_path, skipped, time.monotonic() - t0,
    )
    logger.info('Next step: .tune/bin/sancheck <workdir>')


def export_op(conn_params: dict, output_path: Path) -> None:
    t0 = time.monotonic()

    logger.info('Querying best_optune_results...')
    with psycopg.connect(**conn_params, autocommit=True,
                         row_factory=dict_row) as pg:
        with pg.cursor() as cur:
            cur.execute("""
                SELECT b.task_id, b.arch, b.op_name, b.task_config,
                       b.impl_desc, b.backend_index
                FROM best_optune_results b
                JOIN task_queue t ON t.id = b.task_id AND t.arch = b.arch
                WHERE t.status != 'cancelled'
                ORDER BY b.arch, b.op_name
            """)
            rows = cur.fetchall()

    logger.info('Fetched %d rows from best_optune_results', len(rows))

    counts: dict[str, int] = {}
    skipped = 0

    with sqlite3.connect(output_path) as db:
        for op_name, (cols, unique) in OP_SCHEMAS.items():
            ensure_table(db, op_name, cols, unique)

        for row in rows:
            task_id       = row['task_id']
            arch          = row['arch']
            op_name       = row['op_name']
            task_config   = row['task_config']
            backend_index = row['backend_index']

            if op_name not in OP_SCHEMAS:
                logger.warning('Skipping unknown op %s (task_id=%s)', op_name, task_id)
                skipped += 1
                continue

            cols, _ = OP_SCHEMAS[op_name]
            entry = task_config['entry']
            gpu   = f'{arch}_mod0'

            try:
                values = []
                for col_def in cols:
                    source = col_def[2]
                    if source == 'op_backend':
                        values.append(backend_index)
                    elif source == 'op_tflops':
                        values.append(0.0)
                    else:
                        values.append(extract_value(col_def, gpu, entry, {}, {}))
            except Exception as exc:
                logger.warning(
                    'Skipping task_id=%s arch=%s op=%s entry=%s: %s',
                    task_id, arch, op_name, entry, exc,
                )
                skipped += 1
                continue

            insert_row(db, op_name, cols, values)
            counts[op_name] = counts.get(op_name, 0) + 1

        db.commit()

    for op_name, n in sorted(counts.items()):
        logger.info('  %-20s: %d rows', op_name, n)

    total = sum(counts.values())
    logger.info(
        'Done: %d rows exported to %s, %d skipped in %.1fs',
        total, output_path, skipped, time.monotonic() - t0,
    )
    logger.info('Next step: .tune/bin/sancheck <workdir> --tuning_mode op')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--workdir',
                     help='Project workdir containing config.rc')
    src.add_argument('--host',
                     help='PostgreSQL host (use with --port/--user/--password)')

    parser.add_argument('--port', type=int, default=5432)
    parser.add_argument('--user')
    parser.add_argument('--password')
    parser.add_argument('--output', required=True, type=Path,
                        help='Output SQLite file path (e.g. tuning_database.sqlite3)')
    parser.add_argument('--tuning_mode', choices=['kernel', 'op'], default='kernel',
                        help='kernel: export best_tuning_results; op: export best_optune_results')
    args = parser.parse_args()

    if args.workdir:
        conn_params = get_db_connection_params(Path(args.workdir))
    else:
        conn_params = {'host': args.host, 'port': args.port}
        if args.user:
            conn_params['user'] = args.user
        if args.password:
            conn_params['password'] = args.password

    if args.tuning_mode == 'op':
        export_op(conn_params, args.output)
    else:
        export(conn_params, args.output)


if __name__ == '__main__':
    main()
