# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Database query functions for performance visualization.

All functions accept a psycopg connection object (caller manages lifecycle).
Returns dicts ready for JSON serialization by the webui or export script.
"""

from psycopg.rows import dict_row

from .vis_descriptors import DESCRIPTORS


def _build_query(desc: dict, arch: str, kernel_or_op: str, mode: str,
                 seqlen_min: int, seqlen_max: int) -> tuple[str, list]:
    """Build the SELECT query for a single kernel/op using the descriptor.

    Joins tuning_results (kernel mode) or optune_results (op mode) to retrieve
    BATCH and N_HEADS from result_data->'bim', so the TFLOPS formula accounts
    for the actual benchmark dimensions.
    """
    if mode == 'op':
        table = desc['op_table']
        name_col = desc['op_name_col']
        dim_selects = ',\n    '.join(f'{expr} AS {alias}' for expr, alias in desc['dims'])
        dim_groups  = ', '.join(alias for _, alias in desc['dims'])
        # Join optune_results on the winning backend_index to get bim BATCH/N_HEADS.
        sql = f"""
            SELECT
                {dim_selects},
                b.median_time AS median_ms,
                b.task_id     AS task_id,
                (r.result_data->'bim'->>'BATCH')::int AS batch,
                CASE
                    WHEN jsonb_typeof(r.result_data->'bim'->'N_HEADS') = 'array'
                    THEN (r.result_data->'bim'->'N_HEADS'->0)::int
                    ELSE (r.result_data->'bim'->>'N_HEADS')::int
                END AS n_heads
            FROM {table} b
            JOIN optune_results r
              ON r.task_id = b.task_id
             AND r.{name_col} = b.{name_col}
             AND r.backend_index = b.backend_index
            WHERE b.arch = %s
              AND b.{name_col} = %s
              AND (b.task_config->'entry'->>'seqlen_q')::int >= %s
              AND (b.task_config->'entry'->>'seqlen_q')::int <= %s
              AND (b.task_config->'entry'->>'seqlen_k')::int >= %s
              AND (b.task_config->'entry'->>'seqlen_k')::int <= %s
            ORDER BY {dim_groups}
        """
        params = [arch, kernel_or_op, seqlen_min, seqlen_max, seqlen_min, seqlen_max]
        return sql, params

    table = desc['kernel_table']
    name_col = desc['kernel_name_col']

    # Qualify column references with table alias to avoid ambiguity after JOIN.
    dim_selects = ',\n    '.join(f'{expr} AS {alias}' for expr, alias in desc['dims'])
    dim_groups  = ', '.join(alias for _, alias in desc['dims'])

    # Join tuning_results on the exact winning row to get bim BATCH/N_HEADS.
    # N_HEADS may be a JSON array (GQA); take the first element in that case.
    sql = f"""
        SELECT
            {dim_selects},
            b.median_time AS median_ms,
            b.task_id     AS task_id,
            (r.result_data->'bim'->>'BATCH')::int AS batch,
            CASE
                WHEN jsonb_typeof(r.result_data->'bim'->'N_HEADS') = 'array'
                THEN (r.result_data->'bim'->'N_HEADS'->0)::int
                ELSE (r.result_data->'bim'->>'N_HEADS')::int
            END AS n_heads
        FROM {table} b
        JOIN tuning_results r
          ON r.task_id = b.task_id
         AND r.kernel_name = b.kernel_name
         AND r.hsaco_index = b.hsaco_index
        WHERE b.arch = %s
          AND b.{name_col} = %s
          AND (b.task_config->'entry'->>'seqlen_q')::int >= %s
          AND (b.task_config->'entry'->>'seqlen_q')::int <= %s
          AND (b.task_config->'entry'->>'seqlen_k')::int >= %s
          AND (b.task_config->'entry'->>'seqlen_k')::int <= %s
        ORDER BY {dim_groups}
    """
    params = [arch, kernel_or_op, seqlen_min, seqlen_max, seqlen_min, seqlen_max]
    return sql, params


def _build_axes(rows: list[dict], desc: dict) -> dict:
    """Compute sorted unique values for each dimension from the result rows."""
    axes: dict[str, list] = {}
    for _, alias in desc['dims']:
        vals = sorted({r.get(alias) for r in rows if r.get(alias) is not None})
        axes[alias] = vals
    return axes


def query_best_results(conn, arch: str, kernel: str, mode: str = 'kernel',
                       seqlen_min: int = 0, seqlen_max: int = 65536,
                       descriptor_id: str = 'flash') -> dict:
    """
    Query best tuning results for one arch+kernel (or op) combination.

    Returns:
        {
          'arch': str,
          'kernel': str,
          'axes': {dim: [sorted unique values], ...},
          'rows': [{dim: value, ..., 'median_ms': float}, ...]
        }
    """
    desc = DESCRIPTORS[descriptor_id]
    sql, params = _build_query(desc, arch, kernel, mode, seqlen_min, seqlen_max)

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    rows = [dict(r) for r in rows]
    axes = _build_axes(rows, desc)

    return {
        'arch': arch,
        'kernel': kernel,
        'axes': axes,
        'rows': rows,
    }


def query_all_best_results(conn, descriptor_id: str = 'flash') -> dict:
    """
    Query best tuning results for ALL arches and kernels (for static export).

    Returns:
        {
          arch: {
            kernel: {arch, kernel, axes, rows}
          }
        }
    """
    desc = DESCRIPTORS[descriptor_id]

    # Enumerate all arches. Union across kernel_table and op_table so
    # descriptors that expose only ops (no kernel_table) still appear.
    arch_tables = [t for t in (desc.get('kernel_table'), desc.get('op_table')) if t]
    archs: list[str] = []
    seen: set[str] = set()
    with conn.cursor() as cur:
        for tbl in arch_tables:
            cur.execute(f"SELECT DISTINCT arch FROM {tbl} ORDER BY arch")
            for (a,) in cur.fetchall():
                if a not in seen:
                    seen.add(a)
                    archs.append(a)

    result: dict[str, dict[str, dict]] = {}
    for arch in archs:
        result[arch] = {}
        for kernel in desc['kernels']:
            data = query_best_results(conn, arch, kernel, mode='kernel',
                                      descriptor_id=descriptor_id)
            if data['rows']:
                result[arch][kernel] = data

        for op in desc['ops']:
            data = query_best_results(conn, arch, op, mode='op',
                                      descriptor_id=descriptor_id)
            if data['rows']:
                result[arch][op] = data

    return result


def query_cell_detail(conn, task_id: int, kernel: str, mode: str = 'kernel') -> dict:
    """
    Fetch all candidate tuning_results / optune_results rows for a single
    (task_id, kernel) cell, plus the per-(test_case, tensor_name) accuracy
    threshold from most_accurate_tuning_results / most_accurate_optune_results.

    Returns:
        {
          'task_id': int,
          'kernel': str,
          'mode': str,                # 'kernel' | 'op'
          'candidates': [
            {
              'index': int,           # hsaco_index or backend_index
              'median_ms': float|None,
              'times': [float, ...],
              'psels': {key: val, ...},
              'copts': {key: val, ...},
              'adiffs': {test_case: {tensor: [target, abs_err, ref_err], ...}, ...},
              'result': str,          # OK/NotOK/crash/ERROR
            },
            ...
          ],
          'thresholds': [
            {'test_case': str, 'tensor': str, 'absolute_error': float},
            ...
          ],
        }
    """
    assert mode in ('kernel', 'op'), f"query_cell_detail: invalid mode {mode!r}"
    if mode == 'op':
        results_table = 'optune_results'
        accuracy_table = 'most_accurate_optune_results'
        name_col = 'op_name'
        idx_col = 'backend_index'
    else:
        results_table = 'tuning_results'
        accuracy_table = 'most_accurate_tuning_results'
        name_col = 'kernel_name'
        idx_col = 'hsaco_index'

    candidates: list[dict] = []
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT {idx_col}, result, result_data
              FROM {results_table}
             WHERE task_id = %s AND {name_col} = %s
             ORDER BY {idx_col}
        """, (task_id, kernel))
        for index, result, rd in cur:
            rd = rd or {}
            impl = rd.get('impl_desc') or {}
            times = rd.get('times') or []
            candidates.append({
                'index': index,
                'median_ms': float(times[0]) if times else None,
                'times': [float(t) for t in times],
                'psels': impl.get('psels') or {},
                'copts': impl.get('copts') or {},
                'adiffs': rd.get('adiffs') or {},
                'result': result,
            })

    thresholds: list[dict] = []
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT test_case, tensor_name, absolute_error
              FROM {accuracy_table}
             WHERE task_id = %s AND {name_col} = %s
        """, (task_id, kernel))
        for tc, tn, ae in cur:
            thresholds.append({
                'test_case': tc,
                'tensor': tn,
                'absolute_error': float(ae) if ae is not None else None,
            })

    return {
        'task_id': task_id,
        'kernel': kernel,
        'mode': mode,
        'candidates': candidates,
        'thresholds': thresholds,
    }


_ARCH_ORDER = ['gfx942', 'gfx950', 'gfx1201', 'gfx90a', 'gfx1100']

def get_available_archs(conn, descriptor_id: str = 'flash') -> list[str]:
    """Return arches present in best_tuning_results in preferred display order."""
    desc = DESCRIPTORS[descriptor_id]
    with conn.cursor() as cur:
        cur.execute(f"SELECT DISTINCT arch FROM {desc['kernel_table']}")
        archs = [r[0] for r in cur.fetchall()]
    priority = {a: i for i, a in enumerate(_ARCH_ORDER)}
    return sorted(archs, key=lambda a: (priority.get(a, len(_ARCH_ORDER)), a))
