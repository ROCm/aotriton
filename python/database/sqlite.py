# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sqlite3
import pandas as pd
from ..template_instantiation.ir import typed_choice as TC
from ..utils import log

def format_sql(stmt, params):
    return (stmt, params)


class Factory(object):
    SIGNATURE_FILE = 'database/tuning_database.sqlite3'
    SECONDARY_DATABASES = {
        'op': 'database/op_database.sqlite3',
    }

    def __init__(self, path):
        log(lambda : f'sqlite3.connect({path / self.SIGNATURE_FILE})')
        self._conn = sqlite3.connect(path / self.SIGNATURE_FILE)
        self._conn.set_trace_callback(log) # Debug
        for schema, bn in self.SECONDARY_DATABASES.items():
            fn = path / bn
            if fn.is_file():
                log(lambda : f"ATTACH DATABASE '{fn.as_posix()}' AS {schema};")
                self._conn.execute(f"ATTACH DATABASE '{fn.as_posix()}' AS {schema};")
            else:
                assert False, f'{fn} is not a file, {path}'

    def create_view(self, functional):
        """Query tuning database with N-tier GPU prioritization via SQL.

        Uses window functions to select highest-priority GPU per config within
        a single query. Results are tagged with target_gpu for LUT filtering.
        """
        log(lambda : f'{functional=}')
        meta = functional.meta_object
        pfx = 'op.' if getattr(meta, 'CODEGEN_MODULE', None) == 'op' else ''
        table_name = f"'{pfx}{meta.FAMILY.upper()}${meta.NAME}'"

        # Columns to partition by (the LUT index dimensions)
        # Triton KernelDescription has autotune_keys, Operator has OPTUNE_KEYS,
        # flyc/affine KernelDescription delegates to its _functionals_source (Operator)
        if hasattr(meta, 'autotune_keys'):
            partition_keys = [k for k, _ in meta.autotune_keys]
        elif hasattr(meta, 'OPTUNE_KEYS'):
            partition_keys = list(meta.OPTUNE_KEYS.keys())
        elif hasattr(meta, '_functionals_source') and meta._functionals_source:
            partition_keys = list(meta._functionals_source.OPTUNE_KEYS.keys())
        else:
            partition_keys = []
        partition_cols = ', '.join(f'"inputs${k}"' for k in partition_keys)

        # {target_gpu: [self, fallback1, ...]} priority chains
        target_priority_map = functional.database_gpus

        # Collect all GPUs across all chains for the WHERE clause
        all_gpus = sorted({g for chain in target_priority_map.values() for g in chain})

        def build_sql(choice_dict):
            """Build single SQL query with CTEs per target, combined via UNION ALL."""
            # Build WHERE clause for fixed inputs
            where_parts = [f'gpu IN ({",".join("?" * len(all_gpus))})']
            base_params = list(all_gpus)
            for key, value in choice_dict.items():
                col = f'inputs${key}_dtype' if isinstance(value, TC.TypedChoice) and value.is_tensor else f'inputs${key}'
                where_parts.append(f'"{col}" = ?')
                base_params.append(value.sql_value if isinstance(value, TC.TypedChoice) else value)
            where_clause = ' AND '.join(where_parts)

            # Build one CTE per target with ROW_NUMBER for priority selection
            ctes, selects, params = [], [], []
            for i, (target_gpu, chain) in enumerate(target_priority_map.items()):
                # CASE gpu WHEN 'gpu1' THEN 1 WHEN 'gpu2' THEN 2 ... ELSE 99 END
                case_expr = 'CASE gpu ' + ' '.join(f'WHEN ? THEN {j}' for j in range(1, len(chain)+1)) + ' ELSE 99 END'
                ctes.append(f"""r{i} AS (
SELECT *, ROW_NUMBER() OVER (PARTITION BY {partition_cols} ORDER BY {case_expr}) AS rn
FROM {table_name} WHERE {where_clause})""")
                selects.append(f"SELECT *, '{target_gpu}' AS target_gpu FROM r{i} WHERE rn = 1")
                params.extend(chain)  # CASE params
                params.extend(base_params)  # WHERE params

            sql = 'WITH ' + ',\n'.join(ctes) + '\n' + '\nUNION ALL\n'.join(selects)
            log(lambda : f'Priority query: {sql}')

            try:
                df = pd.read_sql_query(sql, self._conn, params=params)
                if not df.empty:
                    df = df.drop(columns=['rn'])
                return df, format_sql(sql, params)
            except pd.errors.DatabaseError:
                log(lambda : f'Table {table_name} may not exist')
                return None, format_sql(sql, params)

        # Try compact_choices first, then fallback_choices
        df, sql = build_sql(functional.compact_choices)
        if df is None or not df.empty:
            return df, sql
        return build_sql(functional.fallback_choices)
