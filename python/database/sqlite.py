# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sqlite3
import pandas as pd
from ..template_instantiation.ir import typed_choice as TC
from ..utils import log
from ..gpu_targets import AOTRITON_TUNING_DATABASE_REUSE

'''
We don't really need a LazyTableView, if Lazy evaluation is needed, a
LazyPandasDataFrame is more preferrable
'''
# from .view import LazyTableView as SqliteTableView

def create_select_stmt(table_name, wheres):
    stmt = f"SELECT * FROM {table_name} WHERE "
    where_stmt = []
    params = []
    for k, v in wheres.items():
        if isinstance(v, list) or isinstance(v, tuple):
            qm = ', '.join(['?'] * len(v))
            where_stmt.append(f'{k} IN ({qm})')
            params += v
        else:
            where_stmt.append(f'{k} = ?')
            params.append(v.sql_value if isinstance(v, TC.TypedChoice) else v)
    stmt += ' AND '.join(where_stmt)
    # print('create_select_stmt', stmt)
    return stmt, params

def format_sql(stmt, params):
    template = stmt.replace('?', '{!r}')
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
        self._input_config_cols_cache = {} # Cache "inputs$" columns in database
        for schema, bn in self.SECONDARY_DATABASES.items():
            fn = path / bn
            if fn.is_file():
                log(lambda : f"ATTACH DATABASE '{fn.as_posix()}' AS {schema};")
                self._conn.execute(f"ATTACH DATABASE '{fn.as_posix()}' AS {schema};")
            else:
                assert False, f'{fn} is not a file, {path}'

    def _get_input_config_cols(self, table_name):
        """
        Retrieves 'inputs$' columns with caching.

        These columns identify kernel input parameters (dtype, head dimension, causal
        type, etc.). Cached per table since schema doesn't change during build.
        """
        if table_name not in self._input_config_cols_cache:
            try:
                cursor = self._conn.cursor()
                cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 0')
                cols = [desc[0] for desc in cursor.description if desc[0].startswith('inputs$')]
                self._input_config_cols_cache[table_name] = cols
            except sqlite3.OperationalError:
                # Table doesn't exist - return empty list, will be caught later
                self._input_config_cols_cache[table_name] = []
        return self._input_config_cols_cache[table_name]

    def create_view(self, functional):
        log(lambda : f'{functional=}')
        meta = functional.meta_object
        pfx = 'op.' if getattr(meta, 'CODEGEN_MODULE', None) == 'op' else ''
        table_name = pfx + meta.FAMILY.upper() + '$' + meta.NAME

        # Get all target GPUs and their priority chains
        target_priority_map = functional.database_gpus  # dict[str, list[str]]

        def build_sql_with_window_functions(choice_dict):
            """Build UNION ALL query with window functions for each target GPU."""

            # Build WHERE conditions for choices
            choice_conditions = []
            choice_params = []
            for key, value in choice_dict.items():
                col = f'inputs${key}_dtype' if isinstance(value, TC.TypedChoice) and value.is_tensor else f'inputs${key}'
                choice_conditions.append(f'{col} = ?')
                choice_params.append(value.sql_value if isinstance(value, TC.TypedChoice) else value)

            # Get partition columns (cached)
            input_cols_joined = ', '.join(self._get_input_config_cols(table_name))

            # Build UNION branches for each target GPU
            union_parts = []
            gpu_params = []

            for target_gpu, priority_chain in target_priority_map.items():
                # Build CASE statement for priority order
                case_stmt = ' '.join(f"WHEN '{gpu}' THEN {idx}"
                                    for idx, gpu in enumerate(priority_chain, 1)) + ' ELSE 99'
                gpu_placeholders = ','.join(['?'] * len(priority_chain))

                union_parts.append(f"""
    SELECT *, '{target_gpu}' as target_gpu,
           ROW_NUMBER() OVER (
               PARTITION BY {input_cols_joined}
               ORDER BY CASE gpu {case_stmt} END
           ) AS rn
    FROM {{base_table}}
    WHERE gpu IN ({gpu_placeholders})""")
                gpu_params.extend(priority_chain)

            # Build final query - use filtered_base CTE if we have choice conditions
            if choice_conditions:
                choice_where = ' AND '.join(choice_conditions)
                base_table_def = f'filtered_base AS (\n    SELECT * FROM "{table_name}"\n    WHERE {choice_where}\n)'
                base_ref = 'filtered_base'
                all_params = choice_params + gpu_params
            else:
                base_table_def = None
                base_ref = f'"{table_name}"'
                all_params = gpu_params

            # Format union_parts with actual base table reference
            formatted_unions = '\n UNION ALL '.join(part.format(base_table=base_ref) for part in union_parts)

            if base_table_def:
                full_query = f"""
WITH {base_table_def},
all_targets AS (
{formatted_unions}
)
SELECT * FROM all_targets WHERE rn = 1
"""
            else:
                full_query = f"""
WITH all_targets AS (
{formatted_unions}
)
SELECT * FROM all_targets WHERE rn = 1
"""
            return full_query, all_params

        stmt, params = build_sql_with_window_functions(functional.compact_choices)
        try:
            log(lambda : f'select stmt: {stmt} params {params}')
            df = pd.read_sql_query(stmt, self._conn, params=params)
            if not df.empty:
                return df, format_sql(stmt, params)
            # Downgrade
            stmt, params = build_sql_with_window_functions(functional.fallback_choices)
            df = pd.read_sql_query(stmt, self._conn, params=params)
            return df, format_sql(stmt, params)
        except pd.errors.DatabaseError:
            log(lambda : f'Table {table_name} may not exist. select stmt: {stmt} params {params}')
            return None, format_sql(stmt, params)
