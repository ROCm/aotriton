# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sqlite3
import pandas as pd
from ..template_instantiation.ir import typed_choice as TC
from ..utils import log
from ..gpu_targets import AOTRITON_TUNING_DATABASE_FALLBACK

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
        for schema, bn in self.SECONDARY_DATABASES.items():
            fn = path / bn
            if fn.is_file():
                log(lambda : f"ATTACH DATABASE '{fn.as_posix()}' AS {schema};")
                self._conn.execute(f"ATTACH DATABASE '{fn.as_posix()}' AS {schema};")
            else:
                assert False, f'{fn} is not a file, {path}'

    def create_view(self, functional):
        log(lambda : f'{functional=}')
        meta = functional.meta_object
        pfx = 'op.' if getattr(meta, 'CODEGEN_MODULE', None) == 'op' else ''
        table_name = pfx + meta.FAMILY.upper() + '$' + meta.NAME

        # Extract GPU lists from tuples: database_gpus now returns [(target_gpu, fallback_gpu), ...]
        target_gpus = [target for target, _ in functional.database_gpus]
        fallback_gpus = [fallback for _, fallback in functional.database_gpus]

        def build_sql(choice_dict, gpu_list):
            wheres = {'gpu': gpu_list}
            for key, value in choice_dict.items():
                if isinstance(value, TC.TypedChoice) and value.is_tensor:
                    wheres[f'inputs${key}_dtype'] = value
                else:
                    wheres[f'inputs${key}'] = value
            return create_select_stmt(table_name, wheres)

        # Tier 1: Query fallback database
        stmt, params = build_sql(functional.compact_choices, fallback_gpus)
        log(lambda : f'Tier 1 (fallback {fallback_gpus}): {stmt}')
        try:
            df_fallback = pd.read_sql_query(stmt, self._conn, params=params)
        except pd.errors.DatabaseError:
            log(lambda : f'Table {table_name} may not exist. select stmt: {stmt} params {params}')
            return None, format_sql(stmt, params)

        # If fallback empty, try fallback_choices
        if df_fallback.empty:
            stmt, params = build_sql(functional.fallback_choices, fallback_gpus)
            df_fallback = pd.read_sql_query(stmt, self._conn, params=params)

        # Tier 2: Query target GPUs (may include partial tuning)
        # Only query if target differs from fallback
        if target_gpus != fallback_gpus:
            stmt_opt, params_opt = build_sql(functional.compact_choices, target_gpus)
            log(lambda : f'Tier 2 (target {target_gpus}): {stmt_opt}')
            try:
                df_optimized = pd.read_sql_query(stmt_opt, self._conn, params=params_opt)
            except pd.errors.DatabaseError:
                df_optimized = pd.DataFrame()

            # Merge: fallback first, then override with target
            if not df_optimized.empty:
                df_combined = pd.concat([df_fallback, df_optimized], ignore_index=True)
                input_cols = [c for c in df_combined.columns if c.startswith('inputs$')] + ['gpu']
                df_combined = df_combined.drop_duplicates(subset=input_cols, keep='last')
                log(lambda : f'Merged: {len(df_fallback)} fallback + {len(df_optimized)} target = {len(df_combined)} total')
                return df_combined, format_sql(stmt_opt, params_opt)

        # No optimization tier or it was empty
        return df_fallback, format_sql(stmt, params)
