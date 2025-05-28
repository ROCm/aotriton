# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sqlite3
import pandas as pd
from ..base import typed_choice as TC
from ..op import Operator
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
    return stmt, params

class Factory(object):
    SIGNATURE_FILE = 'tuning_database.sqlite3'

    def __init__(self, path):
        log(lambda : f'sqlite3.connect({path / self.SIGNATURE_FILE})')
        self._conn = sqlite3.connect(path / self.SIGNATURE_FILE)
        self._conn.set_trace_callback(log) # Debug

    def create_view(self, functional):
        log(lambda : f'{functional=}')
        meta = functional.meta_object
        pfx = 'OP$' if isinstance(meta, Operator) else ''
        table_name = pfx + meta.FAMILY.upper() + '$' + meta.NAME
        # TODO: Incremental changes:
        # 1. load database_gpus first
        # 2. then override entries with optimized_for gpus
        def build_sql(choice_dict):
            wheres = {
                'gpu' : functional.database_gpus,
            }
            for key, value in choice_dict.items():
                if isinstance(value, TC.TypedChoice) and value.is_tensor:
                    wheres[f'inputs${key}_dtype'] = value
                else:
                    wheres[f'inputs${key}'] = value
            return create_select_stmt(table_name, wheres)
        stmt, params = build_sql(functional.compact_choices)
        try:
            log(lambda : f'select stmt: {stmt} params {params}')
            df = pd.read_sql_query(stmt, self._conn, params=params)
            if not df.empty:
                return df
            # Downgrade
            stmt, params = build_sql(functional.fallback_choices)
            return pd.read_sql_query(stmt, self._conn, params=params)
        except pd.errors.DatabaseError:
            log(lambda : f'Table {table_name} may not exist. select stmt: {stmt} params {params}')
            return None
