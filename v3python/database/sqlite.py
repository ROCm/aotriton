# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sqlite3
import pandas as pd
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
            where_stmt.append(f'{k} = (qm)')
            params += v
        else:
            where_stmt.append(f'{k} = ?')
            params.append(v)
    stmt += ' AND '.join(where_stmt)
    return stmt, params

class Factory(object):
    SIGNATURE_FILE = 'tuning_database.sqlite3'

    def __init__(self, path):
        self._conn = sqlite3.connect(path / self.SIGNATURE_FILE)

    def create_view(self, functional):
        print(f'{functional=}')
        meta = functional.meta_object
        is_op = hasattr(meta, 'OPERATOR')
        pfx = 'OP$' if is_op else ''
        table_name = pfx + meta.FAMILY + '$' + meta.NAME
        wheres = {
            'gpu' : functional.optimized_for,
        }
        for key, value in functional.compact_choices.items():
            wheres[f'inputs${key}'] = value
        stmt, params = create_select_stmt(table_name, wheres)
        try:
            return pd.read_sql_query(stmt, self._conn, params=params)
        except pd.errors.DatabaseError:
            print(f'Table {table_name} do not exist')
            return None
