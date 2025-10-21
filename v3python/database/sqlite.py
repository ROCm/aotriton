# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sqlite3
import pandas as pd
from pathlib import Path, PurePath
from ..base import typed_choice as TC
from ..op import Operator
from ..utils import log
from ..gpu_targets import AOTRITON_TUNING_DATABASE_REUSE, gpu2arch

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
    return template.format(*params)

class Factory(object):
    SIGNATURE_FILE = 'database/tuning_database.sqlite3'
    SECONDARY_DATABASES = {
        'op': 'database/op_database.sqlite3',
    }

    def __init__(self, build_dir: Path):
        log(lambda : f'sqlite3.connect({build_dir / self.SIGNATURE_FILE})')
        self._conn = sqlite3.connect(build_dir / self.SIGNATURE_FILE)
        self._conn.set_trace_callback(log) # Debug
        def gen():
            database_dir = build_dir / 'database'
            for fn in database_dir.glob('*/*/*.sqlite3'):
                rfn = fn.relative_to(database_dir)
                # print(f'{rfn.parts=}')
                arch, family, fname = rfn.parts
                kernel = PurePath(fname).stem
                schema = f'{family}${kernel}@{arch}'
                yield schema, fn
            yield from self.SECONDARY_DATABASES
        for schema, bn in gen():
            fn = build_dir / bn
            if fn.is_file():
                stmt = f"ATTACH DATABASE '{fn.as_posix()}' AS '{schema}';"
                # print(f'{stmt=}')
                self._conn.execute(stmt)
            else:
                assert False, f'{fn} is not a file, {build_dir=}'

    def create_view(self, functional):
        log(lambda : f'{functional=}')
        meta = functional.meta_object
        pfx = 'op.' if isinstance(meta, Operator) else ''
        table_name = pfx + meta.FAMILY.upper() + '$' + meta.NAME
        # TODO: Incremental changes:
        # 1. load database_gpus first
        # 2. then override entries with optimized_for gpus
        def build_sql(choice_dict):
            wheres = {
                'gpu' : functional.database_gpus,
            }
            database_arch = gpu2arch(functional.database_gpus[0])
            for key, value in choice_dict.items():
                if isinstance(value, TC.TypedChoice) and value.is_tensor:
                    wheres[f'inputs${key}_dtype'] = value
                else:
                    wheres[f'inputs${key}'] = value
            if not pfx:
                schema = f'{functional.family}${functional.name}@{database_arch}'
                # Both schema name and table name must be single quoted
                full_table_name = f"'{schema}'.'{table_name}@{database_arch}'"
            else:
                full_table_name = table_name
            return create_select_stmt(full_table_name, wheres)
        stmt, params = build_sql(functional.compact_choices)
        try:
            log(lambda : f'select stmt: {stmt} params {params}')
            df = pd.read_sql_query(stmt, self._conn, params=params)
            if not df.empty:
                return df, format_sql(stmt, params)
            # Downgrade
            stmt, params = build_sql(functional.fallback_choices)
            df = pd.read_sql_query(stmt, self._conn, params=params)
            return df, format_sql(stmt, params)
        except pd.errors.DatabaseError:
            log(lambda : f'Table {table_name} may not exist. select stmt: {stmt} params {params}')
            return None, ''
