# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sqlite3
import tarfile
from pathlib import Path
import pandas as pd
from ..template_instantiation.ir import typed_choice as TC
from ..utils import log
from ..gpu_targets import AOTRITON_TUNING_DATABASE_REUSE

# Checked-in baseline secondary databases (e.g. op_database.sqlite3), shipped
# as .tar.xz since they're static across build_dirs. Used as a fallback below
# when a caller hasn't staged its own copy (e.g. a freshly tuned database).
_MODULES_DATABASE = Path(__file__).resolve().parents[2] / 'modules' / 'database'

def _extract_if_needed(fn: Path) -> None:
    tarball = fn.parent / (fn.name + '.tar.xz')
    if not fn.is_file() and tarball.is_file():
        with tarfile.open(tarball, mode='r:xz') as tf:
            tf.extractall(fn.parent)

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
            if not fn.is_file():
                # Caller didn't stage its own copy (e.g. a freshly tuned
                # database) -- fall back to the checked-in baseline.
                fn = _MODULES_DATABASE / Path(bn).name
                _extract_if_needed(fn)
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
                return df, format_sql(stmt, params)
            # Downgrade
            stmt, params = build_sql(functional.fallback_choices)
            df = pd.read_sql_query(stmt, self._conn, params=params)
            return df, format_sql(stmt, params)
        except pd.errors.DatabaseError:
            log(lambda : f'Table {table_name} may not exist. select stmt: {stmt} params {params}')
            return None, format_sql(stmt, params)
