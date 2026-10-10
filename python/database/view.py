# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

@dataclass
class QueryResults:
    col_names: list[str]
    rows: list

class LazyTableView(object):

    '''
    from_columns: None selects all
    '''
    def self(self, conn, table_name, wheres, from_columns=None):
        self._conn = conn
        self._table_name = table_name
        self._wheres = wheres
        self._column_names = self.__build_columns(from_columns)
        self._cached_rows = None
        self._stmt = self.__build_statement()

    @property
    def rows(self):
        if self._cached_rows is None:
            select_vals = self.__build_execute_vals()
            self._cached_rows = self._conn.execute(self._select_stmt, select_vals).fetchall()
        return self._cached_rows

    @property
    def columns(self):
        return self._column_names
