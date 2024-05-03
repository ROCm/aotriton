#!/usr/bin/env python

import sqlite3
import itertools
import json
import argparse
import sys
import csv

def parse():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-f', '--file', type=str, required=True, help='Database file')
    p.add_argument('-k', '--kernel_family', type=str, required=True, help='Kernel family')
    p.add_argument('-v', '--verbose', action='store_true', help='Verbose')
    p.add_argument('--action', type=str, required=False, default='pipejson',
                   choices=['pipejson', 'createtableonly', 'dumpcsv', 'loadcsv'],
                   help='Action to perform')
    p.add_argument('--table_name', type=str, help='Table to dump/load')
    p.add_argument('--table_file', type=str, help='CSV file of dump/load')
    p.add_argument('--select_where', type=str, default='', help='Extra WHERE clause for SQL to only dump selected rows to CSV file')
    p.add_argument('--ignore_id', action='store_true', help='Ignore row IDs when loading CSV to database, useful for table merge')
    args = p.parse_args()
    return args

class TuningDatabase(object):
    PYTYPE_TO_SQLTYPE = {
        int : 'INTEGER',
        str : 'TEXT',
        float : 'REAL',  # Not used right now. Possible to store actual profiling results
    }

    def __init__(self, args):
        self._args = args
        self._conn = sqlite3.connect(args.file)  # TODO: use autocommit for python 3.12+
        self._conn.isolation_level = None  # TODO: add --batch mode,
        self._cur = self._conn.cursor()
        self._table_existance_checked = set()

    @property
    def verbose(self):
        return self._args.verbose

    def value_to_pytype(self, v):
        if isinstance(v, int):
            return int
        if isinstance(v, str):
            return str
        if isinstance(v, float):
            return float
        assert False, f"Unsupported type for value {v}"

    def sqltype(self, pytype):
        return self.PYTYPE_TO_SQLTYPE[pytype]

    def collect_columns(self, sub_tune_info: dict, *, prefix=''):
        ret = []
        for k, v in sub_tune_info.items():
            # Do not store Q.shape and others, which is an array and makes things more complicated
            if k.endswith('.shape'):
                continue
            tup = (f'{prefix}{k}', v, self.value_to_pytype(v))
            ret.append(tup)
        if self.verbose:
            print(f'collect_columns: {ret=}')
        return ret

    def get_table_name(self, tune_info: dict) -> str:
        kn = tune_info['kernel_name']
        return f'{self._args.kernel_family}${kn}'

    def ensure_table(self, tune_info : dict) -> str:
        kn = tune_info['kernel_name']
        if kn in self._table_existance_checked:
            return self.get_table_name(tune_info)
        table_name = self._create_table(tune_info)
        self._table_existance_checked.add(kn)
        return table_name

    def _create_table(self, tune_info):
        columns = self.collect_columns(tune_info['inputs'], prefix='inputs$')
        # UNIQUE = 'UNIQUE'
        col_def = ['id INTEGER PRIMARY KEY', f'arch TEXT']
        col_def += [f'{colname} {self.sqltype(pytype)}' for colname, _, pytype in columns]
        unique = ', '.join(['arch'] + [colname for colname, _, _ in columns])
        columns = self.collect_columns(tune_info['tuned_kernel'], prefix='tuned_kernel$')
        col_def += [f'{colname} {self.sqltype(pytype)}' for colname, _, pytype in columns]
        columns = self.collect_columns(tune_info['compiler_options'], prefix='compiler_options$')
        col_def += [f'{colname} {self.sqltype(pytype)}' for colname, _, pytype in columns]
        col_def_stmt = ', '.join(col_def)
        table_name = self.get_table_name(tune_info)
        stmt = f"CREATE TABLE IF NOT EXISTS {table_name} ({col_def_stmt}, UNIQUE({unique}));"
        if self.verbose:
            print("Executing", stmt)
        self._cur.execute(stmt)
        self._conn.commit()
        return table_name

    def upsert(self, line_text, *, create_table_only):
        if not line_text:
            return
        tune_info = json.loads(line_text)
        if self.verbose:
            print(f'{line_text=}')
            print(f'{tune_info=}')
        sql_table = self.ensure_table(tune_info)
        if create_table_only:
            return
        inputs_columns = self.collect_columns(tune_info['inputs'], prefix='inputs$')
        tuned_kernel_columns = self.collect_columns(tune_info['tuned_kernel'], prefix='tuned_kernel$')
        compiler_options_columns = self.collect_columns(tune_info['compiler_options'], prefix='compiler_options$')
        all_colnames = ['arch'] + [colname for colname, _, _ in itertools.chain(inputs_columns, tuned_kernel_columns, compiler_options_columns)]
        stmt_colnames = ', '.join(all_colnames)
        stmt_placeholders = ', '.join(['?'] * len(all_colnames))
        stmt = f'INSERT INTO {sql_table}({stmt_colnames}) VALUES({stmt_placeholders})'
        values = [tune_info['arch']] + [v for _, v, _ in itertools.chain(inputs_columns, tuned_kernel_columns, compiler_options_columns)]
        if self.verbose:
            print("values 1: ", values)
        stmt += ' ON CONFLICT DO UPDATE SET '
        stmt += ', '.join([f'{colname}=?' for colname, _, _ in itertools.chain(tuned_kernel_columns, compiler_options_columns)])
        values += [v for _, v, _ in itertools.chain(tuned_kernel_columns, compiler_options_columns)]
        if self.verbose:
            print("values 2: ", values)
        if self.verbose:
            print("Executing", stmt, "with", values)
        self._cur.execute(stmt, values)
        self._conn.commit()

    def close(self):
        self._cur.close()
        self._conn.close()

    def dumpcsv(self, table_name, table_file):
        with open(table_file, mode='w', newline='') as file:
            self._cur.execute(f"SELECT * FROM {table_name} WHERE {self._args.select_where};")
            writer = csv.writer(file)
            colunm_names = [tup[0] for tup in self._cur.description]
            writer.writerow(colunm_names)
            while True:
                tup = self._cur.fetchone()
                if tup is None:
                    break
                writer.writerow(tup)

    def loadcsv(self, table_file, table_name):
        with open(table_file, mode='r', newline='') as file:
            reader = csv.reader(file)
            csv_headers = next(reader)
            if self._args.ignore_id:
                assert csv_headers[0] == 'id', "--ignore_id: First column of CSV is not 'id'. This tool does not handle more compilicated situations."
                csv_headers = csv_headers[1:]
            colunm_names = ', '.join(csv_headers)
            placeholders = ', '.join(['?'] * len(csv_headers))
            stmt = f'INSERT INTO {table_name} ({colunm_names}) VALUES({placeholders});'
            for row in reader:
                if self._args.ignore_id:
                    row = row[1:]
                self._cur.execute(stmt, row)
            self._conn.commit()

def main():
    args = parse()
    db = TuningDatabase(args)
    if args.action == 'pipejson' or args.action == 'createtableonly':
        # FIXME: support pipes and streaming file with json_stream
        if args.action == 'createtableonly':
            create_table_only = True
        else:
            create_table_only = False
        for line in sys.stdin:
            db.upsert(line, create_table_only=create_table_only)
        print("[table_tool] Input closed, exiting", file=sys.stderr)
    elif args.action == 'dumpcsv':
        db.dumpcsv(args.table_name, args.table_file)
    elif args.action == 'loadcsv':
        db.loadcsv(args.table_file, args.table_name)
    db.close()

if __name__ == '__main__':
    main()
