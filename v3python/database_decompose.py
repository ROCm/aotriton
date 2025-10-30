#!/usr/bin/env python
# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path
import sqlite3
from .gpu_targets import gpu2arch

def parse():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    db_base = Path(__file__).parent / 'database'
    p.add_argument('--database_file', type=Path, default=(db_base / 'tuning_database.sqlite3'))
    p.add_argument('--script_output', type=Path, default=(db_base / 'decompose_db.sh'))
    p.add_argument('--decompose_output', type=Path, default=db_base)
    args = p.parse_args()
    return args

TARXZ = r'''
tarxz() {
  (
    d=$(dirname "$1")
    b=$(basename "$1")
    cd "$d"
    tar cJf "$b.tar.xz" "$b"
  )
}'''

def write_script(args, dbc, out):
    print('#/bin/bash', file=out)
    db_meta = [tup for tup in dbc.execute("SELECT tbl_name,sql FROM sqlite_master WHERE type='table'")]
    db_tables = {tup[0]: tup[1] for tup in db_meta}
    def gen_col_names():
        for table in db_tables.keys():
            ret = []
            for col in dbc.execute(f"PRAGMA table_info({table})"):
                if col[1] != 'id':
                    ret.append(col[1])
            yield table, ret
    db_cols_strings = dict(gen_col_names())
    VENDOR = 'amd'
    # print(db_cols_strings)
    def gen():
        for table, sql in db_tables.items():
            FAMILY, kernel = table.split('$')
            for gpu, in dbc.execute(f'SELECT DISTINCT gpu FROM {table}'):
                yield table, sql, VENDOR, gpu, FAMILY.lower(), kernel
    central_dbf = args.database_file.as_posix()
    for table, raw_sql, vendor, gpu, family, kernel in gen():
        arch = gpu2arch(gpu)
        sql = raw_sql.replace('id INTEGER PRIMARY KEY,', '')
        db_dir = args.decompose_output / vendor / arch / family
        dbf = db_dir / f'{kernel}.sqlite3'
        print(f'mkdir -p {db_dir.as_posix()}', file=out)
        print(f"sqlite3 '{dbf}' << 'EOF'", file=out)
        print(sql, ';', file=out)
        print(f"ATTACH DATABASE '{central_dbf}' AS central;", file=out)
        cols = ','.join(db_cols_strings[table])
        print(f"INSERT INTO '{table}' SELECT {cols} FROM 'central'.'{table}' WHERE gpu LIKE '{arch}_%';", file=out)
        print('EOF', file=out)
    db_base = args.decompose_output / VENDOR
    print(TARXZ, file=out)
    print(f'''export -f tarxz''', file=out)
    print(f'''find {db_base.as_posix()} -name '*.sqlite3' | parallel tarxz''', file=out)


def main():
    args = parse()
    with (
        sqlite3.connect(args.database_file) as dbc,
        open(args.script_output, 'w') as f,
    ):
        write_script(args, dbc, f)

if __name__ == "__main__":
    main()
