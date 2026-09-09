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
    p.add_argument('--arch', default=None,
                   help='Only emit shards for this architecture (e.g. gfx942). '
                        'Other architectures already present under '
                        '--decompose_output are left alone. Default: every '
                        'architecture found in the database.')
    args = p.parse_args()
    return args

TARXZ = r'''
tarxz() {
  (
    d=$(dirname "$1")
    b=$(basename "$1")
    cd "$d"
    tar cJf "$b.tar.xz" "$b"
    rm "$b"
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
                # Filter on the arch, not the gpu: one arch can have several
                # mods (gfx942_mod0/mod1/...), and restricting to an
                # architecture means all of its mods, not an assumed _mod0.
                if args.arch is not None and gpu2arch(gpu) != args.arch:
                    continue
                yield table, sql, VENDOR, gpu, FAMILY.lower(), kernel
    central_dbf = args.database_file.as_posix()
    emitted = 0
    for table, raw_sql, vendor, gpu, family, kernel in gen():
        emitted += 1
        arch = gpu2arch(gpu)
        sql = raw_sql.replace('id INTEGER PRIMARY KEY,', '')
        db_dir = args.decompose_output / vendor / arch
        dbf = db_dir / f'{kernel}.sqlite3'
        print(f'mkdir -p {db_dir.as_posix()}', file=out)
        print(f"sqlite3 '{dbf}' << 'EOF'", file=out)
        print(sql, ';', file=out)
        print(f"ATTACH DATABASE '{central_dbf}' AS central;", file=out)
        cols = ','.join(db_cols_strings[table])
        print(f"INSERT INTO '{table}' SELECT {cols} FROM 'central'.'{table}' WHERE gpu LIKE '{arch}_%';", file=out)
        print('EOF', file=out)
    if args.arch is not None and emitted == 0:
        # An empty script exits 0 and looks like a successful decompose. Say so
        # instead: the usual cause is that the export step ran for a different
        # architecture, so this one is simply not in the database.
        print(f"echo 'Warning: no gpu in {central_dbf} belongs to arch "
              f"{args.arch}; nothing to decompose.' >&2", file=out)

    db_base = args.decompose_output / VENDOR
    print(TARXZ, file=out)
    print(f'''export -f tarxz''', file=out)
    # Deliberately NOT scoped to amd/<arch> even under --arch. tarxz() removes
    # each .sqlite3 once it is archived, so a completed earlier run leaves none
    # behind for this find to pick up -- it only ever sees what the INSERTs
    # above just created. Narrowing it would add a path that does not exist yet
    # on a first run, for no gain.
    print(f'''find {db_base.as_posix()} -name '*.sqlite3' | "$GNU_PARALLEL" tarxz''', file=out)


def main():
    args = parse()
    with (
        sqlite3.connect(args.database_file) as dbc,
        open(args.script_output, 'w') as f,
    ):
        write_script(args, dbc, f)

if __name__ == "__main__":
    main()
