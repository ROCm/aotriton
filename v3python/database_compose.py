#!/usr/bin/env python
# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path
import sqlite3
import tarfile

def parse():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    db_base = Path(__file__).parent / 'database'
    p.add_argument('--database_file', type=Path, required=True)
    p.add_argument('--decomposed', type=Path, default=db_base)
    p.add_argument('--vendors', nargs='*', default=['amd'])
    args = p.parse_args()
    return args

def compose_database(args, dbc):
    for vendor in args.vendors:
        vendor_dir = args.decomposed / vendor
        for fn in vendor_dir.glob('*/*/*.sqlite3'):
            arch, family, base = fn.parts[-3:]
            kernel_name = Path(base).stem
            table_string = f"'{family.upper()}${kernel_name}'"
            dbc.execute(f"ATTACH DATABASE '{fn.as_posix()}' AS subdb;")
            dbc.execute(f"INSERT INTO {table_string} SELECT * FROM subdb.{table_string};")
            dbc.commit()
            dbc.execute(f"DETACH DATABASE subdb;")

def main():
    args = parse()
    with (
        sqlite3.connect(args.database_file) as dbc,
    ):
        compose_database(args, dbc)

if __name__ == "__main__":
    main()
