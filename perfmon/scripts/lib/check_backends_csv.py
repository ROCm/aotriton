#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Checks a release's hand-written backends.csv against its own build.

    check_backends_csv.py <backends.csv> <aotriton build dir>

No release can list its backends, so the csv is written by hand from the
source. The build is the one place that states them: the code generator writes
`enum BackendEnum` into iface.op_attn_{fwd,bwd}.h, in the order of the index
`attn_options::force_backend_index` takes. This fails unless, for each iface of
the csv, the enum has as many backends as the csv has rows and the name of each
contains the row's enum_hint (ignoring case and underscores).

Run with `python3 -I`: it needs only the standard library.
"""

import csv
import pathlib
import re
import sys

IFACES = ('op_attn_fwd', 'op_attn_bwd')


def read_csv(path):
    rows = {iface: [] for iface in IFACES}
    with open(path, newline='') as f:
        lines = [line for line in f if line.strip() and not line.lstrip().startswith('#')]
    reader = csv.DictReader(lines)
    if reader.fieldnames is None or reader.fieldnames[:3] != ['iface', 'index', 'name']:
        sys.exit(f'{path}: the header must start with iface,index,name')
    for row in reader:
        iface = row['iface']
        if iface not in rows:
            sys.exit(f'{path}: unknown iface {iface!r}')
        if int(row['index']) != len(rows[iface]):
            sys.exit(f'{path}: {iface} rows must be indexed 0, 1, ... in order')
        rows[iface].append((row['name'], (row.get('enum_hint') or '').strip()))
    return rows


def read_enum(header):
    """The enumerator names of BackendEnum in index order, without None and Max."""
    text = pathlib.Path(header).read_text()
    m = re.search(r'enum\s+BackendEnum\s*(?::\s*\w+)?\s*\{(.*?)\}', text, re.S)
    if not m:
        sys.exit(f'{header}: no enum BackendEnum')
    found = {}
    for name, value in re.findall(r'\b(\w+)\s*=\s*(-?\d+)', m.group(1)):
        if name in ('None', 'Max'):
            continue
        found[int(value)] = name
    if sorted(found) != list(range(len(found))):
        sys.exit(f'{header}: BackendEnum values are not 0..n-1: {sorted(found)}')
    return [found[i] for i in range(len(found))]


def normalized(text):
    return text.replace('_', '').lower()


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    csv_path, build_dir = sys.argv[1], pathlib.Path(sys.argv[2])
    rows = read_csv(csv_path)
    failed = False
    for iface, expected in rows.items():
        if not expected:
            continue
        headers = sorted(build_dir.rglob(f'iface.{iface}.h'))
        if not headers:
            sys.exit(f'no iface.{iface}.h under {build_dir}: was the build kept?')
        names = read_enum(headers[0])
        if len(names) != len(expected):
            print(f'{csv_path}: {iface} has {len(expected)} rows but the build has '
                  f'{len(names)} backends: {names}', file=sys.stderr)
            failed = True
            continue
        for index, ((name, hint), enum_name) in enumerate(zip(expected, names)):
            if hint and normalized(hint) not in normalized(enum_name):
                print(f'{csv_path}: {iface} {index} is "{name}" (hint {hint!r}) but the '
                      f'build calls it {enum_name}', file=sys.stderr)
                failed = True
            elif not hint:
                print(f'{csv_path}: {iface} {index} "{name}" has no enum_hint; only the '
                      f'count was checked ({enum_name})', file=sys.stderr)
    if failed:
        sys.exit(1)
    print(f'{csv_path}: matches the BackendEnum of {build_dir}', file=sys.stderr)


if __name__ == '__main__':
    main()
