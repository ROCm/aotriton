#!/usr/bin/env python
# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path
import json
import triton
import hashlib
import itertools
import importlib.metadata

def parse():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--build_dir', type=Path, required=True)
    p.add_argument('--root_dir', type=Path, default='.', help="Root of repository directory")
    p.add_argument('--external_tuning_database', type=Path, default=None,
                   help='Hash this tuning database instead of the modules\' own; same as aotriton.generate')
    p.add_argument('--git_sha1', required=True)
    p.add_argument('--git_treesha1', required=True)
    p.add_argument('--target_arch', nargs='+')
    p.add_argument('--vendors', nargs='*', default=['amd'])
    p.add_argument('--output_file', required=True)
    args = p.parse_args()
    return args

def hashfile(fn):
    CHUNK_SIZE = 1024 * 1024
    m = hashlib.sha256()
    with open(fn, 'rb') as file:
        while chunk := file.read(CHUNK_SIZE):
            m.update(chunk)
    return m.hexdigest()

def hash_iface(files):
    '''
    Combine the content hashes of an interface's files into one. The names are
    part of the hash because the order of the patches is.
    '''
    m = hashlib.sha256()
    for fn in files:
        m.update(f'{fn.name}\0{hashfile(fn)}\n'.encode())
    return m.hexdigest()

def patch_order(fn):
    # <iface>.<NNNN>-<name>.dbpatch, applied in numeric order
    return int(fn.name.split('.')[1].split('-')[0]), fn.name

'''
Note: the database is hashed by content, so a symlink (an arch sharing a base)
is read through. Retuning the base changes the signature of every arch that
links it, and a patch changes the effective tensors without touching the base.
'''
def hash_primary(database, vendor, arch):
    '''
    {<iface>: hash} for amd/<arch>/ and its mod<N>/ subdirectories: the
    <iface>.csv tensor file followed by its patches in application order.
    '''
    result = {}
    arch_dir = database / vendor / arch
    for d in [arch_dir] + sorted(arch_dir.glob('mod[0-9]*')):
        if not d.is_dir():
            continue
        for csv in sorted(d.glob('*.csv')):
            patches = sorted(d.glob(f'{csv.stem}.*.dbpatch'), key=patch_order)
            name = csv.stem if d == arch_dir else f'{d.name}/{csv.stem}'
            result[name] = hash_iface([csv] + patches)
    return result

def hash_metadata(database):
    '''The tables shared by all arches: every file under .metadata/'''
    return hash_iface(sorted(p for p in (database / '.metadata').rglob('*') if p.is_file()))

def main():
    args = parse()
    sig = {}
    sig['AOTRITON_GIT_SHA1'] = args.git_sha1
    sig['AOTRITON_GIT_TREESHA1'] = args.git_treesha1
    # Each module holds its tuning database at <module>/database, unless an
    # external tuning database replaces it; enumerate the modules and resolve
    # each database as aotriton.generate does.
    from aotriton.codegen.parser import Parser
    from aotriton.database.csvdb import tuning_database_dir
    parser = Parser(args.root_dir / 'modules')
    databases = {family: tuning_database_dir(parser.module_dir / family, args.external_tuning_database)
                 for family in parser.discover_families()}
    databases = {family: d for family, d in databases.items() if d.is_dir()}
    db = {}
    def gen_primary_db_hash():
        for vendor, arch in itertools.product(args.vendors, args.target_arch):
            yield arch, {family: hash_primary(d, vendor, arch) for family, d in databases.items()}
    db['primary'] = dict(gen_primary_db_hash())
    db['secondary'] = {f'{family}.metadata': hash_metadata(d) for family, d in databases.items()}
    sig['DB_SHA256'] = db
    sig['TRITON_VERSION'] = str(importlib.metadata.version("triton"))
    try:
        sig['FLYC_VERSION'] = str(importlib.metadata.version("flydsl"))
    except importlib.metadata.PackageNotFoundError:
        sig['FLYC_VERSION'] = ''
    with open(args.output_file, 'w') as f:
        json.dump(sig, f, indent=2)
        print('', file=f)

if __name__ == "__main__":
    main()
