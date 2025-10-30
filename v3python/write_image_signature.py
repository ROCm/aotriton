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
from .database import Factories as DatabaseFactories
from .rules import kernels

def parse():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--build_dir', type=Path, required=True)
    p.add_argument('--git_sha1', required=True)
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

'''
Note: here we hash the uncompressed sqlite3 file, because hash of tar file is
affected by metadata like owner/mdate
'''
def hash_primary(args, vendor, arch, k):
    fn = args.build_dir / 'database' / vendor / arch / k.FAMILY / f'{k.NAME}.sqlite3'
    if fn.is_file():
        return hashfile(fn)
    return None

def main():
    args = parse()
    sig = {}
    sig['AOTRITON_GIT_SHA1'] = args.git_sha1
    fac = DatabaseFactories.create_factory(args.build_dir)
    db = {}
    def gen_primary_db_hash():
        for vendor, arch in itertools.product(args.vendors, args.target_arch):
            d = {}
            for k in kernels:
                if k.FAMILY not in d:
                    d[k.FAMILY] = {}
                dbhash = hash_primary(args, vendor, arch, k)
                if dbhash is not None:
                    d[k.FAMILY][k.NAME] = dbhash
            yield arch, d
    db['primary'] = dict(gen_primary_db_hash())
    def gen_secondary_db_hash():
        for k, v in fac.SECONDARY_DATABASES.items():
            yield k, hashfile(args.build_dir / v)
    db['secondary'] = dict(gen_secondary_db_hash())
    sig['DB_SHA256'] = db
    sig['TRITON_VERSION'] = str(importlib.metadata.version("triton"))
    with open(args.output_file, 'w') as f:
        json.dump(sig, f, indent=2)
        print('', file=f)

if __name__ == "__main__":
    main()
