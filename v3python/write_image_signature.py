#!/usr/bin/env python
# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path
import json
import triton
import hashlib
import importlib.metadata
from .database import Factories as DatabaseFactories

def parse():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('build_dir', type=Path)
    p.add_argument('git_sha1')
    p.add_argument('output_file')
    args = p.parse_args()
    return args

def hashfile(fn):
    CHUNK_SIZE = 1024 * 1024
    m = hashlib.sha256()
    with open(fn, 'rb') as file:
        while chunk := file.read(CHUNK_SIZE):
            m.update(chunk)
    return m.hexdigest()

def main():
    args = parse()
    sig = {}
    sig['AOTRITON_GIT_SHA1'] = args.git_sha1
    fac = DatabaseFactories.create_factory(args.build_dir)
    db = { 'primary' : hashfile(args.build_dir / fac.SIGNATURE_FILE) }
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
