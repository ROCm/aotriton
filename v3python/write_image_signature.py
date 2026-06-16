#!/usr/bin/env python
# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import argparse
from pathlib import Path
import os
import sys
import json
import hashlib
import itertools
import importlib.metadata
import subprocess
import yaml
from .database import Factories as DatabaseFactories
from .rules import kernels

# Mirror v3python/codegen/root.py: the venv python relative to its prefix.
REL_PYTHON = Path(os.path.abspath(sys.executable)).relative_to(Path(sys.exec_prefix))

def parse():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--build_dir', type=Path, required=True)
    p.add_argument('--git_sha1', required=True)
    p.add_argument('--git_treesha1', required=True)
    p.add_argument('--target_arch', nargs='+')
    p.add_argument('--vendors', nargs='*', default=['amd'])
    p.add_argument('--alt_venv_config', default=None,
                   help='Alt wheel YAML config. When given, TRITON_VERSION in '
                        'the signature becomes a structured object mirroring '
                        'this config, with each venv resolved to its triton '
                        'version. When absent, TRITON_VERSION is a plain string.')
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

def _triton_version_local():
    try:
        return str(importlib.metadata.version("triton"))
    except importlib.metadata.PackageNotFoundError:
        return 'unknown'

def _venv_python(args, name, value):
    # Matches v3python/codegen/root.py venv resolution: a "python:" value points
    # at an explicit interpreter; any other value (including for 'default') is an
    # alt wheel installed under altvenvs/<name>.
    if value.startswith("python:"):
        return Path(value[len("python:"):]).absolute()
    return (args.build_dir.parent / "altvenvs" / name / REL_PYTHON).absolute()

def _default_venv_python(args):
    # Mirror root.py's fallback used only when the config has no 'default' venv:
    # prefer the active VIRTUAL_ENV, else build_dir.parent/venv.
    venv_dir = os.getenv('VIRTUAL_ENV')
    base = Path(venv_dir) if venv_dir else (args.build_dir.parent / 'venv')
    return (base / REL_PYTHON).absolute()

def _triton_version_in_venv(python_path):
    # Query the triton version installed in another venv by running its python.
    try:
        out = subprocess.check_output(
            [str(python_path), '-c',
             'import importlib.metadata as m; print(m.version("triton"))'],
            stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return 'unknown'

def make_triton_version(args):
    # Without an alt wheel config, TRITON_VERSION is a plain string resolved
    # from the venv running this script.
    if not args.alt_venv_config:
        return _triton_version_local()
    # With a config, TRITON_VERSION becomes:
    #   { "default": "<version>",
    #     "rules": [ {"rmatches": {"arch": "gfx125."}, "triton_version": "..."}, ... ] }
    # i.e. the fallback 'default' version up front, then each rule's matcher
    # keys (matches/rmatches/gmatches) with 'venv' replaced by the resolved
    # 'triton_version'.
    with open(args.alt_venv_config) as f:
        cfg = yaml.safe_load(f)
    venvs = cfg.get('venvs', {})
    resolved = {name: _triton_version_in_venv(_venv_python(args, name, value))
                for name, value in venvs.items()}
    if 'default' not in resolved:
        # No 'default' venv in the config: resolve it from VIRTUAL_ENV/venv,
        # mirroring root.py's fallback.
        resolved['default'] = _triton_version_in_venv(_default_venv_python(args))
    out = {'default': resolved.get('default', _triton_version_local())}
    rules_out = []
    for rule in cfg.get('rules', []):
        new_rule = {k: v for k, v in rule.items() if k != 'venv'}
        new_rule['triton_version'] = resolved.get(rule['venv'], 'unknown')
        rules_out.append(new_rule)
    out['rules'] = rules_out
    return out

def main():
    args = parse()
    sig = {}
    sig['AOTRITON_GIT_SHA1'] = args.git_sha1
    sig['AOTRITON_GIT_TREESHA1'] = args.git_treesha1
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
    sig['TRITON_VERSION'] = make_triton_version(args)
    with open(args.output_file, 'w') as f:
        json.dump(sig, f, indent=2)
        print('', file=f)

if __name__ == "__main__":
    main()
