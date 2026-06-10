# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Golden codegen snapshot harness (ATI executive plan Step 0.1).

Captures the C++ that the *current* generator emits for the flash family and
lets every later ATI step prove generation-output-neutrality byte-for-byte.

Usage:
    python -m test.ati.golden --update   # (re)write the committed snapshot
    python -m test.ati.golden --check    # regenerate and diff against snapshot

The snapshot is a manifest (relative path -> sha256) plus verbatim copies of the
primary shim/op files for human-readable diffs. Generation is deterministic: it
is driven entirely by the checked-in tuning database (no HSACO compilation, via
--noimage_mode), so identical inputs yield identical bytes.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_DIR = Path(__file__).resolve().parent / 'golden'
DB_SRC_DIR = REPO_ROOT / 'v3python' / 'database'

# Default arch + the selective targets snapshotted. Kept small but covering both
# a Triton kernel shim and an operator (the two byte-for-byte gates referenced by
# executive plan Steps 4.2 and 5.2). Broaden as more kernels are ported.
DEFAULT_ARCH = 'gfx942_mod0'
DEFAULT_TARGETS = [
    'flash/triton/attn_fwd',
    'flash/op/op_attn_fwd',
]

MANIFEST_NAME = 'manifest.json'
# Primary files copied verbatim for readable diffs (relative to build dir).
VERBATIM_FILES = [
    'flash/shim.attn_fwd.h',
    'flash/shim.attn_fwd.cc',
    'flash/iface.op_attn_fwd.h',
    'flash/iface.op_attn_fwd.cc',
]


def _venv_python() -> str:
    """The interpreter to drive generation. Prefer the one running this harness."""
    return sys.executable


def _compose_database(build_dir: Path):
    """Reproduce v3src/CMakeLists.txt 'Assembling Central Database': extract every
    *.sqlite3.tar.xz schema (preserving relative path) then compose the central
    tuning database from the decomposed per-kernel sqlite files."""
    db_dir = build_dir / 'database'
    db_dir.mkdir(parents=True, exist_ok=True)
    for tarxz in DB_SRC_DIR.glob('*.sqlite3.tar.xz'):
        with tarfile.open(tarxz) as tf:
            tf.extractall(db_dir)
    subprocess.run(
        [_venv_python(), '-m', 'v3python.database_compose',
         '--database_file', str(db_dir / 'tuning_database.sqlite3'),
         '--decomposed', str(DB_SRC_DIR)],
        cwd=REPO_ROOT, check=True,
    )


def _run_generator(build_dir: Path, arch: str, targets: list[str]):
    for target in targets:
        subprocess.run(
            [_venv_python(), '-m', 'v3python.generate',
             '--selective', target,
             '--build_dir', str(build_dir),
             '--noimage_mode',
             '--target_gpus', arch],
            cwd=REPO_ROOT, check=True,
        )


def _generate_into(build_dir: Path, arch: str, targets: list[str]):
    _compose_database(build_dir)
    _run_generator(build_dir, arch, targets)


def _manifest_of(build_dir: Path) -> dict[str, str]:
    """sha256 of every generated C++ text file, keyed by path relative to build
    dir. The database/ and any intermediate dirs are excluded; only .h/.cc count."""
    manifest = {}
    for path in sorted(build_dir.rglob('*')):
        if not path.is_file():
            continue
        if path.suffix not in ('.h', '.cc'):
            continue
        rel = path.relative_to(build_dir).as_posix()
        manifest[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest


def _produce(arch: str, targets: list[str]):
    """Generate into a temp build dir; return (manifest, build_dir). Caller must
    clean up build_dir."""
    build_dir = Path(tempfile.mkdtemp(prefix='ati_golden_'))
    _generate_into(build_dir, arch, targets)
    return _manifest_of(build_dir), build_dir


def update(arch: str, targets: list[str]):
    manifest, build_dir = _produce(arch, targets)
    try:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        (GOLDEN_DIR / MANIFEST_NAME).write_text(
            json.dumps({'arch': arch, 'targets': targets, 'files': manifest},
                       indent=2, sort_keys=True) + '\n')
        verbatim_dir = GOLDEN_DIR / 'files'
        if verbatim_dir.exists():
            shutil.rmtree(verbatim_dir)
        verbatim_dir.mkdir(parents=True)
        for rel in VERBATIM_FILES:
            src = build_dir / rel
            if not src.exists():
                continue
            dst = verbatim_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        print(f'Wrote golden snapshot: {len(manifest)} files -> {GOLDEN_DIR}')
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)
    return 0


def check(arch: str, targets: list[str]):
    golden_path = GOLDEN_DIR / MANIFEST_NAME
    if not golden_path.exists():
        print(f'No golden snapshot at {golden_path}; run --update first.',
              file=sys.stderr)
        return 2
    golden = json.loads(golden_path.read_text())
    fresh, build_dir = _produce(arch, targets)
    try:
        expected = golden['files']
        missing = sorted(set(expected) - set(fresh))
        added = sorted(set(fresh) - set(expected))
        changed = sorted(k for k in expected.keys() & fresh.keys()
                         if expected[k] != fresh[k])
        if not (missing or added or changed):
            print(f'OK: {len(fresh)} generated files match golden snapshot.')
            return 0
        for k in missing:
            print(f'MISSING (in golden, not regenerated): {k}')
        for k in added:
            print(f'ADDED   (regenerated, not in golden):  {k}')
        for k in changed:
            print(f'CHANGED (content differs):             {k}')
        print(f'\nFAIL: {len(missing)} missing, {len(added)} added, '
              f'{len(changed)} changed.', file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Golden codegen snapshot harness for the flash family.')
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--update', action='store_true',
                      help='Regenerate and overwrite the committed snapshot.')
    mode.add_argument('--check', action='store_true',
                      help='Regenerate and diff against the committed snapshot.')
    p.add_argument('--arch', type=str, default=DEFAULT_ARCH,
                   help='Target GPU arch passed to --target_gpus.')
    p.add_argument('--targets', type=str, nargs='+', default=DEFAULT_TARGETS,
                   help='Selective generation targets to snapshot.')
    args = p.parse_args()
    if args.update:
        return update(args.arch, args.targets)
    return check(args.arch, args.targets)


if __name__ == '__main__':
    sys.exit(main())
