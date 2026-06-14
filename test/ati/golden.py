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
# Fallback DB source (schema tarballs + decomposed per-kernel blobs). The primary
# path is the prebuilt FUSED_DB below; this only fires when that is absent. Points at
# the retired legacy tree (kept on disk, untracked) since the central schema tarballs
# never moved out of it; relocating them is deferred (see to-remove.txt section A).
DB_SRC_DIR = REPO_ROOT / 'v3python.retiring' / 'database'
# Prebuilt fused tuning database (real tuning rows, exercises translate_dataframe).
# Override with AOTRITON_GOLDEN_DB. Falls back to composing the decomposed DB.
import os
FUSED_DB_DIR = Path(os.getenv('AOTRITON_GOLDEN_DB', '/tmp/ati_golden_fused_db'))

# Default arch + the selective targets snapshotted. Kept small but covering both
# The golden covers the FULL flash family (every triton/op/affine target), not a
# subset: the byte-for-byte gate must include the metro launcher and the bare
# debug_simulate_encoded_softmax shim (the ME backend calls the latter directly,
# without an operator). Generation runs with no --selective, mirroring
# v3python.generate's production per-item worker path.
DEFAULT_ARCH = 'gfx942_mod0'

MANIFEST_NAME = 'manifest.json'
# Primary files copied verbatim for readable diffs (relative to build dir).
# Covers a bare kernel shim, the metro-bearing operator, and the bare debug
# kernel the ME backend launches directly.
VERBATIM_FILES = [
    'flash/shim.attn_fwd.h',
    'flash/shim.attn_fwd.cc',
    'flash/iface.op_attn_fwd.h',
    'flash/iface.op_attn_fwd.cc',
    'flash/shim.debug_simulate_encoded_softmax.h',
    'flash/shim.debug_simulate_encoded_softmax.cc',
]


def _venv_python() -> str:
    """The interpreter to drive generation. Prefer the one running this harness."""
    return sys.executable


def _compose_database(build_dir: Path):
    """Populate build_dir/database with the tuning + op databases.

    Prefer the prebuilt FUSED database (real tuning rows, so translate_dataframe
    is exercised). Fall back to composing the decomposed per-kernel sqlite files
    (sparse) only when the fused DB is absent."""
    db_dir = build_dir / 'database'
    db_dir.mkdir(parents=True, exist_ok=True)
    fused_tuning = FUSED_DB_DIR / 'tuning_database.sqlite3'
    fused_op = FUSED_DB_DIR / 'op_database.sqlite3'
    if fused_tuning.exists():
        shutil.copyfile(fused_tuning, db_dir / 'tuning_database.sqlite3')
        if fused_op.exists():
            shutil.copyfile(fused_op, db_dir / 'op_database.sqlite3')
        return
    # Fallback: extract schemas + compose from the decomposed DB.
    for tarxz in DB_SRC_DIR.glob('*.sqlite3.tar.xz'):
        with tarfile.open(tarxz) as tf:
            tf.extractall(db_dir)
    subprocess.run(
        [_venv_python(), '-m', 'aotriton.database_compose',
         '--database_file', str(db_dir / 'tuning_database.sqlite3'),
         '--decomposed', str(DB_SRC_DIR)],
        cwd=REPO_ROOT, check=True,
    )


def _run_generator(build_dir: Path, arch: str):
    """Generate the full family with no --selective, mirroring the production
    per-item worker path (RootGenerator.launch_workers).

    Runs the modular generator (`aotriton.generate`), which is ATI-always — the
    legacy env-switched path is gone, so no AOTRITON_ATI_KERNELS is needed."""
    subprocess.run(
        [_venv_python(), '-m', 'aotriton.generate',
         '--build_dir', str(build_dir),
         '--noimage_mode',
         '--target_gpus', arch],
        cwd=REPO_ROOT, check=True,
    )


def _generate_into(build_dir: Path, arch: str):
    _compose_database(build_dir)
    _run_generator(build_dir, arch)


def _normalize(text: str) -> str:
    """Drop the human-readable-signature comment block before hashing.

    `human_readable_signature` is a non-load-bearing C++ comment (// name = value).
    The legacy and ATI IRs group arguments differently (legacy: one line per
    functional TP, so B/A/grouped-strides; ATI: one line per axis), so the comment
    text differs while all generated CODE is byte-identical. Exact-matching this
    comment is a postati TODO (agent-plans/postati_todo.md); until then it is
    excluded from the golden so the comparison covers only load-bearing output."""
    out = []
    for line in text.splitlines(keepends=True):
        s = line.lstrip()
        # lines like `// <argname> = <value>` within the human-readable block
        if s.startswith('// ') and ' = ' in s and s[3:s.index(' = ')].replace('_', '').isalnum():
            continue
        out.append(line)
    return ''.join(out)


def _manifest_of(build_dir: Path) -> dict[str, str]:
    """sha256 of every generated C++ text file (human-readable comment block
    normalized out), keyed by path relative to build dir. Only .h/.cc count."""
    manifest = {}
    for path in sorted(build_dir.rglob('*')):
        if not path.is_file():
            continue
        if path.suffix not in ('.h', '.cc'):
            continue
        rel = path.relative_to(build_dir).as_posix()
        text = path.read_text(encoding='utf-8', errors='surrogateescape')
        manifest[rel] = hashlib.sha256(
            _normalize(text).encode('utf-8', errors='surrogateescape')).hexdigest()
    return manifest


def _produce(arch: str, keep_dir: Path = None):
    """Generate into a build dir (a temp one unless keep_dir is given); return
    (manifest, build_dir). Caller cleans up a temp build_dir."""
    build_dir = Path(keep_dir) if keep_dir else Path(tempfile.mkdtemp(prefix='ati_golden_'))
    _generate_into(build_dir, arch)
    return _manifest_of(build_dir), build_dir


def update(arch: str, keep_dir: Path = None):
    manifest, build_dir = _produce(arch, keep_dir=keep_dir)
    try:
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        (GOLDEN_DIR / MANIFEST_NAME).write_text(
            json.dumps({'arch': arch, 'files': manifest},
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
        if keep_dir is None:
            shutil.rmtree(build_dir, ignore_errors=True)
    return 0


def check(arch: str, keep_dir: Path = None):
    golden_path = GOLDEN_DIR / MANIFEST_NAME
    if not golden_path.exists():
        print(f'No golden snapshot at {golden_path}; run --update first.',
              file=sys.stderr)
        return 2
    golden = json.loads(golden_path.read_text())
    fresh, build_dir = _produce(arch, keep_dir=keep_dir)
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
        if keep_dir is None:
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
    p.add_argument('--keep_dir', type=Path, default=None,
                   help='Generate into this dir and keep it (full tree on disk '
                        'for file-level diffing) instead of a temp dir.')
    args = p.parse_args()
    if args.update:
        return update(args.arch, keep_dir=args.keep_dir)
    return check(args.arch, keep_dir=args.keep_dir)


if __name__ == '__main__':
    sys.exit(main())
