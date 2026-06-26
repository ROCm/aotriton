#!/usr/bin/env python
# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Triton installation tracking for the AOTriton build.

The gate is *stateless*: it compares the wheel against the files already
installed in the target venv, in three layers of increasing cost:

  1. <name>.dist-info DIRECTORY NAME -- e.g. triton-3.8.0+git359bd664.dist-info.
     pip preserves this name verbatim from the wheel, and it encodes the PEP 440
     version (incl. the git hash for git builds). A cheap string compare that
     flags a version change WITHOUT touching METADATA or the .so.
  2. <name>.dist-info/METADATA       -- a small text file, so we compare it BYTE
     FOR BYTE (no hashing). Carries the full version/provenance and catches any
     metadata difference at the same version label.
  3. triton/_C/libtriton.{so,pyd}    -- the compiler binary, hashed last because
     it is large. Catches binary differences between two builds that share both
     a version string and identical METADATA (e.g. non-git rebuilds).

Two subcommands:

  sync   Reinstall the wheel iff ANY layer differs from what is installed.
         A version change short-circuits on layer 1; matching name + METADATA
         still verifies layer 3 (catches same-version rebuilds).

  stamp  Write a content stamp (SHA-256 of the installed .so + METADATA) to a
         file, rewriting it ONLY when the fingerprint changes. Downstream CMake
         rules depend on this stamp so the kernel signature and HSACO
         compilation are invalidated exactly when Triton changes.

The module is dependency-free (stdlib only) and does NOT import ``triton``: it
must run before Triton is (re)installed.
"""

import argparse
import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path

# Relative path of the Triton C extension inside a wheel and inside an installed
# site-packages tree.  Platform decides which one is present.
_WHEEL_SO_MEMBERS = ('triton/_C/libtriton.so', 'triton/_C/libtriton.pyd')

_CHUNK = 1024 * 1024


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_file(path) -> str | None:
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    m = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(_CHUNK):
            m.update(chunk)
    return m.hexdigest()


def _short(h) -> str:
    return 'none' if h is None else h[:12]


# ---- layer 1: dist-info directory name -------------------------------------

def wheel_dist_info_name(wheel) -> str | None:
    """The '<name>-<version>.dist-info' directory name carried in the wheel."""
    with zipfile.ZipFile(wheel) as zf:
        for n in zf.namelist():
            if n.endswith('.dist-info/METADATA'):
                return n.rsplit('/', 1)[0]
    return None


def installed_dist_info_name(installed_so) -> str | None:
    """The installed Triton '<name>-<version>.dist-info' directory name.

    The dist-info sits beside the triton/ package; the 'triton-' prefix (with a
    dash) excludes siblings such as triton_kernels whose normalized dist-info
    name uses an underscore.
    """
    so = Path(installed_so).resolve()
    if len(so.parents) < 3:
        return None
    site = so.parents[2]
    matches = sorted(p.name for p in site.glob('triton-*.dist-info'))
    return matches[0] if matches else None


# ---- layer 2: METADATA (compared byte for byte) ----------------------------

def wheel_metadata_bytes(wheel) -> bytes | None:
    with zipfile.ZipFile(wheel) as zf:
        for n in zf.namelist():
            if n.endswith('.dist-info/METADATA'):
                return zf.read(n)
    return None


def _installed_metadata_path(installed_so):
    name = installed_dist_info_name(installed_so)
    if name is None:
        return None
    return Path(installed_so).resolve().parents[2] / name / 'METADATA'


def installed_metadata_bytes(installed_so) -> bytes | None:
    """Installed METADATA, or None for source/editable installs (.egg-info)."""
    p = _installed_metadata_path(installed_so)
    if p is None or not p.is_file():
        return None
    return p.read_bytes()


# ---- layer 3: .so content --------------------------------------------------

def wheel_so_hash(wheel) -> str:
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
        nameset = set(names)
        for member in _WHEEL_SO_MEMBERS:
            if member in nameset:
                return _hash_bytes(zf.read(member))
        # Fall back to any member that looks like the C extension, in case the
        # wheel layout shifts (e.g. an ABI tag in the filename).
        for n in names:
            base = n.rsplit('/', 1)[-1]
            if '/_C/' in n and base.startswith('libtriton.'):
                return _hash_bytes(zf.read(n))
    raise SystemExit(f'[track_triton] no libtriton shared object found in wheel {wheel}')


# ---- subcommands -----------------------------------------------------------

def _pip_install(args) -> None:
    env = dict(os.environ)
    if args.virtual_env:
        env['VIRTUAL_ENV'] = args.virtual_env
    # --force-reinstall --no-deps: swap exactly the Triton package content even
    # when the version string is unchanged (rebuilt wheel); deps were already
    # resolved by the requirements.txt / first install.
    cmd = [args.pip_python, '-m', 'pip', 'install',
           '--force-reinstall', '--no-deps', args.wheel]
    subprocess.run(cmd, env=env, check=True)


def _verify_install(args) -> None:
    if (wheel_dist_info_name(args.wheel) != installed_dist_info_name(args.installed_so)
            or wheel_metadata_bytes(args.wheel) != installed_metadata_bytes(args.installed_so)
            or wheel_so_hash(args.wheel) != _hash_file(args.installed_so)):
        raise SystemExit(
            f'[track_triton] post-install mismatch at {args.installed_so}: '
            f'installed Triton still differs from {args.wheel}')


def cmd_sync(args) -> int:
    # Layer 1 (cheapest): the .dist-info directory name encodes the version. A
    # mismatch means a different version is installed -- reinstall without
    # reading METADATA or decompressing the large .so.
    w_name = wheel_dist_info_name(args.wheel)
    i_name = installed_dist_info_name(args.installed_so)
    if w_name != i_name:
        print(f'[track_triton] installing {args.wheel}: '
              f'dist-info {i_name or "none"} != {w_name or "none"}')
        _pip_install(args)
        _verify_install(args)
        return 0

    # Layer 2: METADATA is small -> compare byte for byte (no hashing).
    w_md = wheel_metadata_bytes(args.wheel)
    i_md = installed_metadata_bytes(args.installed_so)
    if w_md != i_md:
        print(f'[track_triton] installing {args.wheel}: METADATA differs')
        _pip_install(args)
        _verify_install(args)
        return 0

    # Layer 3 (most expensive, last): same version + identical METADATA ->
    # confirm the binary itself matches (catches same-version non-git rebuilds).
    w_so = wheel_so_hash(args.wheel)
    i_so = _hash_file(args.installed_so)
    if w_so != i_so:
        print(f'[track_triton] installing {args.wheel}: so {w_so[:12]} != {_short(i_so)}')
        _pip_install(args)
        _verify_install(args)
        return 0

    print(f'[track_triton] {args.installed_so} matches {args.wheel} '
          f'(dist-info {w_name}, METADATA identical, so {w_so[:12]}); skipping install')
    return 0


def cmd_stamp(args) -> int:
    parts = []
    for so in args.installed_so:
        s_h = _hash_file(so)
        if s_h is None:
            print(f'[track_triton] warning: {so} does not exist, skipping',
                  file=sys.stderr)
            continue
        parts.append(s_h)
        md = installed_metadata_bytes(so)
        parts.append(_hash_bytes(md) if md is not None else '')  # may be absent
    combined = _hash_bytes('\n'.join(parts).encode())
    stamp = Path(args.stamp)
    old = stamp.read_text().strip() if stamp.is_file() else None
    if old == combined:
        print(f'[track_triton] stamp unchanged ({combined[:12]})')
        return 0
    stamp.write_text(combined + '\n')
    print(f'[track_triton] stamp updated -> {combined[:12]}')
    return 0


def parse():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)

    s = sub.add_parser('sync', help='reinstall Triton wheel iff dist-info name, METADATA, or .so differs')
    s.add_argument('--wheel', required=True, help='Triton wheel to install')
    s.add_argument('--pip_python', required=True, help='python interpreter used to run pip')
    s.add_argument('--virtual_env', default=None, help='VIRTUAL_ENV for the pip install')
    s.add_argument('--installed_so', required=True, help='path to the installed libtriton shared object')
    s.set_defaults(func=cmd_sync)

    t = sub.add_parser('stamp', help='write a content stamp of the installed Triton .so(s) + METADATA')
    t.add_argument('--stamp', required=True, help='stamp file to write')
    t.add_argument('--installed_so', nargs='+', required=True, help='installed libtriton shared object(s)')
    t.set_defaults(func=cmd_stamp)

    return p.parse_args()


def main():
    args = parse()
    sys.exit(args.func(args))


if __name__ == '__main__':
    main()
