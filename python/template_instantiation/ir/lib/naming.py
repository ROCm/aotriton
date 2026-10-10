# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The hsaco-entry-name grammar + its hash — the two things every language's
ksignature.py must agree on, factored out of the Triton-only implementation
so a second language does not have to reverse-engineer the format from
ir/triton/ksignature.py.

`entry_name` takes the functional directly (reading `.unified_signature` and
`.arch` off it) rather than having each caller destructure it first; `perf` /
`copt` stay caller-assembled strings, in whatever vocabulary their own
language uses, since ir/lib/ has no notion of either.
"""

import hashlib


def entry_name(functional: 'Functional', perf: str = '', copt: str = '') -> str:
    """The hsaco entry-name grammar:

        ;;#F;<unified_signature>;;#P;<perf>;;#CO;<copt>;;arch=<arch>

    `unified_signature` and `arch` are read off `functional`. `perf` and
    `copt` default to '' for languages/callers with no perf or
    compiler-option section to report.
    """
    return (
        f';;#F;{functional.unified_signature}'
        f';;#P;{perf}'
        f';;#CO;{copt}'
        f';;arch={functional.arch}'
    )


def blake2b_hash(package_path: str, entry: str):
    """blake2b-8 digest of `package_path + entry`. Returns (hexdigest, raw_bytes)."""
    raw = (package_path + entry).encode('utf-8')
    h = hashlib.blake2b(raw, digest_size=8)
    return h.hexdigest(), raw
