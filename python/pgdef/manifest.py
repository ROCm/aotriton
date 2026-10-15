# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Printing an application's MANIFEST for shell scripts.

Per-application shell knowledge (where its build is installed, what to fetch
from the build node, which host role runs it) lives in one
place: the application's `MANIFEST` dict, in `aotriton.<app>.manifest`. Scripts
ask for one key at a time:

    python3 -m aotriton.tune.manifest deploy.role
    python3 -m aotriton.tune.manifest install.subtree --arch gfx950 --cat_level1 op

This module is the shared lookup, not a command: pgdef has no manifest CLI of
its own, and never reads a manifest it was not handed. Each application's
manifest module calls `main(MANIFEST)`.

A manifest is a nested dict of strings, lists and dicts (a Python dict rather
than TOML because this project targets Python 3.10, which has no tomllib; and
not config.rc, which is for the tuning infrastructure only). A dict whose keys
are category levels, with the sentinel key `*` for "any", is selected by
`--cat_level1`. Placeholders `{arch}` and `{cat_level1}` in strings are filled
from the command line.

Output: a string prints as is, a list one item per line, anything else as JSON.
A missing key exits with status 1 and a message on stderr.
"""

import argparse
import json
import sys

ANY = '*'


def lookup(manifest: dict, key: str, *, cat_level1: str | None = None, **fmt):
    """The value at the dotted `key`, with per-level dicts resolved and
    `{placeholders}` formatted. Raises KeyError for a missing key."""
    node = manifest
    for part in key.split('.'):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(key)
        node = node[part]
    return _resolve(node, cat_level1, {'cat_level1': cat_level1 or '', **fmt})


def _resolve(node, cat_level1, fmt):
    if isinstance(node, dict) and cat_level1 is not None and (cat_level1 in node or ANY in node):
        node = node[cat_level1] if cat_level1 in node else node[ANY]
    if isinstance(node, str):
        return node.format(**fmt)
    if isinstance(node, list):
        return [_resolve(item, cat_level1, fmt) for item in node]
    if isinstance(node, dict):
        return {k: _resolve(v, cat_level1, fmt) for k, v in node.items()}
    return node


def render(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return '\n'.join(value)
    return json.dumps(value, sort_keys=True)


def main(manifest: dict, argv: list[str] | None = None, *, prog: str | None = None) -> int:
    parser = argparse.ArgumentParser(prog=prog, description='Print one key of the application manifest.')
    parser.add_argument('key', help="dotted key, e.g. deploy.role")
    parser.add_argument('--arch', default='', help='fills {arch}')
    parser.add_argument('--cat_level1', default=None, help='selects the per-level value and fills {cat_level1}')
    args = parser.parse_args(argv)
    try:
        value = lookup(manifest, args.key, cat_level1=args.cat_level1, arch=args.arch)
    except KeyError:
        print(f"manifest has no key {args.key!r}", file=sys.stderr)
        return 1
    print(render(value))
    return 0
