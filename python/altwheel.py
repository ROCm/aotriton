# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Parse the resolved altwheel config (docs/AltWheelExample.yaml).

The one parser of `venvs.*` entries: codegen/root.py imports it, and
CMakeLists.txt runs it as `python -m aotriton.altwheel`, so the two readers
cannot disagree on what an entry means.
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import PurePath

import yaml

PYTHON_PREFIX = 'python:'
PEP508_PREFIX = 'pep508:'

# Names become directories (altvenvs/<name>) and a CMake list.
_NAME_RE = re.compile(r'[A-Za-z0-9_.+-]+')


@dataclass(frozen=True)
class VenvEntry:
    name: str
    # 'wheel': value is a wheel path; installed into a fresh venv.
    # 'python': value is an existing interpreter; nothing is installed.
    # 'pep508': value is a pip requirement; installed into a fresh venv.
    mode: str
    value: str


def parse_entry(name, raw) -> VenvEntry:
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise ValueError(f'altwheel venv name {name!r} must match {_NAME_RE.pattern}')
    if not isinstance(raw, str):
        raise ValueError(f'altwheel venv {name!r}: expected a string, got {raw!r}')
    if raw.startswith(PYTHON_PREFIX):
        python = raw[len(PYTHON_PREFIX):]
        # The two readers run in different directories, so relative is ambiguous.
        if not PurePath(python).is_absolute():
            raise ValueError(f'altwheel venv {name!r}: {PYTHON_PREFIX} needs an absolute '
                             f'interpreter path, got {python!r}')
        return VenvEntry(name, 'python', python)
    if raw.startswith(PEP508_PREFIX):
        req = raw[len(PEP508_PREFIX):].strip()
        from packaging.requirements import Requirement, InvalidRequirement
        try:
            Requirement(req)
        except InvalidRequirement as e:
            raise ValueError(f'altwheel venv {name!r}: invalid PEP 508 requirement '
                             f'{req!r}: {e}') from e
        return VenvEntry(name, 'pep508', req)
    if not raw.endswith('.whl'):
        raise ValueError(f'altwheel venv {name!r}: a bare entry must be a .whl path, got '
                         f'{raw!r}. Write a pip requirement as "{PEP508_PREFIX}{raw}".')
    return VenvEntry(name, 'wheel', raw)


def parse_venvs(d: dict) -> dict[str, VenvEntry]:
    venvs = d.get('venvs') or {}
    return {name: parse_entry(name, raw) for name, raw in venvs.items()}


def load(path) -> dict:
    with open(path) as f:
        return yaml.load(f, Loader=yaml.Loader) or {}


def parse():
    p = argparse.ArgumentParser(description='Read the resolved altwheel config. '
                                            'Without --venv, print every venv name as a CMake list.')
    p.add_argument('config', help='resolved altwheel YAML')
    p.add_argument('--venv', default=None, help='print one field of this venv entry')
    p.add_argument('--field', choices=['mode', 'value'], default='value')
    return p.parse_args()


def main():
    args = parse()
    try:
        venvs = parse_venvs(load(args.config))
    except ValueError as e:
        sys.exit(f'{args.config}: {e}')
    if args.venv is None:
        print(';'.join(venvs), end='')
        return
    if args.venv not in venvs:
        sys.exit(f'{args.config}: no venvs.{args.venv} entry')
    print(getattr(venvs[args.venv], args.field), end='')


if __name__ == '__main__':
    main()
