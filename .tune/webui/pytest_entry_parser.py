# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Turn a pytest node ID into a task_queue lookup.

The work is split in two and neither half lives here any more:

  aotriton.tune.pytest_node          splits `path::test[p0-p1-...]` into a
                                     ParsedNode. Pure string work, identical
                                     for every family.
  modules/<family>/tune/pytest_entry translates that ParsedNode into the
                                     family's entry fields. Parameter order,
                                     token spellings and the tuning-database
                                     axes are all family-specific.

The family comes from the node's own path (`modules/<family>/tests/...`), so
this module dispatches without being told which family it is looking at.

Usage as CLI:
  python pytest_entry_parser.py \
      "modules/flash/tests/test_backward.py::test_regular_bwd[Split-False-l1-dtype2-0.0-CausalOff-256-8192-hdim8-5-3]"
"""

import json
import sys
import argparse

from aotriton.tune.pytest_node import parse_node_id
from aotriton.tune.pq.queue import entry_filter
from aotriton.tune.registry import load_family_tune


def parse_pytest_node_id(node_id: str) -> dict:
    """Parse a pytest node ID into the owning family's entry fields.

    Raises ValueError with a human-readable message on any failure: an
    unparseable ID, a path outside modules/<family>/, a family with no tune
    block, or a test that family does not know how to translate.
    """
    node = parse_node_id(node_id)
    family = node.family
    if family is None:
        raise ValueError(
            f'Cannot tell which family {node.path!r} belongs to -- '
            f'expected a path under modules/<family>/.')
    try:
        tune = load_family_tune(family)
    except Exception as e:
        raise ValueError(str(e)) from None
    translate = getattr(tune, 'pytest_entry', None)
    if translate is None:
        raise ValueError(
            f'Family {family!r} does not translate pytest node IDs '
            f'(no pytest_entry module in its tune block).')
    return translate.entry_from_pytest_node(node)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Parse a pytest node ID and show the equivalent FlashEntry fields.',
        epilog=(
            'Example:\n'
            '  %(prog)s '
            '"modules/flash/tests/test_backward.py::test_regular_bwd'
            '[Split-False-l1-dtype2-0.0-CausalOff-256-8192-hdim8-5-3]"'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'node_id',
        help='Pytest node ID string (e.g. modules/flash/tests/test_backward.py::test_name[params])',
    )
    parser.add_argument(
        '--json',
        action='store_true',
        dest='as_json',
        help='Output as JSON instead of human-readable text',
    )
    args = parser.parse_args()

    try:
        entry = parse_pytest_node_id(args.node_id)
    except ValueError as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)

    if args.as_json:
        # tuples are not JSON-serializable natively
        serialisable = {
            k: list(v) if isinstance(v, tuple) else v
            for k, v in entry.items()
        }
        print(json.dumps(serialisable, indent=2))
    else:
        print('Parsed FlashEntry fields:')
        for k, v in entry.items():
            print(f'  {k} = {v!r}')
        print()
        sql, params = entry_filter(entry)
        print('SQL WHERE clause (no arch filter -- pytest IDs do not encode arch):')
        print(f'  {sql}')
        print(f'  params = {params!r}')


if __name__ == '__main__':
    main()
