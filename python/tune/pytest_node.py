# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Family-neutral pytest node-ID parsing.

A node ID is `<path>::<test>[<p0>-<p1>-...]`. Splitting it is pure string
work and identical for every family, so it lives here. What the bracket
parameters *mean* is not: their order, their spelling and the axes they map
onto are properties of one family's test parametrization, so translating a
ParsedNode into a tuning entry belongs to that family's tune block (see
`entry_from_pytest_node` in modules/<family>/tune/).

    >>> node = parse_node_id(
    ...     'modules/flash/tests/test_backward.py::test_regular_bwd'
    ...     '[Split-False-l1-dtype2-0.0-CausalOff-256-8192-hdim8-5-3]')
    >>> node.family, node.test, node.params[3]
    ('flash', 'test_regular_bwd', 'dtype2')
"""

import re
from dataclasses import dataclass, field

_NODE_RE = re.compile(r'^(?P<path>[^:]+)::(?P<test>\w+)\[(?P<params>.*)\]$')
_FAMILY_RE = re.compile(r'(?:^|/)modules/(?P<family>[^/]+)/')


@dataclass(frozen=True)
class ParsedNode:
    """One pytest node ID, split but not interpreted.

    params is positional on purpose: pytest renders parametrize arguments in
    declaration order with no names, so position is the only thing the node ID
    actually carries. Naming them requires knowing the test's signature, which
    is exactly the family-specific knowledge this class does not have.
    """
    path: str
    test: str
    params: tuple[str, ...] = field(default_factory=tuple)

    @property
    def family(self) -> str | None:
        """Family from the path, e.g. 'flash' for modules/flash/tests/x.py.

        None when the path is not under modules/<family>/, which is how a
        caller can tell "not a family test" from "a family I cannot load".
        """
        m = _FAMILY_RE.search(self.path)
        return m.group('family') if m else None

    def require(self, count: int) -> None:
        """Raise unless exactly `count` parameters are present.

        Family translators index positionally, so an unexpected parametrize
        change should fail with the counts rather than an IndexError.

        Exact, not a minimum: adding one parametrize decorator to a test
        shifts every position by one, and a `>=` check would let that decode
        silently -- reading sm_scale as dtype, one seqlen as the other -- and
        resolve a real but wrong tuning entry. Too many parameters is just as
        much a layout mismatch as too few.
        """
        if len(self.params) != count:
            raise ValueError(
                f'Expected exactly {count} bracket parameters for '
                f'{self.test}, got {len(self.params)}: '
                f'{"-".join(self.params)}')


def parse_node_id(node_id: str) -> ParsedNode:
    """Split a pytest node ID into path, test name and positional parameters.

    Parameters are split on '-', which is pytest's own separator; a family
    whose parametrize values contain '-' would need its own splitting, and
    none currently does.
    """
    m = _NODE_RE.match(node_id.strip())
    if not m:
        raise ValueError(
            'Could not parse pytest node ID -- expected '
            'path::test_name[params]')
    params = m.group('params')
    return ParsedNode(path=m.group('path'), test=m.group('test'),
                      params=tuple(params.split('-')) if params else ())
