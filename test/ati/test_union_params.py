# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit test for union_params order-preserving merge (executive plan Step 5.1)."""

import sys
from pathlib import Path
from graphlib import CycleError

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v3python.template_instantiation.operator import union_params


def test_shared_spine_simple():
    # Two chains sharing Q, Out — interleave deterministically.
    a = ['Q', 'K', 'Out']
    b = ['Q', 'V', 'Out']
    assert union_params([a, b]) == ['Q', 'K', 'V', 'Out']


def test_bwd_interleave_dk_dv_dq_db():
    # The canonical bwd metro (ati_rev1.md §4.2): the metro calls dk_dv (DK,DV)
    # before dq (DQ,DB); they must group as (...,DK,DV,DQ,DB,...), not concatenate.
    dk_dv = ['Q', 'K', 'V', 'Out', 'DO', 'DK', 'DV', 'L']   # metro call order: 0
    dq    = ['Q', 'K', 'V', 'Out', 'DO', 'DQ', 'DB', 'L']   # metro call order: 1
    merged = union_params([dk_dv, dq])
    # DK,DV (sub-kernel 0) come before DQ,DB (sub-kernel 1), all in the same gap.
    i = {n: merged.index(n) for n in ('DK', 'DV', 'DQ', 'DB', 'DO', 'L')}
    assert i['DO'] < i['DK'] < i['DV'] < i['DQ'] < i['DB'] < i['L']


def test_contradictory_order_raises_cycle():
    a = ['X', 'Y']
    b = ['Y', 'X']
    try:
        union_params([a, b])
    except CycleError:
        return
    raise AssertionError('expected CycleError on contradictory order')


def test_order_hint_resolves_cycle():
    a = ['X', 'Y']
    b = ['Y', 'X']
    # Pin X before Y to break the cycle.
    assert union_params([a, b], order_hint=['X', 'Y']) == ['X', 'Y']


def test_rename_merges_wired_operands():
    # Kernel 'main' has arg 'Out'; debug kernel has arg 'R' wired to operand
    # 'encoded_softmax'. Without rename they'd be two nodes; with the rename
    # 'R'->'encoded_softmax' they participate as the operand name.
    main = ['Q', 'Out', 'encoded_softmax']
    debug = ['Q', 'R']
    renames = [None, {'R': 'encoded_softmax'}]
    merged = union_params([main, debug], renames=renames)
    assert merged.count('encoded_softmax') == 1     # merged, not duplicated
    assert 'R' not in merged                          # renamed away
    assert merged == ['Q', 'Out', 'encoded_softmax']


def test_three_way_merge():
    a = ['Q', 'A', 'Z']
    b = ['Q', 'B', 'Z']
    c = ['Q', 'C', 'Z']
    merged = union_params([a, b, c])
    assert merged == ['Q', 'A', 'B', 'C', 'Z']


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} union_params tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
