# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit test for union_params order-preserving merge (executive plan Step 5.1)."""

import sys
from pathlib import Path
from graphlib import CycleError

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tritonsrc'))

from aotriton.template_instantiation.ops import union_params


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


def test_real_bwd_kernels_merge_without_hint():
    # The actual tritonsrc bwd kernels: dk_dv (DO,DK,DV,D) + dq (DO,DQ,DB,D). They
    # merge to DK,DV,DQ,DB with NO order_hint — the metro call order alone suffices.
    from bwd_kernel_dk_dv import bwd_kernel_dk_dv
    from bwd_kernel_dq import bwd_kernel_dq
    dkv = [p.name for p in bwd_kernel_dk_dv.params]
    dq = [p.name for p in bwd_kernel_dq.params]
    merged = union_params([dkv, dq])             # metro call order: dk_dv, then dq
    assert len(merged) == len(set(merged))       # no duplicates
    i = {n: merged.index(n) for n in ('DO', 'DK', 'DV', 'DQ', 'DB', 'D')}
    assert i['DO'] < i['DK'] < i['DV'] < i['DQ'] < i['DB'] < i['D']


def test_adjacent_reversal_first_kernel_wins():
    # A shared block appears in different orders across sub-kernels; the earlier
    # (first-listed) kernel's order wins, no error.
    k1 = ['x', 'y', 'z', 'k1a', 'k1b']
    k2 = ['z', 'y', 'x', 'k2a', 'k2b']
    assert union_params([k1, k2]) == ['x', 'y', 'z', 'k1a', 'k1b', 'k2a', 'k2b']


def test_contradictory_pair_first_wins_no_error():
    # The simplest reversal: a says X<Y, b says Y<X -> a direct transposition, so
    # first kernel wins (X,Y); the contradicting edge is dropped, no error.
    assert union_params([['X', 'Y'], ['Y', 'X']]) == ['X', 'Y']


def test_nonadjacent_cycle_raises_without_hint():
    # Shared params NOT adjacent: k1=[x,k1p,z] (x<z), k2=[z,k2p,x] (z<x). The cycle
    # closes through private params (a longer path), not a direct transposition, so
    # it is a genuine conflict the author must resolve -> CycleError.
    try:
        union_params([['x', 'k1p', 'z'], ['z', 'k2p', 'x']])
    except CycleError:
        return
    raise AssertionError('expected CycleError on non-adjacent cycle')


def test_nonadjacent_cycle_resolved_by_hint():
    # The same conflict, resolved: order_hint pins x before z and takes
    # responsibility, so contradicting edges are dropped instead of raising.
    merged = union_params([['x', 'k1p', 'z'], ['z', 'k2p', 'x']],
                          order_hint=['x', 'z'])
    assert merged.index('x') < merged.index('z')
    assert set(merged) == {'x', 'k1p', 'z', 'k2p'}


def test_order_hint_forces_order():
    # order_hint is the highest-priority chain: it overrides both kernels' order.
    a = ['X', 'Y']
    b = ['X', 'Y']
    assert union_params([a, b], order_hint=['Y', 'X']) == ['Y', 'X']


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
