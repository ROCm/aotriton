# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
union_params — order-preserving merge of a METRO KERNEL's sub-kernel argument
lists (executive plan Step 5.1; agent-plans/ati_rev1.md §4.2).

A metro kernel implements one operator functional with a *set of collaborating
kernels* (fwd: attn_fwd + debug; bwd: preprocess + dk_dv + dq). Each sub-kernel's
argument list is a totally-ordered chain; their union is a DAG and the merged
order is a stable topological sort (graphlib.TopologicalSorter). The tiebreak is
the **sub-kernel's index in the metro call sequence** (collaboration order) then
position — so e.g. dk_dv (called first) contributes DK,DV before dq's DQ,DB.

Shared arguments stitch the chains together; a contradictory order is a cycle ->
CycleError, a hard error the author resolves with `order_hint` (which PINS a
relative order, overriding the contradicting edges).

Rename-aware: the metro DSL wires sub-kernel arguments to operator operands (the
transpiler's by-name + kwarg renames). Pass per-sub-kernel rename maps
{kernel_arg: operand} so wired-together arguments merge into a single node — the
merge runs over the RENAMED (operand) names.
"""

from graphlib import TopologicalSorter


def _renamed(args, rename):
    """Translate a kernel's argument list through its rename map (identity for
    args not in the map)."""
    if not rename:
        return list(args)
    return [rename.get(a, a) for a in args]


def union_params(arg_lists, *, renames=None, order_hint=None):
    """Merge a metro kernel's sub-kernel argument lists into one order-preserving
    sequence.

    arg_lists:  list of per-sub-kernel argument-name sequences, IN METRO CALL
                ORDER (each a chain). The list index is the collaboration order
                tiebreak.
    renames:    optional list of per-sub-kernel {kernel_arg: operand} maps, aligned
                with arg_lists; the merge runs over the renamed (operand) names so
                wired operands collapse to one node.
    order_hint: optional sequence pinning a relative order; it OVERRIDES any
                contradicting chain edges among the hinted nodes (so it can break a
                cycle, not merely add to it).

    Returns the merged list of (operand) names. Raises graphlib.CycleError on a
    contradictory ordering not resolved by order_hint.
    """
    renames = renames or [None] * len(arg_lists)
    assert len(renames) == len(arg_lists), 'renames must align with arg_lists'

    succ = {}       # node -> set(successors)
    pos = {}        # node -> (subkernel_index, position) tiebreak key
    for ki, (args, rename) in enumerate(zip(arg_lists, renames)):
        chain = _renamed(args, rename)
        for i, a in enumerate(chain):
            pos.setdefault(a, (ki, i))          # first mention owns the tiebreak
            if i + 1 < len(chain):
                succ.setdefault(a, set()).add(chain[i + 1])

    if order_hint:
        # The hint pins the relative order of its nodes. Remove any chain edge that
        # contradicts it (b->a where the hint says a before b), then add the hint's
        # own consecutive edges. This is what lets a hint BREAK a cycle.
        hinted = set(order_hint)
        rank = {n: r for r, n in enumerate(order_hint)}
        for a, outs in list(succ.items()):
            for b in list(outs):
                if a in hinted and b in hinted and rank[a] > rank[b]:
                    outs.discard(b)             # contradicts the hint -> drop
        for a, b in zip(order_hint, order_hint[1:]):
            succ.setdefault(a, set()).add(b)
            pos.setdefault(a, (len(arg_lists), 0))
        pos.setdefault(order_hint[-1], (len(arg_lists), 0))

    preds = {n: set() for n in pos}             # graphlib wants {node: predecessors}
    for a, outs in succ.items():
        for b in outs:
            preds[b].add(a)

    ts = TopologicalSorter(preds)
    ts.prepare()                                # raises CycleError on a true conflict
    out = []
    ready = list(ts.get_ready())
    while ready:
        # Emit ONE node at a time (lowest tiebreak), then re-fetch ready so a
        # node unblocked by this emission can be picked before the rest of the
        # current batch. This is what produces run-grouping: after DK, DV becomes
        # ready and wins over DQ (same sub-kernel, earlier position) -> DK,DV,DQ,DB.
        ready.sort(key=lambda n: pos[n])
        n = ready.pop(0)
        out.append(n)
        ts.done(n)
        ready += list(ts.get_ready())
    return out
