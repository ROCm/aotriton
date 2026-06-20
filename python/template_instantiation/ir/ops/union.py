# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
union_params / build_merged_struct_cfields — order-preserving merge of a METRO
KERNEL's sub-kernel argument lists, and the operator params-struct cfield union.

A metro kernel implements one operator functional with a *set of collaborating
kernels* (fwd: attn_fwd + debug; bwd: preprocess + dk_dv + dq). Each sub-kernel's
argument list is a totally-ordered chain; their union is a DAG and the merged
order is a stable topological sort (graphlib.TopologicalSorter). The tiebreak is
the **sub-kernel's index in the metro call sequence** (collaboration order) then
position — so e.g. dk_dv (called first) contributes DK,DV before dq's DQ,DB.

Shared arguments stitch the chains together. Conflicts are resolved by PRIORITY,
not by error: edges are added order_hint-first then sub-kernel by sub-kernel, and
an edge that would close a cycle is dropped. So an adjacent reversal (a shared
block ordered x,y,z in one kernel and z,y,x in another) auto-resolves to the
earlier-listed kernel's order — the result is always a DAG. `order_hint` is simply
the highest-priority chain when an author wants to force an order.

Rename-aware: the metro DSL wires sub-kernel arguments to operator operands (the
transpiler's by-name + kwarg renames). Pass per-sub-kernel rename maps
{kernel_arg: operand} so wired-together arguments merge into a single node — the
merge runs over the RENAMED (operand) names.
"""

from graphlib import TopologicalSorter


def build_merged_struct_cfields(subkernels):
    """The operator params-struct field list, merged across all backends' sub-kernels.

    For an operator whose params struct is NOT a single kernel's superset (the bwd
    operator: the struct is the union of dk_dv/dq + the preprocess kernels' `Out` +
    the affine kernel's `DQ_ACC`), merge the sub-kernels' `func_cfields` into one
    order-preserving list via union_params over their (apparel) field names. The
    FIRST sub-kernel to define a name owns its cfield (so each operand's ctype/nbits
    come from its defining kernel). `subkernels` must be given in the desired priority
    order (key kernels first), since union_params resolves order conflicts by
    first-listed-wins.

    A backend may provide a `union_order` (e.g. the affine kernel: an anchored chain
    like [DB, DQ_ACC, L]) used purely for ORDERING — so an operand only that backend
    supplies lands between its declared neighbors — while its cfields still come from
    `func_cfields`.
    """
    from ..cfield import cfield
    cfield_by_name = {}
    name_lists = []
    for s in subkernels:
        for cf in s.func_cfields:
            cfield_by_name.setdefault(cf.aname, cf)   # first definer owns the cfield
        order_hint = getattr(s, 'union_order', None)
        if order_hint:
            name_lists.append(list(order_hint))       # anchored ordering chain
        else:
            name_lists.append([cf.aname for cf in s.func_cfields])
    order = union_params(name_lists)
    merged = []
    for i, name in enumerate(order):
        if name not in cfield_by_name:
            from ...builder.errors import DescriptionError
            sources = [type(s).__name__ for s in subkernels]
            raise DescriptionError(
                f'build_merged_struct_cfields: operand {name!r} appears in '
                f'a union_order chain but is not defined in any sub-kernel\'s '
                f'func_cfields. Merged order: {order}. '
                f'Defined operands: {sorted(cfield_by_name)}. '
                f'Sub-kernels (in priority order): {sources}. '
                f'Check the union_order declaration for a misspelled or '
                f'missing operand name.')
        cf = cfield_by_name[name]
        merged.append(cfield(ctype=cf.ctype, aname=cf.aname, ctext=cf.ctext,
                             index=i, nbits=cf.nbits))
    return merged


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
    order_hint: optional sequence given the HIGHEST priority — its consecutive
                edges are added before any sub-kernel's, so it wins all conflicts.

    Conflicts are resolved by PRIORITY, not by error: edges are added
    order_hint-first, then sub-kernel by sub-kernel, and an edge is skipped if it
    would close a cycle (its target already reaches its source). So when a shared
    block appears in different orders across sub-kernels (e.g. x,y,z vs z,y,x), the
    earlier-listed kernel's order wins and the later contradicting edges are
    dropped — no CycleError. The result is always a DAG; the toposort never fails.

    Returns the merged list of (operand) names.
    """
    renames = renames or [None] * len(arg_lists)
    assert len(renames) == len(arg_lists), 'renames must align with arg_lists'

    pos = {}        # node -> (subkernel_index, position) tiebreak key
    # Candidate edges in PRIORITY order, each tagged authoritative?: order_hint
    # first (authoritative — may break any cycle), then each sub-kernel's
    # consecutive pairs in metro call order.
    # When an order_hint is given the author takes responsibility for ordering, so
    # ALL chains become authoritative: any edge that would close a cycle is dropped
    # (the hint, added first, wins). Without a hint, sub-kernel chains are
    # non-authoritative: only adjacent reversals auto-resolve, longer cycles raise.
    hint_mode = bool(order_hint)
    edge_sources = []       # list of (chain, authoritative)
    if order_hint:
        edge_sources.append((list(order_hint), True))
    for ki, (args, rename) in enumerate(zip(arg_lists, renames)):
        chain = _renamed(args, rename)
        for i, a in enumerate(chain):
            pos.setdefault(a, (ki, i))          # first mention owns the tiebreak
        edge_sources.append((chain, hint_mode))

    succ = {}       # node -> set(successors)
    nodes = set(pos)

    def _reaches(src, dst):
        """True if dst is reachable from src in the edges added so far."""
        if src == dst:
            return True
        stack = [src]
        seen = set()
        while stack:
            n = stack.pop()
            if n == dst:
                return True
            if n in seen:
                continue
            seen.add(n)
            stack.extend(succ.get(n, ()))
        return False

    # `authoritative` chains (order_hint) may break ANY cycle by dropping the
    # contradicting lower-priority edge. Sub-kernel chains only auto-resolve an
    # ADJACENT reversal (a direct b->a transposition); a cycle that closes through
    # a longer path (shared params interleaved with private ones, e.g.
    # k1=[x,p,z] k2=[z,q,x]) is a genuine conflict left to surface as CycleError,
    # for the author to resolve with order_hint.
    for chain, authoritative in edge_sources:
        for a, b in zip(chain, chain[1:]):
            if a == b:
                continue
            if b in succ.get(a, ()):            # already have a->b
                continue
            if _reaches(b, a):
                if authoritative or (b in succ and a in succ[b]):
                    continue                    # drop a->b; higher priority wins
                # else: genuine non-adjacent conflict; let toposort raise.
            succ.setdefault(a, set()).add(b)

    preds = {n: set() for n in nodes}           # graphlib wants {node: predecessors}
    for a, outs in succ.items():
        for b in outs:
            preds[b].add(a)

    ts = TopologicalSorter(preds)
    ts.prepare()                                # raises CycleError on a genuine
                                                # (non-adjacent) conflict -> order_hint
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
