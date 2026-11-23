# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Per-call rules: `ati.derives` overrides a tuning module yields from `variants(meta)`.

The same `Override`/`Predicate` objects as the compile-time `@ati.derives`, read
per call instead of per functional: each predicate compares one runtime value
(declared by the family, e.g. `seqlen_k`) with an int, and the targets are
knobs. A rule's `when` is therefore a box -- one inclusive interval per runtime
value it names -- and:

- rules on the same target must be disjoint (`check`);
- where none of a target's rules holds, the target keeps its default (the
  tuning module's per-functional resolution), so every call has a build;
- JIT applies the rules holding for the call (`overrides_at`);
- AOT compiles one image per distinct override set of the common refinement
  (`builds`) and selects among them per call by box.

`domains` maps each runtime value a family declares to its inclusive integer
range, the range of the C++ value it is wired to; "every call" is every value
in it.
"""

import itertools

from .override import Override, Predicate, ValueFn, VarRef


def predicates(rule):
    """`rule`'s predicates as a tuple: its `when`, unpacked."""
    if isinstance(rule.predicate, Predicate):
        return (rule.predicate,)
    if isinstance(rule.predicate, tuple):
        return rule.predicate
    raise ValueError(f'{rule!r}: a per-call rule takes ati.eq/lt/le/gt/ge predicates, not a callable')


def box(rule, domains):
    """`{runtime value: (lo, hi)}`: the calls `rule` holds for. Validates the rule."""
    if not isinstance(rule, Override):
        raise TypeError(f'a per-call rule is an ati.derives(...), got {rule!r}')
    if isinstance(rule.value, (VarRef, ValueFn)):
        raise ValueError(f'{rule!r}: a per-call rule pins a literal (a string `to=` reads as a '
                         f'choice variable, which a call does not have)')
    found = {}
    for p in predicates(rule):
        if p.var_name not in domains:
            raise ValueError(f'{p!r}: {p.var_name!r} is not a runtime value; declared: {sorted(domains)}')
        lo, hi = p.interval(domains[p.var_name])
        if p.var_name in found:
            lo, hi = max(lo, found[p.var_name][0]), min(hi, found[p.var_name][1])
        if lo > hi:
            raise ValueError(f'{rule!r} holds for no call')
        found[p.var_name] = (lo, hi)
    return found


def overrides(rule):
    """`{knob: value}` that `rule` pins."""
    return {t: rule.value for t in rule.targets}


def reads(rules, domains):
    """The runtime values `rules` read."""
    return sorted({name for r in rules for name in box(r, domains)})


def contains(found_box, values):
    return all(lo <= values[name] <= hi for name, (lo, hi) in found_box.items())


def _intersect(a, b):
    return all(max(a[n][0], b[n][0]) <= min(a[n][1], b[n][1]) for n in a.keys() & b.keys())


def check(rules, domains):
    """Rules on the same target must not overlap."""
    boxes = [box(r, domains) for r in rules]
    for (r, rb), (s, sb) in itertools.combinations(zip(rules, boxes), 2):
        shared = set(r.targets) & set(s.targets)
        if shared and _intersect(rb, sb):
            raise ValueError(f'{r!r} and {s!r} both pin {sorted(shared)} on overlapping calls')


def overrides_at(rules, values, domains):
    """JIT: the knob overrides for a call with these runtime values.

    Every value a rule reads must be given: guessing one would pick a build AOT
    would not, for some call.
    """
    check(rules, domains)
    missing = [name for name in reads(rules, domains) if values.get(name) is None]
    if missing:
        raise ValueError(f'this functional has several builds: pass the runtime value(s) {missing}')
    found = {}
    for r in rules:
        if contains(box(r, domains), values):
            found.update(overrides(r))
    return found


def builds(rules, domains):
    """`[(boxes, overrides)]`: every distinct override set and the calls it serves.

    The common refinement: each runtime value is cut at every rule bound, so
    every cell of the product is all-in or all-out of each rule and one point
    per cell decides it. Cells with equal overrides are merged back into boxes.
    Ordered by first cell, so the first is the build for the smallest values.
    """
    check(rules, domains)
    boxes = [box(r, domains) for r in rules]
    names = reads(rules, domains)
    segments = []
    for n in names:
        lo, hi = domains[n]
        cuts = {lo, hi + 1} | {b for rb in boxes if n in rb for b in (rb[n][0], rb[n][1] + 1)}
        cuts = sorted(cuts)
        segments.append([(a, b - 1) for a, b in zip(cuts, cuts[1:])])
    groups = {}
    for cell in itertools.product(*segments):
        values = {n: seg[0] for n, seg in zip(names, cell)}
        found = {}
        for r, rb in zip(rules, boxes):
            if contains(rb, values):
                found.update(overrides(r))
        groups.setdefault(tuple(sorted(found.items())), []).append(dict(zip(names, cell)))
    return [(_merge(cells, domains), dict(key)) for key, cells in groups.items()]


def verify(rules, domains):
    """What AOT selects agrees with what JIT applies, for every call.

    At one point per cell of the common refinement -- exhaustive, since every
    rule bound is a cut -- exactly one build's boxes hold the call, and its
    overrides are `overrides_at`'s. Raises on the first disagreement.
    """
    found = builds(rules, domains)
    boxes = [box(r, domains) for r in rules]
    names = reads(rules, domains)
    axes = []
    for n in names:
        lo, hi = domains[n]
        cuts = {lo, hi + 1} | {e for rb in boxes if n in rb for e in (rb[n][0], rb[n][1] + 1)}
        axes.append(sorted(cuts)[:-1])
    for point in itertools.product(*axes):
        values = dict(zip(names, point))
        serving = [over for bs, over in found if any(contains(b, values) for b in bs)]
        if len(serving) != 1:
            raise AssertionError(f'{values} is served by {len(serving)} builds')
        if serving[0] != overrides_at(rules, values, domains):
            raise AssertionError(f'{values}: AOT selects {serving[0]}, JIT applies '
                                 f'{overrides_at(rules, values, domains)}')
    return found


def _merge(cells, domains):
    """Greedily join boxes that differ in one runtime value and touch there."""
    found = [dict(c) for c in cells]
    merged = True
    while merged:
        merged = False
        for a, b in itertools.combinations(range(len(found)), 2):
            x, y = found[a], found[b]
            diff = [n for n in x if x[n] != y[n]]
            if len(diff) == 1:
                n = diff[0]
                (l1, h1), (l2, h2) = sorted((x[n], y[n]))
                if h1 + 1 == l2:
                    x[n] = (l1, h2)
                    del found[b]
                    merged = True
                    break
    # A runtime value spanning its whole domain is no condition at all.
    return [{n: iv for n, iv in f.items() if iv != domains[n]} for f in found]
