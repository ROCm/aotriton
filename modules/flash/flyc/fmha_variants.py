# SPDX-License-Identifier: Apache-2.0

"""Per-call knob rules for flyc kernels, shared by JIT and AOT.

A tuning module's `variants(meta)` yields `derive(targets, values, when=...)`
rules: "for calls whose runtime values satisfy `when`, pin `targets` to
`values`". They are the per-call counterpart of the `if cond: override` steps
in `resolve()`, which decide per functional. Both paths read the same rules:
JIT applies the ones matching the call (`overrides_at`), AOT compiles one image
per distinct override set and selects among them by box (`builds`).

Spelled like `ati.derives` / `ati.eq` but deliberately not the same objects:
those are compile-time (their predicates read functional axes), these read
runtime values only, and this module imports neither aotriton nor flydsl, so
the JIT path can use it.

Semantics:
- `when` is one predicate, a tuple of them (AND), or `()` (every call). Each
  predicate compares one runtime value with a constant, so a `when` is a box:
  one inclusive interval per runtime value it names.
- Rules on the same target must be disjoint (`check`). Where none of a
  target's rules matches, the target keeps its default (`resolve` decides),
  so rules need not cover anything and every call still gets a build.
- The builds of a functional are the common refinement of all its rules;
  rules on different targets may cut the same runtime value.
"""

import itertools

# Runtime values a rule may read, with their domains: the range of the C++
# value each is wired to, so "every call" means every value C++ can hold.
_INT32 = (-2**31, 2**31 - 1)
RUNTIME_VALUES = {
    'hdim_qk': _INT32,
    'seqlen_q': _INT32,
    'seqlen_k': _INT32,
}


class Pred:
    """`name <op> constant` over a runtime value: one interval of it."""
    __slots__ = ('name', 'op', 'operand')

    def __init__(self, name, op, operand):
        if name not in RUNTIME_VALUES:
            raise ValueError(f'{name!r} is not a runtime value; known: {sorted(RUNTIME_VALUES)}')
        self.name, self.op, self.operand = name, op, int(operand)

    def interval(self):
        lo, hi = RUNTIME_VALUES[self.name]
        c = self.operand
        return {'==': (c, c), '<': (lo, c - 1), '<=': (lo, c), '>': (c + 1, hi), '>=': (c, hi)}[self.op]

    def __repr__(self):
        return f'{self.name} {self.op} {self.operand}'


def eq(name, c): return Pred(name, '==', c)
def lt(name, c): return Pred(name, '<', c)
def le(name, c): return Pred(name, '<=', c)
def gt(name, c): return Pred(name, '>', c)
def ge(name, c): return Pred(name, '>=', c)


class Rule:
    """Pin `overrides` on the calls in `box`."""
    __slots__ = ('overrides', 'box', 'when')

    def __init__(self, overrides, box, when):
        self.overrides, self.box, self.when = overrides, box, when

    def __repr__(self):
        return f'derive({self.overrides}, when={self.when})'


def derive(targets, values, *, when):
    """`targets` (a knob name, or a tuple of coupled ones) <- `values` where `when` holds."""
    if isinstance(targets, str):
        targets, values = (targets,), (values,)
    if len(targets) != len(values):
        raise ValueError(f'{len(targets)} targets but {len(values)} values')
    preds = when if isinstance(when, tuple) else (when,)
    box = {}
    for p in preds:
        lo, hi = p.interval()
        if p.name in box:
            lo, hi = max(lo, box[p.name][0]), min(hi, box[p.name][1])
        if lo > hi:
            raise ValueError(f'`when` {preds} holds for no call')
        box[p.name] = (lo, hi)
    return Rule(dict(zip(targets, values)), box, preds)


def reads(rules):
    """The runtime values `rules` read."""
    return sorted({name for r in rules for name in r.box})


def contains(box, values):
    return all(lo <= values[name] <= hi for name, (lo, hi) in box.items())


def _intersect(a, b):
    return all(max(a[n][0], b[n][0]) <= min(a[n][1], b[n][1]) for n in a.keys() & b.keys())


def check(rules):
    """Rules on the same target must not overlap."""
    for r, s in itertools.combinations(rules, 2):
        shared = r.overrides.keys() & s.overrides.keys()
        if shared and _intersect(r.box, s.box):
            raise ValueError(f'{r} and {s} both pin {sorted(shared)} on overlapping calls')


def overrides_at(rules, values):
    """JIT: the knob overrides for a call with these runtime values.

    Every value a rule reads must be given: guessing one would pick a build AOT
    would not, for some call.
    """
    check(rules)
    missing = [name for name in reads(rules) if values.get(name) is None]
    if missing:
        raise ValueError(f'this functional has several builds: pass the runtime value(s) {missing}')
    found = {}
    for r in rules:
        if contains(r.box, values):
            found.update(r.overrides)
    return found


def builds(rules):
    """[(boxes, overrides)]: every distinct override set and where it applies.

    The common refinement: each runtime value is cut at every rule bound, so
    every cell of the product is all-in or all-out of each rule and one point
    per cell decides it. Cells with equal overrides are merged back into boxes.
    Ordered by first cell; the first is the build for the smallest values.
    """
    check(rules)
    names = reads(rules)
    segments = []
    for n in names:
        lo, hi = RUNTIME_VALUES[n]
        cuts = {lo, hi + 1} | {b for r in rules if n in r.box for b in (r.box[n][0], r.box[n][1] + 1)}
        cuts = sorted(cuts)
        segments.append([(a, b - 1) for a, b in zip(cuts, cuts[1:])])
    groups = {}
    for cell in itertools.product(*segments):
        values = {n: seg[0] for n, seg in zip(names, cell)}
        found = {}
        for r in rules:
            if contains(r.box, values):
                found.update(r.overrides)
        groups.setdefault(tuple(sorted(found.items())), []).append(dict(zip(names, cell)))
    return [(_merge(boxes), dict(key)) for key, boxes in groups.items()]


def _merge(boxes):
    """Greedily join boxes that differ in one runtime value and touch there."""
    boxes = [dict(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        for a, b in itertools.combinations(range(len(boxes)), 2):
            x, y = boxes[a], boxes[b]
            diff = [n for n in x if x[n] != y[n]]
            if len(diff) == 1:
                n = diff[0]
                (l1, h1), (l2, h2) = sorted((x[n], y[n]))
                if h1 + 1 == l2:
                    x[n] = (l1, h2)
                    del boxes[b]
                    merged = True
                    break
    # A runtime value spanning its whole domain is no condition at all.
    return [{n: iv for n, iv in b.items() if iv != RUNTIME_VALUES[n]} for b in boxes]
