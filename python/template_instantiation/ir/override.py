# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Predicate / VarRef / Override — the conditional primitive of the new IR

An Override pushes a value onto target arguments when its Predicate holds for a
functional's free-axis selection. This replaces the old pull-style deferred
conditional choices (ConditionalConstexpr / CDC / CDETensor): target, value, and
condition are explicit and separate, so no fixpoint settling is needed.

Resolution runs in declared order over a `picked` map {var_name -> TypedChoice}:
  - Predicate.holds(picked) reads only free axes (asserted at build time), which
    is the invariant that keeps resolution a single pure pass (§8-1).
  - Override.value is a literal (-> TypedChoice.parse) or a VarRef (-> copy the
    TypedChoice currently assigned to that variable, e.g. Hdim_qk <- BLOCK_DMODEL).
"""

import operator as _op

from .typed_choice import TypedChoice
from ..specs.base import StackedSpec

# Closed comparison grammar. Same operator set as the metro `if` conditions.
_OPS = {
    '==': _op.eq,
    '!=': _op.ne,
    '<':  _op.lt,
    '>':  _op.gt,
    '<=': _op.le,
    '>=': _op.ge,
}


class VarRef:
    """Reference to another choice variable's value (`to='BLOCK_DMODEL'`)."""
    __slots__ = ('var_name',)

    def __init__(self, var_name: str):
        self.var_name = var_name

    def __repr__(self):
        return f'VarRef({self.var_name!r})'


class ValueFn:
    """A derive value computed from the functional: `to(functional)` -> value.
    For values that are a function of functional state (e.g. NUM_XCDS from arch,
    possibly several values 1/3/6/8)."""
    __slots__ = ('fn',)

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, functional):
        return self.fn(functional)

    def __repr__(self):
        return f'ValueFn({getattr(self.fn, "__name__", self.fn)!r})'


class Predicate:
    """`(var_name <op> operand)` over a free-axis choice value."""
    __slots__ = ('var_name', 'op', 'operand')

    def __init__(self, var_name: str, op: str, operand):
        assert op in _OPS, f'unsupported predicate op {op!r}; allowed: {sorted(_OPS)}'
        self.var_name = var_name
        self.op = op
        self.operand = operand

    def holds(self, picked: dict) -> bool:
        """True if this functional's selection satisfies the predicate. `picked`
        maps var_name -> TypedChoice; constexpr choices expose their python value via
        triton_compile_signature (0 / False / 3 ...)."""
        assert self.var_name in picked, \
            f'predicate variable {self.var_name!r} not in selection {sorted(picked)}'
        value = picked[self.var_name].triton_compile_signature
        return _OPS[self.op](value, self.operand)

    def validate(self, free_var_names) -> None:
        """A predicate may read free axes only (§8-1)."""
        assert self.var_name in free_var_names, (
            f'predicate references {self.var_name!r}, which is not a free choice '
            f'axis; predicates may read free axes only')

    def __repr__(self):
        return f'Predicate({self.var_name!r} {self.op} {self.operand!r})'


# Convenience builders (ati.eq/ne/lt/gt lower to these in Step 2.2).
def eq(var_name, operand): return Predicate(var_name, '==', operand)
def ne(var_name, operand): return Predicate(var_name, '!=', operand)
def lt(var_name, operand): return Predicate(var_name, '<',  operand)
def gt(var_name, operand): return Predicate(var_name, '>',  operand)
def le(var_name, operand): return Predicate(var_name, '<=', operand)
def ge(var_name, operand): return Predicate(var_name, '>=', operand)


class Override(StackedSpec):
    """Rewrite `targets` to `value` for functionals where `predicate` fires.

    `predicate` is either a structured `Predicate` (var op operand, over a free
    choice axis) or a plain CALLABLE taking the functional and returning bool —
    e.g. `when=lambda f: f.arch in ('gfx942', 'gfx950')`. The callable form can
    read any functional state (arch, several choices) without growing a predicate
    operator/var vocabulary."""
    __slots__ = ('targets', 'predicate', 'value')

    def __init__(self, targets, predicate, value):
        if isinstance(targets, str):
            targets = (targets,)
        self.targets = tuple(targets)
        assert isinstance(predicate, Predicate) or callable(predicate), \
            f'Override needs a Predicate or a callable, got {predicate!r}'
        self.predicate = predicate
        self.value = value          # literal | str type | VarRef

    def fires(self, functional) -> bool:
        """Whether this override applies to `functional`. Dispatches on the
        predicate kind: a structured Predicate reads the choice dict; a callable
        receives the functional itself."""
        if isinstance(self.predicate, Predicate):
            return self.predicate.holds(functional.choice)
        return bool(self.predicate(functional))

    def materialize(self, ctx) -> 'TypedChoice':
        """The TypedChoice to write into the targets for a firing functional. `ctx` is
        a functional-like object exposing `.choice` (var->TypedChoice) and `.arch`, or
        a plain {var->TypedChoice} dict (no ValueFn support in that form)."""
        picked = ctx.choice if hasattr(ctx, 'choice') else ctx
        if isinstance(self.value, VarRef):
            assert self.value.var_name in picked, (
                f'override value {self.value!r} references a variable not in the '
                f'selection {sorted(picked)}')
            return picked[self.value.var_name]
        if isinstance(self.value, ValueFn):
            return TypedChoice.parse(self.value(ctx))
        return TypedChoice.parse(self.value)

    def validate(self, free_var_names) -> None:
        # A callable predicate is opaque to static validation; only structured
        # Predicates are checked against the free axes.
        if isinstance(self.predicate, Predicate):
            self.predicate.validate(free_var_names)
        if isinstance(self.value, VarRef):
            assert self.value.var_name in free_var_names, (
                f'override value {self.value!r} references {self.value.var_name!r}, '
                f'which is not a free choice axis; VarRef may read free axes only')

    def __repr__(self):
        return f'Override(targets={self.targets}, {self.predicate!r}, value={self.value!r})'
