# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Predicate / VarRef / Override — the conditional primitive of the new IR
(ATI executive plan Step 1.3; see agent-plans/ati+newbinds_rev1.md §2.2, §6).

An Override pushes a value onto target arguments when its Predicate holds for a
functional's free-axis selection. This replaces the old pull-style deferred
conditional choices (ConditionalConstexpr / CDC / CDETensor): target, value, and
condition are explicit and separate, so no fixpoint settling is needed.

Resolution runs in declared order over a `picked` map {var_name -> Choice}:
  - Predicate.holds(picked) reads only free axes (asserted at build time), which
    is the invariant that keeps resolution a single pure pass (§8-1).
  - Override.value is a literal (-> Choice.parse) or a VarRef (-> copy the Choice
    currently assigned to that variable, e.g. Hdim_qk <- BLOCK_DMODEL).
"""

import operator as _op

from .choice import Choice

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
        maps var_name -> Choice; constexpr choices expose their python value via
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


class Override:
    """Rewrite `targets` to `value` for functionals where `predicate` holds."""
    __slots__ = ('targets', 'predicate', 'value')

    def __init__(self, targets, predicate: Predicate, value):
        if isinstance(targets, str):
            targets = (targets,)
        self.targets = tuple(targets)
        assert isinstance(predicate, Predicate), \
            f'Override needs a Predicate, got {predicate!r}'
        self.predicate = predicate
        self.value = value          # literal | str type | VarRef

    def materialize(self, picked: dict) -> Choice:
        """The Choice to write into the targets for a firing functional."""
        if isinstance(self.value, VarRef):
            assert self.value.var_name in picked, (
                f'override value {self.value!r} references a variable not in the '
                f'selection {sorted(picked)}')
            return picked[self.value.var_name]
        return Choice.parse(self.value)

    def validate(self, free_var_names) -> None:
        self.predicate.validate(free_var_names)
        if isinstance(self.value, VarRef):
            assert self.value.var_name in free_var_names, (
                f'override value {self.value!r} references {self.value.var_name!r}, '
                f'which is not a free choice axis; VarRef may read free axes only')

    def __call__(self, kernel):
        """Stacked-@ form: accumulate this override onto the kernel below it."""
        from ..describe import accumulate_spec
        return accumulate_spec(self, kernel)

    def __repr__(self):
        return f'Override(targets={self.targets}, {self.predicate!r}, value={self.value!r})'
