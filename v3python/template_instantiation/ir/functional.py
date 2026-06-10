# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Functional + enumerate_functionals (ATI executive plan Step 1.4; see
agent-plans/ati+newbinds_rev1.md §6).

A Functional is one fully-pinned instantiation of a kernel, excluding perf:
every choice axis selected and all overrides applied. It is frozen — it stores
the resolved {arg_name -> Choice} table rather than anything that still needs
settling. `(arch_number, godel_number)` is its global identity; arches share the
godel space.

enumerate_functionals is the outer loop of the build: product over the
multi-choice axes (godel), fan out each variable's choice onto its arguments,
then apply overrides in declared order. No fixpoint — predicates read only the
free-axis selection, known in full before overrides run.
"""

import itertools

from .axis import assign_godel, godel_of


class ChoiceView:
    """Ergonomic accessor over a Functional's pinned choices (executive plan
    Step 1.5; agent-plans/ati_rev1.md §6, ati+newbinds_rev1.md §5).

    Attribute access is keyed by *choice-variable name*: `f.choices.T_io` returns
    the variable's triton signature (replacing the old check_value(f, ['Q'])
    argument-name lookup). `.tc(var)` hands back the raw Choice; `.arg(aname)`
    reads the resolved per-argument value (post-override)."""

    __slots__ = ('_choice', '_resolved')

    def __init__(self, functional):
        self._choice = functional.choice       # var_name -> Choice
        self._resolved = functional.resolved   # arg_name -> Choice

    def __getattr__(self, var):
        choice = self._choice.get(var)
        if choice is None:
            raise AttributeError(
                f'{var!r} is not a choice variable of this functional; '
                f'valid: {sorted(self._choice)}')
        return choice.triton_compile_signature

    def tc(self, var):
        """The raw Choice for a variable (for .itype / .type_enum / rank)."""
        if var not in self._choice:
            raise KeyError(
                f'{var!r} is not a choice variable; valid: {sorted(self._choice)}')
        return self._choice[var]

    def arg(self, aname):
        """Resolved (post-override) triton signature for an argument."""
        if aname not in self._resolved:
            raise KeyError(
                f'{aname!r} is not a resolved argument; '
                f'valid: {sorted(self._resolved)}')
        return self._resolved[aname].triton_compile_signature

    def arg_tc(self, aname):
        """Resolved (post-override) raw Choice for an argument."""
        if aname not in self._resolved:
            raise KeyError(
                f'{aname!r} is not a resolved argument; '
                f'valid: {sorted(self._resolved)}')
        return self._resolved[aname]


class Functional:
    __slots__ = ('arch', 'arch_number', 'godel_number', 'choice', 'resolved',
                 '_choices_view')

    def __init__(self, arch, arch_number, godel_number, choice, resolved):
        self.arch = arch
        self.arch_number = arch_number
        self.godel_number = godel_number
        self.choice = dict(choice)        # var_name -> Choice (free + trivial)
        self.resolved = dict(resolved)    # arg_name -> Choice (post-override)
        self._choices_view = None

    @property
    def choices(self) -> ChoiceView:
        """Cached ergonomic accessor; see ChoiceView."""
        if self._choices_view is None:
            self._choices_view = ChoiceView(self)
        return self._choices_view

    @property
    def identity(self):
        """Global identity: (arch_number, godel_number)."""
        return (self.arch_number, self.godel_number)

    def __repr__(self):
        return (f'Functional(arch={self.arch!r}, arch_number={self.arch_number}, '
                f'godel={self.godel_number})')


def _resolve(axes_all, overrides, picked):
    """Step 3+4: fan out var->args (tensor ranks specialized per arg), then push
    overrides in declared order. Pure function of `picked`."""
    resolved = {}
    for axis in axes_all:
        nth = axis.choices.index(picked[axis.var_name])
        for arg in axis.arg_names:
            resolved[arg] = axis.choice_for_arg(nth, arg)
    for ov in overrides:
        if ov.predicate.holds(picked):
            c = ov.materialize(picked)
            for t in ov.targets:
                resolved[t] = c
    return resolved


def enumerate_functionals(axes, overrides, target_arch):
    """Yield every Functional of a kernel.

    axes:        all Axis objects (ordering need not be canonical; sorted here).
    overrides:   Override list, in declared order.
    target_arch: ordered {arch -> gpus}; arch_number is its enumeration index.
    """
    axes_all = sorted(axes, key=lambda a: a.anchor)
    axes_multi = [a for a in axes_all if not a.is_trivial]
    assign_godel(axes_multi)

    # Trivial axes pin a single choice; precompute it.
    trivial_pick = {a.var_name: a.choices[0] for a in axes_all if a.is_trivial}

    for arch_number, (arch, _gpus) in enumerate(target_arch.items()):
        for sel in itertools.product(*[range(a.radix) for a in axes_multi]):
            godel = godel_of(axes_multi, sel)
            picked = dict(trivial_pick)
            picked.update({a.var_name: a.choices[i]
                           for i, a in zip(sel, axes_multi)})
            resolved = _resolve(axes_all, overrides, picked)
            yield Functional(arch=arch, arch_number=arch_number,
                             godel_number=godel,
                             choice=picked, resolved=resolved)
