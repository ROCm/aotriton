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


class Functional:
    __slots__ = ('arch', 'arch_number', 'godel_number', 'choice', 'resolved')

    def __init__(self, arch, arch_number, godel_number, choice, resolved):
        self.arch = arch
        self.arch_number = arch_number
        self.godel_number = godel_number
        self.choice = dict(choice)        # var_name -> Choice (free + trivial)
        self.resolved = dict(resolved)    # arg_name -> Choice (post-override)

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
