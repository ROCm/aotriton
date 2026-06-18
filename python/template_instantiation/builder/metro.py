# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Metro lowering (pipeline Stage 4 — LOWER).

Lowers a transpiled MetroPlan (specs/metro.py) to the MetroKernel/ConditionalKernel
IR (ir/metro.py). The linker calls build_metro once each sub-kernel is built.
"""

from ..specs.metro import Call, MetroError


def lower_plan(plan, kernel_map, metro_factory, conditional_factory):
    """Lower a MetroPlan to the existing MetroKernel/ConditionalKernel IR.

    kernel_map:          {sub-kernel name -> KernelDescription object}.
    metro_factory:       callable(steps:list) -> MetroKernel (the lowered backend
                         list). Argument wiring is NOT threaded here — it lives on
                         each sub-kernel's kdesc (wires_to=, rev0 §4.3).
    conditional_factory: the ConditionalKernel class/callable
                         (if_parameter, if_expr, if_kernel, else_kernel).

    Each Cond branch must be a single sub-kernel call (the C++ if/else launcher
    template supports one kernel per branch); a multi-step branch is an error.
    """
    def resolve(call):
        if call.kernel not in kernel_map:
            raise MetroError(
                f'metro {plan.name!r}: unknown sub-kernel {call.kernel!r}; '
                f'known: {sorted(kernel_map)}')
        return kernel_map[call.kernel]

    def one_call(name, branch, which):
        if len(branch) != 1 or not isinstance(branch[0], Call):
            raise MetroError(
                f'metro {plan.name!r}: the {which} branch of a condition must be a '
                f'single sub-kernel call')
        return branch[0]

    steps = []
    for step in plan.steps:
        if isinstance(step, Call):
            steps.append(resolve(step))
        else:  # Cond
            if_call = one_call(plan.name, step.then, 'if')
            else_kernel = None
            if step.orelse:
                else_kernel = resolve(one_call(plan.name, step.orelse, 'else'))
            steps.append(conditional_factory(step.if_parameter, step.if_expr,
                                             resolve(if_call), else_kernel))
    return metro_factory(steps)


def build_metro(plan, kernel_map, name, *, family):
    """Lower a transpiled MetroPlan to a MetroKernel (a metro launcher).

    plan:        the @ati.metro_kernel transpiler output (fn.__ati_node__).
    kernel_map:  {sub-kernel name -> built kdesc}.
    name:        the metro/backend NAME (-> kMetro_<Name> enum).
    """
    from ..ir.metro import MetroKernel, ConditionalKernel
    return lower_plan(plan, kernel_map,
                      lambda steps: MetroKernel(name, steps, family=family),
                      ConditionalKernel)
