# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Aux-kernel xref Step 9: the forward metro built from the @ati.metro_kernel
transpiler (lower_plan) is structurally equal to the hand-written MetroFwdKernel
(agent-plans/ati_aux-kernel-xref_exec0.md Step 9).

We build the fwd metro both ways via lower_plan against the real metro_fwd plan and
the hand-written list, and compare: same backend NAME, same step kinds, same
sub-kernel order, same ConditionalKernel (if_parameter, if_expr, if/else kernels).
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'modules' / 'flash'))

from aotriton.op import MetroKernel, ConditionalKernel
from aotriton.template_instantiation.metro import lower_plan


class _K:
    def __init__(self, name):
        self.NAME = name
        self.SHARED_IFACE = None
    def list_non_functional_params(self):
        return []


def _load_metro_fwd_plan():
    import aot.metro_fwd as mod
    return mod.metro_fwd.__ati_metro__


def _structure(metro):
    """A comparable fingerprint of a metro's lowered steps."""
    out = []
    for step in metro.list_kernels():
        if isinstance(step, ConditionalKernel):
            out.append(('cond', step.if_parameter, step.if_expr,
                        step.if_kernel.NAME,
                        step.else_kernel.NAME if step.else_kernel else None))
        else:
            out.append(('kernel', step.NAME))
    return out


def test_transpiled_fwd_metro_matches_handwritten():
    attn = _K('attn_fwd')
    debug = _K('debug_simulate_encoded_softmax')
    kmap = {'attn_fwd': attn, 'debug_simulate_encoded_softmax': debug}

    def factory(steps):
        m = object.__new__(MetroKernel)
        m.NAME = 'triton'
        m._kernels = steps
        return m

    transpiled = lower_plan(_load_metro_fwd_plan(), kmap, factory, ConditionalKernel)

    # Hand-written equivalent.
    handwritten = factory([
        attn,
        ConditionalKernel('encoded_softmax', '->data_ptr() != nullptr', debug),
    ])

    assert _structure(transpiled) == _structure(handwritten)
    assert _structure(transpiled) == [
        ('kernel', 'attn_fwd'),
        ('cond', 'encoded_softmax', '->data_ptr() != nullptr',
         'debug_simulate_encoded_softmax', None),
    ]


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} metro-fwd-build tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
