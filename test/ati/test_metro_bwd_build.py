# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Aux-kernel xref Step 12 (part 3): the backward metro built from the
@ati.metro_kernel transpiler (lower_plan) is structurally equal to the
hand-written MetroBwdKernel (agent-plans/ati_aux-kernel-xref_exec0.md Step 12).

The bwd metro is Cond(preprocess) + dk_dv + dq; the condition
`params.num_seqlens > 0` lowers to ('num_seqlens', '> 0')."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tritonsrc'))

import importlib.util
from v3python.op import MetroKernel, ConditionalKernel
from v3python.template_instantiation.metro import lower_plan


class _K:
    def __init__(self, name):
        self.NAME = name
        self.SHARED_IFACE = None
    def list_non_functional_params(self):
        return []


def _load_metro_bwd_plan():
    p = REPO / 'modules' / 'flash' / 'metro_bwd_ati.py'
    spec = importlib.util.spec_from_file_location('_t_metro_bwd_ati', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.metro_bwd.__ati_metro__


def _structure(metro):
    out = []
    for step in metro.list_kernels():
        if isinstance(step, ConditionalKernel):
            out.append(('cond', step.if_parameter, step.if_expr,
                        step.if_kernel.NAME,
                        step.else_kernel.NAME if step.else_kernel else None))
        else:
            out.append(('kernel', step.NAME))
    return out


def test_transpiled_bwd_metro_matches_handwritten():
    subs = {n: _K(n) for n in ('bwd_preprocess', 'bwd_preprocess_varlen',
                               'bwd_kernel_dk_dv', 'bwd_kernel_dq')}

    def factory(steps):
        m = object.__new__(MetroKernel)
        m.NAME = 'triton_split'
        m._kernels = steps
        return m

    transpiled = lower_plan(_load_metro_bwd_plan(), subs, factory, ConditionalKernel)

    handwritten = factory([
        ConditionalKernel('num_seqlens', '> 0',
                          subs['bwd_preprocess_varlen'], subs['bwd_preprocess']),
        subs['bwd_kernel_dk_dv'],
        subs['bwd_kernel_dq'],
    ])

    assert _structure(transpiled) == _structure(handwritten)
    assert _structure(transpiled) == [
        ('cond', 'num_seqlens', '> 0', 'bwd_preprocess_varlen', 'bwd_preprocess'),
        ('kernel', 'bwd_kernel_dk_dv'),
        ('kernel', 'bwd_kernel_dq'),
    ]


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} metro-bwd-build tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
