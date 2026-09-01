# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Aux-kernel xref Step 12 (part 3): the backward metro built from the
@ati.metro_kernel transpiler (lower_plan) is structurally equal to the
hand-written MetroBwdKernel (agent-plans/ati_aux-kernel-xref_exec0.md Step 12).

The bwd metro is three unconditional Calls: preprocess + dk_dv + dq. It used to
open with Cond('num_seqlens', '> 0') selecting bwd_preprocess_varlen; under
varlen_bits the two preprocess kernels are one kernel with varlen_bits == 0, so
the conditional and the kernel it selected are both gone. The ConditionalKernel
path is still covered -- metro_fwd carries one (test_metro_fwd_build.py), and
test_metro_transpile.py keeps a synthetic conditional metro of its own."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'modules' / 'flash'))

from aotriton.template_instantiation.ir.metro import (
    MetroKernel, ConditionalKernel)
from aotriton.template_instantiation.builder import lower_plan


class _K:
    def __init__(self, name):
        self.NAME = name
        self.SHARED_IFACE = None
    def list_non_functional_params(self):
        return []


def _load_real_flash_aot():
    """Import modules/flash/aot BY PATH, under a name that cannot be shadowed.

    A plain `import aot` is NOT safe here. test_port_bwd_dkdv.py puts
    python/test/fakefamily/flash at the FRONT of sys.path at collection time and
    imports `aot` from there, so by the time this test runs sys.modules['aot'] is
    the FAKE family -- and this test then asserts against the fake while claiming
    to check the real one. That went unnoticed for as long as the two metros were
    structurally identical, which is exactly until the real one changed.
    """
    import importlib.util
    path = REPO / 'modules' / 'flash' / 'aot' / '__init__.py'
    spec = importlib.util.spec_from_file_location(
            'flash_aot_real', path, submodule_search_locations=[str(path.parent)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules['flash_aot_real'] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_metro_bwd_plan():
    return _load_real_flash_aot().metro_bwd.__ati_node__


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
    subs = {n: _K(n) for n in ('bwd_preprocess',
                               'bwd_kernel_dk_dv', 'bwd_kernel_dq')}

    def factory(steps):
        return MetroKernel('triton_split', steps, family='flash')

    transpiled = lower_plan(_load_metro_bwd_plan(), subs, factory, ConditionalKernel)

    handwritten = factory([
        subs['bwd_preprocess'],
        subs['bwd_kernel_dk_dv'],
        subs['bwd_kernel_dq'],
    ])

    assert _structure(transpiled) == _structure(handwritten)
    assert _structure(transpiled) == [
        ('kernel', 'bwd_preprocess'),
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
