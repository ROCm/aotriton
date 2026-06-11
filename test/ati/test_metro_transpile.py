# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Step 5.5: the @ati.metro_kernel if/else transpiler. Parses the body's AST
(never executes it) into a MetroPlan, and lowers it to MetroKernel /
ConditionalKernel IR equal to the hand-written legacy metro structure."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import v3python.template_instantiation as ati
from v3python.template_instantiation.metro import (
    transpile, lower_plan, MetroError, Call, Cond,
)


# --- the §5 metro descriptions (defined in this file so getsource works) ---

@ati.metro_kernel
def metro_fwd(params):
    attn_fwd(params)
    if params.encoded_softmax.data_ptr() != 0:
        debug_simulate_encoded_softmax(params, R=params.encoded_softmax)


@ati.metro_kernel
def metro_bwd(params):
    if params.num_seqlens > 0:
        bwd_preprocess_varlen(params)
    else:
        bwd_preprocess(params)
    bwd_kernel_dk_dv(params)
    bwd_kernel_dq(params)


def test_fwd_plan_structure():
    p = metro_fwd.__ati_metro__
    assert [type(s).__name__ for s in p.steps] == ['Call', 'Cond']
    assert p.steps[0].kernel == 'attn_fwd'
    cond = p.steps[1]
    assert cond.if_parameter == 'encoded_softmax'
    assert cond.if_expr == '->data_ptr() != nullptr'   # matches legacy string
    assert cond.then[0].kernel == 'debug_simulate_encoded_softmax'
    assert cond.then[0].renames == {'R': 'encoded_softmax'}   # kwarg rename
    assert cond.orelse == []


def test_bwd_plan_structure():
    p = metro_bwd.__ati_metro__
    assert [type(s).__name__ for s in p.steps] == ['Cond', 'Call', 'Call']
    cond = p.steps[0]
    assert (cond.if_parameter, cond.if_expr) == ('num_seqlens', '> 0')
    assert cond.then[0].kernel == 'bwd_preprocess_varlen'
    assert cond.orelse[0].kernel == 'bwd_preprocess'
    assert [s.kernel for s in p.steps[1:]] == ['bwd_kernel_dk_dv', 'bwd_kernel_dq']


def test_lower_to_ir_matches_legacy_strings():
    # Lower with stand-in kernels + the real ConditionalKernel; the lowered metro
    # must carry the same kernel order and conditional (if_parameter, if_expr) the
    # hand-written MetroFwdKernel uses.
    from v3python.op import ConditionalKernel

    class _K:
        def __init__(self, name):
            self.NAME = name
            self.SHARED_IFACE = None
        def list_non_functional_params(self):
            return []

    kmap = {n: _K(n) for n in
            ('attn_fwd', 'debug_simulate_encoded_softmax')}
    captured = {}

    def metro_factory(steps, renames):
        captured['steps'] = steps
        captured['renames'] = renames
        return steps

    steps = lower_plan(metro_fwd.__ati_metro__, kmap,
                       metro_factory, ConditionalKernel)
    assert steps[0] is kmap['attn_fwd']
    ck = steps[1]
    assert isinstance(ck, ConditionalKernel)
    assert ck.if_parameter == 'encoded_softmax'
    assert ck.if_expr == '->data_ptr() != nullptr'
    assert ck.if_kernel is kmap['debug_simulate_encoded_softmax']
    assert ck.else_kernel is None
    # The debug sub-kernel's R is wired to the operand encoded_softmax; the rename
    # is captured keyed by the concrete sub-kernel object (metro-layer property).
    debug = kmap['debug_simulate_encoded_softmax']
    assert captured['renames'] == {debug: {'R': 'encoded_softmax'}}


def test_metro_kernel_merges_renamed_arg():
    # The renamed arg must collapse into its operand node, not appear twice in the
    # merged operand order. (This is what keeps `R` out of the params struct.) The
    # MetroKernel constructor runs the heavy Interface machinery that needs an
    # operator-bound SHARED_IFACE; here we exercise only merged_operand_order, so
    # build the object directly and let lower_plan populate it.
    from v3python.op import MetroKernel, ConditionalKernel

    class _K:
        def __init__(self, name, arguments):
            self.NAME = name
            self.SHARED_IFACE = None
            self.ARGUMENTS = arguments
        def list_non_functional_params(self):
            return []

    attn = _K('attn_fwd', ['Q', 'K', 'V', 'Out', 'encoded_softmax'])
    debug = _K('debug_simulate_encoded_softmax', ['R', 'dropout_p'])
    kmap = {'attn_fwd': attn, 'debug_simulate_encoded_softmax': debug}

    def metro_factory(steps, renames):
        m = object.__new__(MetroKernel)   # skip Interface init (orthogonal here)
        m.NAME = 'metro_fwd'
        m._kernels = steps
        m._renames = renames
        return m

    metro = lower_plan(metro_fwd.__ati_metro__, kmap,
                       metro_factory, ConditionalKernel)
    assert metro.rename_for(debug) == {'R': 'encoded_softmax'}
    order = metro.merged_operand_order()
    # R folded into encoded_softmax: present once, no bare 'R'.
    assert 'R' not in order
    assert order.count('encoded_softmax') == 1
    assert order == ['Q', 'K', 'V', 'Out', 'encoded_softmax', 'dropout_p']


def test_out_of_grammar_rejected():
    # An assignment in the body is outside the grammar -> MetroError.
    def bad(params):
        x = 1                       # noqa
        attn_fwd(params)
    try:
        transpile(bad)
    except MetroError:
        return
    raise AssertionError('expected MetroError on out-of-grammar statement')


def test_unsupported_condition_rejected():
    def bad(params):
        if params.x + 1:            # not a comparison against a literal
            attn_fwd(params)
    try:
        transpile(bad)
    except MetroError:
        return
    raise AssertionError('expected MetroError on unsupported condition')


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} metro-transpile tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
