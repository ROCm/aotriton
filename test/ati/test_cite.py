# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Aux-kernel xref Step 4: @ati.cite kernel-level resolution + flat kernel registry
(agent-plans/ati_aux-kernel-xref_rev0.md §4.4).

A citing kernel declares only what is unique to it and cites another kernel for the
rest: gap arguments (matched by apparel name) and string dtype-variable references
are inherited from the cited kernel. Unresolved gaps / unknown cite targets raise
DescriptionError. We use the real attn_fwd (cited) and debug (citing) kernels."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'modules' / 'flash'))
sys.path.insert(0, str(REPO / 'modules' / 'flash' / 'kernel'))

import aotriton.template_instantiation as ati
from aotriton.template_instantiation import registry
from aotriton.template_instantiation.describe import describe
from aotriton.template_instantiation.ir.kdesc import build_kernel_description
from aotriton.template_instantiation.builder import DescriptionError

# The cited kernel: the real ATI attn_fwd description (finalized on import).
from aot.attn_fwd import attn_fwd
# The citing kernel: the debug kernel (re-described per-test below).
from dropout_rng import debug_simulate_encoded_softmax

MAIN_DTYPES = ['*fp16:16', '*bf16:16', '*fp32:16']
BLOCK_DMODEL_VALUES = [16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 512]


def _describe_full_attn_fwd():
    # The real ATI attn_fwd (already described via the stacked-@ form on import);
    # just build + register it as the cite target.
    return build_kernel_description(attn_fwd, family='flash',
                                    source_path='tritonsrc/flash.py',
                                    triton_kernel_name='attn_fwd')


def _describe_citing_debug():
    # debug cites attn_fwd; declares only R (wired, strided) + perf-ish constexprs.
    # dropout_p / Num_head_q / Max_seqlen_q/k / philox_* are GAPS filled from the
    # cite by apparel name. 'T_io' is a string dtype-var resolved through the cite.
    specs = [
        ati.cite('op_attn_fwd.triton.attn_fwd'),
        ati.tensor('R', 'T_io', strides='stride_r?', contiguous=-1,
                   wires_to='encoded_softmax'),
        ati.scalar(['BLOCK_M', 'BLOCK_N'], options=[64]),
    ]
    describe(debug_simulate_encoded_softmax, *specs, _validate=False)
    return build_kernel_description(debug_simulate_encoded_softmax, family='flash',
                                    source_path='tritonsrc/flash.py',
                                    register=False)


def setup():
    registry.clear('flash')
    _describe_full_attn_fwd()        # registers attn_fwd


def test_string_dtype_resolved_through_cite():
    setup()
    kdesc = _describe_citing_debug()
    # R's dtype T_io came from the cited attn_fwd: a 3-choice tensor axis.
    R = next(a for a in kdesc._built.axes if 'R' in a.arg_names)
    assert R.radix == 3
    assert [c.tc.triton_compile_signature for c in R.choices] == MAIN_DTYPES


def test_gap_scalars_inherited():
    setup()
    kdesc = _describe_citing_debug()
    args = set(kdesc.ARGUMENTS)
    # gaps that debug never declared locally, now present as inherited axes
    for gap in ('dropout_p', 'Num_head_q', 'Max_seqlen_q', 'philox_offset2'):
        ax = kdesc.axis_of_arg(gap)
        assert ax is not None, f'{gap} not inherited'
    # dropout_p inherited as fp32 scalar
    dp = kdesc.axis_of_arg('dropout_p')
    assert dp.choices[0].tc.triton_compile_signature == 'fp32'


def test_local_scalar_wins_over_cite():
    """A2: a locally-declared argument is NEVER overwritten by a cite that also
    defines it. attn_fwd declares dropout_p as a plain fp32 scalar; debug declaring
    it locally as an enumerated (options) scalar must keep the LOCAL definition."""
    setup()
    specs = [
        ati.cite('op_attn_fwd.triton.attn_fwd'),
        ati.tensor('R', 'T_io', strides='stride_r?', contiguous=-1,
                   wires_to='encoded_softmax'),
        # dropout_p is defined by the cite (fp32) AND locally (as an enumerated
        # constexpr) — the local definition must win.
        ati.scalar('dropout_p', options=[0]),
        ati.scalar(['BLOCK_M', 'BLOCK_N'], options=[64]),
    ]
    describe(debug_simulate_encoded_softmax, *specs, _validate=False)
    kdesc = build_kernel_description(debug_simulate_encoded_softmax, family='flash',
                                     source_path='tritonsrc/flash.py', register=False)
    dp = kdesc.axis_of_arg('dropout_p')
    # Local enumerated [0] -> a single constexpr choice 0, NOT the cited fp32 scalar.
    assert dp is not None
    sigs = [c.tc.triton_compile_signature for c in dp.choices]
    assert sigs == [0], f'local dropout_p overwritten by cite: {sigs!r}'


def test_local_derive_wins_over_cited_derive():
    """A2: a cited @ati.derives is inherited only when the citing kernel does NOT
    override the same target. attn_fwd derives dropout_p->0 when ENABLE_DROPOUT is
    False; debug declaring dropout_p locally (no derive) must NOT inherit that
    derive (the operand is locally claimed, so the cited override is skipped)."""
    setup()
    specs = [
        ati.cite('op_attn_fwd.triton.attn_fwd'),
        ati.tensor('R', 'T_io', strides='stride_r?', contiguous=-1,
                   wires_to='encoded_softmax'),
        ati.scalar('dropout_p', options=[0]),     # local claim, no derive
        ati.scalar(['BLOCK_M', 'BLOCK_N'], options=[64]),
    ]
    describe(debug_simulate_encoded_softmax, *specs, _validate=False)
    kdesc = build_kernel_description(debug_simulate_encoded_softmax, family='flash',
                                     source_path='tritonsrc/flash.py', register=False)
    targets = {t for ov in kdesc._built.overrides for t in ov.targets}
    assert 'dropout_p' not in targets, \
        'cited dropout_p derive leaked onto a locally-claimed operand'


def test_unresolved_gap_raises():
    setup()
    # Cite a kernel that does NOT define some of debug's args -> DescriptionError.
    # Use a tiny cited kernel missing philox.
    from fwd_kernel import attn_fwd as _af  # noqa
    specs = [
        ati.cite('op_attn_fwd.triton.attn_fwd'),
        # Deliberately omit R so encoded_softmax is unrelated; but R has no apparel
        # match in the cite either -> unresolved.
        ati.scalar(['BLOCK_M', 'BLOCK_N'], options=[64]),
    ]
    describe(debug_simulate_encoded_softmax, *specs, _validate=False)
    try:
        build_kernel_description(debug_simulate_encoded_softmax, family='flash',
                                 source_path='tritonsrc/flash.py', register=False)
    except DescriptionError as e:
        assert 'R' in str(e) or 'stride_rz' in str(e)
        return
    raise AssertionError('expected DescriptionError for unresolved gap')


def test_unknown_cite_target_raises():
    setup()
    specs = [
        ati.cite('op_attn_fwd.triton.no_such_kernel'),
        ati.tensor('R', '*fp16:16', strides='stride_r?', contiguous=-1),
        ati.scalar(['BLOCK_M', 'BLOCK_N'], options=[64]),
    ]
    describe(debug_simulate_encoded_softmax, *specs, _validate=False)
    try:
        build_kernel_description(debug_simulate_encoded_softmax, family='flash',
                                 source_path='tritonsrc/flash.py', register=False)
    except DescriptionError as e:
        assert 'no_such_kernel' in str(e)
        return
    raise AssertionError('expected DescriptionError for unknown cite target')


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} cite tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
