# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit test for the specs -> Axis/Override builder (executive plan Step 2.4).

Acceptance: building the real attn_fwd signature reproduces the §7 axis table
(canonical order + radices) from agent-plans/ati+newbinds_rev1.md.

Uses _validate=False because this step exercises the builder's axis/anchor/radix
logic over the §7 subset; the full 74-argument completeness-passing description is
Step 4.2."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tritonsrc'))

import aotriton.template_instantiation as ati
from aotriton.template_instantiation.describe import describe, get_kernel_spec
from aotriton.template_instantiation.builder import build_kernel, BuiltKernel
from aotriton.template_instantiation.ir import Interface


class _IRStub(Interface):
    FAMILY = 'test'
    NAME = 'stub'
    def __init__(self, axes, overrides):
        self._axes = axes
        self._overrides = overrides
    def _axes_overrides(self):
        return self._axes, self._overrides
    # Interface abstract contract (no functional struct in this bare IR stub).
    @property
    def func_cfields(self):
        return []
    def list_functional_params(self):
        return []


def enumerate_functionals(axes, overrides, target_arch):
    return _IRStub(axes, overrides).gen_functionals(target_arch)

# Real kernel (requires triton; it lives in a real file so JITFunction loads).
from fwd_kernel import attn_fwd

MAIN_DTYPES = ['*fp16:16', '*bf16:16', '*fp32:16']
BLOCK_DMODEL_VALUES = [16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 512]  # 12


def _describe_attn_fwd_subset():
    T_io = ati.type_var('T_io', dtype=MAIN_DTYPES, signature_name='Q')
    specs = [
        ati.tensor('Q', T_io, strides='stride_q?', contiguous=-1),
        ati.tensor('K', T_io, strides='stride_k?', contiguous=-1),
        ati.tensor('V', T_io, strides='stride_v?', contiguous=-1),
        ati.tensor('Out', T_io, strides='stride_o?', contiguous=-1),
        ati.tensor('B', T_io, strides='stride_b?'),
        ati.scalar('BLOCK_DMODEL', options=BLOCK_DMODEL_VALUES),
        ati.scalar('PADDED_HEAD', options=[False, True]),
        ati.scalar('ENABLE_DROPOUT', options=[False, True]),
        ati.scalar('CAUSAL_TYPE', options=[0, 3]),
        ati.scalar('BIAS_TYPE', options=[0, 1]),
        # a shared scalar choice variable
        ati.scalar('Max_seqlen_q', ati.scalar_var('Seqlen', options=['i32'])),
        ati.overrides('B', to=0, when=ati.eq('BIAS_TYPE', 0)),
    ]
    describe(attn_fwd, *specs, _validate=False)
    return get_kernel_spec(attn_fwd)


def test_build_returns_built_kernel():
    spec = _describe_attn_fwd_subset()
    bk = build_kernel(spec)
    assert isinstance(bk, BuiltKernel)
    assert bk.name == 'attn_fwd'
    assert bk.arguments[0] == 'Q'              # full signature order preserved


def test_section7_axis_table():
    spec = _describe_attn_fwd_subset()
    bk = build_kernel(spec)
    multi = [a for a in bk.axes if not a.is_trivial]
    # §7: six multi-choice axes, in this canonical (anchor) order, these radices.
    assert [a.var_name for a in multi] == [
        'T_io', 'BLOCK_DMODEL', 'PADDED_HEAD', 'ENABLE_DROPOUT',
        'CAUSAL_TYPE', 'BIAS_TYPE']
    assert [a.radix for a in multi] == [3, 12, 2, 2, 2, 2]


def test_section7_godel_via_enumeration():
    spec = _describe_attn_fwd_subset()
    bk = build_kernel(spec)
    fs = list(enumerate_functionals(bk.axes, bk.overrides, {'gfx942': ['g0']}))
    assert len(fs) == 576                       # 3*12*2*2*2*2
    godels = {f.godel_number for f in fs}
    assert godels == set(range(576))            # dense bijection


def test_tio_groups_five_tensors_one_axis():
    spec = _describe_attn_fwd_subset()
    bk = build_kernel(spec)
    tio = next(a for a in bk.axes if a.var_name == 'T_io')
    assert set(tio.arg_names) == {'Q', 'K', 'V', 'Out', 'B'}
    assert all(tio.ranks[a] == 4 for a in tio.arg_names)   # rank 4 from 4 strides


def test_contiguous_stride_is_constexpr_one():
    spec = _describe_attn_fwd_subset()
    bk = build_kernel(spec)
    # stride_qk is the contiguous (unit) stride -> constexpr 1, hidden trivial axis
    qk = next(a for a in bk.axes if a.var_name == 'stride_qk')
    assert qk.is_trivial
    assert qk.choices[0].is_constexpr and qk.choices[0].triton_compile_signature == 1
    # a non-unit stride is u64:8
    qz = next(a for a in bk.axes if a.var_name == 'stride_qz')
    assert qz.choices[0].triton_compile_signature == 'u64:8'


def test_override_applies_in_enumeration():
    spec = _describe_attn_fwd_subset()
    bk = build_kernel(spec)
    fs = list(enumerate_functionals(bk.axes, bk.overrides, {'gfx942': ['g0']}))
    for f in fs:
        if f.choice['BIAS_TYPE'].triton_compile_signature == 0:
            assert f.resolved['B'].triton_compile_signature == 0
        else:
            assert f.resolved['B'].is_tensor


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} builder tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
