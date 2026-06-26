# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Step 4.1: KernelDescription enumeration/godel parity with the legacy
attn_fwd KernelDescription."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tritonsrc'))

import pytest
import aotriton.template_instantiation as ati
from aotriton.template_instantiation.describe import describe
sys.path.insert(0, str(Path(__file__).resolve().parent))
from registry import InterfaceRegistry, _testonly_build_kernel_description
from aotriton.gpu_targets import cluster_gpus

from fwd_kernel import attn_fwd

MAIN_DTYPES = ['*fp16:16', '*bf16:16', '*fp32:16']
BLOCK_DMODEL_VALUES = [16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 512]


def _adapter():
    reg = InterfaceRegistry()
    T_io = ati.type_var('T_io', dtype=MAIN_DTYPES, signature_name='Q')
    describe(attn_fwd,
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
             ati.overrides('B', to=0, when=ati.eq('BIAS_TYPE', 0)),
             _validate=False)
    return _testonly_build_kernel_description(attn_fwd, family='flash',
                                    source_path='tritonsrc/flash.py',
                                    registry=reg)


try:
    import v3python.rules.flash as _F   # legacy reference for parity comparison
except ModuleNotFoundError:
    _F = None   # unavailable (v3python removed) -> legacy-parity tests skip


def _legacy():
    return next(k for k in _F.kernels if k.NAME == 'attn_fwd')


def test_total_godel_matches_legacy():
    if _F is None:
        pytest.skip('v3python legacy reference unavailable')
    assert _adapter().godel_number == _legacy().godel_number == 576


def test_gen_functionals_count_and_dense_godel():
    ta = cluster_gpus(['gfx942_mod0'])
    fs = list(_adapter().gen_functionals(ta))
    assert len(fs) == 576
    godels = sorted(f.godel_number for f in fs)
    assert godels == list(range(576))               # dense bijection
    assert all(f.arch == 'gfx942' for f in fs)


def test_godel_set_equals_legacy():
    if _F is None:
        pytest.skip('v3python legacy reference unavailable')
    ta = cluster_gpus(['gfx942_mod0'])
    mine = sorted(f.godel_number for f in _adapter().gen_functionals(ta))
    legacy = sorted(f.godel_number for f in _legacy().gen_functionals(ta))
    assert mine == legacy


def test_arguments_match_real_signature():
    a = _adapter()
    names = [p.name for p in attn_fwd.params]
    assert a.ARGUMENTS == names
    assert a.NAME == 'attn_fwd' and a.FAMILY == 'flash'


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} kdesc-adapter tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
