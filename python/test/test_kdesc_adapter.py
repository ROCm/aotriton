# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Step 4.1: KernelDescription enumeration/godel behavior.

(Legacy parity tests against v3python.rules.flash's attn_fwd
KernelDescription were removed when v3python/ was deleted; see
agent-plans/modular-tune.md Phase 1 step 5 / F-list §2g.)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import aotriton.template_instantiation as ati
from aotriton.template_instantiation.describe import describe
from registry import InterfaceRegistry, _testonly_build_kernel_description
from aotriton.gpu_targets import cluster_gpus

from fakekernels import fwd_kernel_stub
attn_fwd = fwd_kernel_stub()

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
                                    registry=reg)


def test_gen_functionals_count_and_dense_godel():
    ta = cluster_gpus(['gfx942_mod0'])
    fs = list(_adapter().gen_functionals(ta))
    assert len(fs) == 576
    godels = sorted(f.godel_number for f in fs)
    assert godels == list(range(576))               # dense bijection
    assert all(f.arch == 'gfx942' for f in fs)


def test_arguments_match_real_signature():
    a = _adapter()
    names = list(attn_fwd.params)
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
