# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Step 5.4: --preview renders the implicit structures (params struct, merged
argument order, axis manifest) of an ATI description."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from functools import lru_cache
from types import SimpleNamespace
from aotriton.codegen.linker import Linker
from aotriton.template_instantiation.tools import preview, preview_kdesc


@lru_cache(maxsize=1)
def _families():
    _k, _o, _a = Linker(REPO / 'modules').link_all_families()
    return SimpleNamespace(kernels=_k, operators=_o, affine_kernels=_a)


def _op():
    return next(o for o in _families().operators if o.NAME == 'op_attn_fwd')


def test_params_struct_rendered():
    text = preview_kdesc(_op())
    assert 'struct OpAttnFwdParams {' in text
    assert 'const TensorView<4>* Q;' in text
    assert 'const TensorView<2>* A;' in text         # alibi is rank 2
    assert 'float                Sm_scale;' in text
    assert 'int8_t               Q_descale;' in text  # single-choice constexpr field


def test_axis_manifest_section7():
    text = preview_kdesc(_op())
    # §7 multi-choice axes with their radices
    assert 'Q                radix=3' in text         # T_io, signature_name=Q
    assert 'BLOCK_DMODEL     radix=12' in text
    assert 'BIAS_TYPE        radix=2' in text
    assert 'total functionals per arch = 576' in text


def test_merged_argument_order():
    text = preview_kdesc(_op())
    assert 'merged argument order' in text
    # the operator's merged order starts with the main tensors
    assert 'Q, K, V, B, A, Sm_scale, L, Out' in text


def test_selective_filter():
    # selective matches the operator only
    F = _families()
    text = preview(selective='flash/op/op_attn_fwd',
                   kernels=F.kernels, operators=F.operators)
    assert 'op_attn_fwd' in text
    assert 'flash/triton/attn_fwd' not in text


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} preview tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
