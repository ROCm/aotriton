# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ATI linker acceptance demo (ati_linker_exec0 Step 7): a sub-kernel @ati.cites the
WHOLE metro that CONTAINS it (a true cycle), and the two-pass linker resolves it via
the header/extern model — the citer inherits the OTHER sub-kernels' argument surface,
never its own. @ati.hints.union_precedence steers the colliding-operand donor to the
KEY kernel (dk_dv) instead of whichever sub-kernel comes first in call order.

bwd_kernel_dq declares only its dQ/dB outputs + perf and cites op_attn_bwd.triton_split
for the shared inputs (Q/K/V/B/DO/L/D + the seqlen/head/dropout/window scalars + the
BLOCK_DMODEL/CAUSAL_TYPE/... constexprs)."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from aotriton.codegen.linker import Linker


def _dq():
    kernels, _ops, _aff = Linker(REPO / 'modules').link_all_families()
    return next(k for k in kernels if k.NAME == 'bwd_kernel_dq')


def test_inherits_shared_inputs_as_gaps():
    dq = _dq()
    fields = {cf.aname for cf in dq.func_cfields}
    # local outputs
    assert 'DQ' in fields and 'DB' in fields
    # inherited shared inputs (gaps from the metro's other sub-kernels)
    for g in ('Q', 'K', 'V', 'B', 'DO', 'L', 'D', 'sm_scale', 'BLOCK_DMODEL',
              'BIAS_TYPE', 'CAUSAL_TYPE'):
        assert g in fields, f'{g} not inherited from the whole-metro cite'


def test_union_precedence_picks_key_kernel_stride_binding():
    dq = _dq()
    args = set(dq.ARGUMENTS)
    # dO's 4th stride: the KEY kernel dk_dv calls it stride_dok; the preprocess kernels
    # call it stride_don. union_precedence([dk_dv, dq, ...]) makes the key kernel the
    # donor, so dq binds stride_dok (its own signature name), not stride_don.
    assert 'stride_dok' in args
    assert 'stride_don' not in args


def test_functional_space_matches_key_kernel():
    dq = _dq()
    # same 576 godel space as the standalone key kernels (T_io*3 * BLOCK_DMODEL*12 * 4
    # binary features) — inheriting inputs did not perturb the functional axes.
    assert dq.godel_number == 3 * 12 * 2 * 2 * 2 * 2
    assert dq.is_tunable is True


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} whole-metro-cite tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
