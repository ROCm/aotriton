# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Step 5.2b: AtiOperator reproduces the legacy OpAttnFwd operator surface — the
identity, functional space, param struct, and backend/optune metadata the
operator codegen reads. Requires the ATI kernel+operator path."""

import os
import sys
from pathlib import Path

os.environ['AOTRITON_ATI_KERNELS'] = 'attn_fwd op_attn_fwd'

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'tritonsrc'))

import v3python.rules.flash as F
from v3python.template_instantiation.compat.operator_adapter import AtiOperator
from v3python.gpu_targets import cluster_gpus


def _ati_op():
    op = next(o for o in F.operators if o.NAME == 'op_attn_fwd')
    assert isinstance(op, AtiOperator), 'op_attn_fwd should be ATI-backed here'
    return op


def test_operator_is_ati_backed():
    assert isinstance(_ati_op(), AtiOperator)


def test_identity_and_param_struct():
    op = _ati_op()
    assert op.NAME == 'op_attn_fwd'
    assert op.enum_name == 'kOp_OpAttnFwd'
    assert op.param_class_name == 'OpAttnFwdParams'
    assert op.context_class_name == 'OpAttnFwdContext'
    assert op.CALL_OPTIONS_NAME == 'attn_options'


def test_functional_space_matches_default_backend():
    op = _ati_op()
    assert op.godel_number == 576
    assert len(op.func_cfields) == 46
    ta = cluster_gpus(['gfx942_mod0'])
    fs = list(op.gen_functionals(ta))
    assert len(fs) == 576
    assert all(f.meta_object is op for f in fs)     # functionals belong to the op


def test_backends_and_optune():
    op = _ati_op()
    assert op.nbackends == 2                         # triton metro + aiter
    assert op.fallback_backend is op.list_backends()[0]
    assert set(op.OPTUNE_KEYS) == {'Max_seqlen_q', 'Max_seqlen_k'}


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} operator-adapter tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
