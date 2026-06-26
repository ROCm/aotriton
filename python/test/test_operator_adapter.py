# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Step 5.2b: Operator reproduces the legacy OpAttnFwd operator surface — the
identity, functional space, param struct, and backend/optune metadata the
operator codegen reads. Requires the ATI kernel+operator path."""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from types import SimpleNamespace
from aotriton.codegen.linker import Linker
from aotriton.template_instantiation.ir.operator import Operator

_k, _o, _a = Linker(REPO / 'modules').link_all_families()
F = SimpleNamespace(kernels=_k, operators=_o, affine_kernels=_a)
from aotriton.gpu_targets import cluster_gpus


def _ati_op():
    op = next(o for o in F.operators if o.NAME == 'op_attn_fwd')
    assert isinstance(op, Operator), 'op_attn_fwd should be ATI-backed here'
    return op


def _ati_bwd_op():
    op = next(o for o in F.operators if o.NAME == 'op_attn_bwd')
    assert isinstance(op, Operator), 'op_attn_bwd should be ATI-backed here'
    return op


def test_operator_is_ati_backed():
    assert isinstance(_ati_op(), Operator)


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


def test_fwd_fallback_is_explicit():
    # fwd declares @ati.tune.fallback(PADDED_HEAD=False) at the OPERATOR level
    # (matching legacy OpAttnFwd.PARTIALLY_TUNED_FUNCTIONALS).
    assert _ati_op().partially_tuned_functionals == {'PADDED_HEAD': False}


# --- op_attn_bwd: the union-struct operator (Step 10) ---

_BWD_FIELDS = [
    'Q', 'K', 'V', 'B', 'sm_scale', 'Out', 'DO', 'DK', 'DV', 'DQ', 'DB', 'DQ_ACC',
    'L', 'D', 'num_head_q', 'num_head_k', 'cu_seqlens_q', 'cu_seqlens_k',
    'num_seqlens', 'max_seqlen_q', 'max_seqlen_k', 'seq_strides_q', 'seq_strides_k',
    'hdim_qk', 'hdim_vo', 'dropout_p', 'philox_seed_ptr', 'philox_offset1',
    'philox_offset2', 'Window_left', 'Window_right', 'BLOCK_DMODEL', 'CAUSAL_TYPE',
    'ENABLE_DROPOUT', 'PADDED_HEAD', 'BIAS_TYPE',
]


def test_bwd_identity_and_backends():
    op = _ati_bwd_op()
    assert op.enum_name == 'kOp_OpAttnBwd'
    assert op.param_class_name == 'OpAttnBwdParams'
    assert op.godel_number == 576
    assert op.nbackends == 3                          # metro_split + fuse + aiter
    assert set(op.OPTUNE_KEYS) == {'max_seqlen_q', 'max_seqlen_k'}


def test_bwd_union_struct_with_supplied_dq_acc():
    op = _ati_bwd_op()
    names = [cf.aname for cf in op.func_cfields]
    assert names == _BWD_FIELDS                       # union order == legacy order
    assert len(names) == 36
    dq_acc = op.func_cfields[11]
    assert dq_acc.aname == 'DQ_ACC'                   # between DB and L (anchored)
    assert dq_acc.ctype == 'LazyTensorInternal<4>*'


def test_dq_acc_supplied_by_affine_backend():
    # DQ_ACC is NOT a triton sub-kernel operand — it is SUPPLIED by the affine
    # backend (@ati.affine.supplies), so the operator struct is a pure union over
    # all backends (no hand-injection).
    op = _ati_bwd_op()
    aiter = next(b for b in op.list_backends()
                 if getattr(b, 'NAME', None) == 'aiter_fmha_v3_bwd')
    supplied = [s.arg_names[0] for s in aiter.supplied_operands]
    assert supplied == ['DQ_ACC']
    # the triton metro sub-kernels do NOT carry DQ_ACC
    metro = op.list_backends()[0]
    for sub in metro.iter_subkernels():
        assert 'DQ_ACC' not in [cf.aname for cf in sub.func_cfields]


def test_bwd_fallback_not_inherited_from_kernel():
    # The representative kernel (bwd_kernel_dk_dv) declares
    # @ati.tune.fallback(PADDED_HEAD=False); the OPERATOR must NOT inherit it
    # (legacy OpAttnBwd has an empty PARTIALLY_TUNED_FUNCTIONALS).
    assert _ati_bwd_op().partially_tuned_functionals == {}


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} operator-adapter tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
