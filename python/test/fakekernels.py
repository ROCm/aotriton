# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fake, minimal kernels for ATI unit tests.

The ATI unit suite tests the *generator machinery* (describe / build / cite /
wiring / apparel / metro), not the flash family. To keep the tests independent of
the real flash Triton sources (modules/flash/kernel + modules/flash/aot), they use
tiny empty-body kernel stubs defined here instead of importing the production
descriptions.

A KernelStub is exactly what @ati.source produces without importing Triton: a name
plus the parameter-name list (and optional string annotations). Tests attach their
own @ati.* specs via describe(); nothing is read from disk.
"""

from aotriton.template_instantiation.decorators import KernelStub


def stub(name, params, annotations=None):
    """A KernelStub like @ati.source produces (no triton import)."""
    return KernelStub(name, list(params), source_path=None, annotations=annotations)


# The citing "debug" kernel used by the cite/wiring/apparel tests. Logical
# argument names only — strides are synthesized by ati.tensor(strides=...), and the
# tests supply the full description via describe(). Superset of what any single
# test references so one stub serves them all.
DEBUG_PARAMS = [
    'R',
    'stride_rz', 'stride_rh', 'stride_rm', 'stride_rn',
    'dropout_p',
    'Num_head_q', 'Max_seqlen_q', 'Max_seqlen_k',
    'philox_seed_ptr', 'philox_offset1', 'philox_offset2',
    'BLOCK_M', 'BLOCK_N',
]


def debug_stub():
    """Fresh stub for the citing debug kernel (describe() mutates it, so each test
    should build its own)."""
    return stub('debug_simulate_encoded_softmax', DEBUG_PARAMS)


# Raw-kernel signatures (param names only), mirroring the real flash Triton kernels
# so tests that describe() a raw kernel with their own specs behave identically
# (match_strides finds the same stride names). No triton, no file read.
FWD_PARAMS = [
    'Q', 'K', 'V', 'B', 'A', 'Sm_scale', 'L', 'Out',
    'Q_descale', 'K_descale', 'P_scale', 'P_descale', 'V_descale',
    'stride_qz', 'stride_qh', 'stride_qm', 'stride_qk',
    'stride_kz', 'stride_kh', 'stride_kn', 'stride_kk',
    'stride_vz', 'stride_vh', 'stride_vk', 'stride_vn',
    'stride_oz', 'stride_oh', 'stride_om', 'stride_on',
    'stride_bz', 'stride_bh', 'stride_bm', 'stride_bn',
    'stride_az', 'stride_ah',
    'Num_head_q', 'Num_head_k', 'Num_seqlens', 'cu_seqlens_q', 'cu_seqlens_k',
    'Max_seqlen_q', 'Max_seqlen_k', 'seq_strides_q', 'seq_strides_k',
    'BLOCK_DMODEL', 'Hdim_qk', 'Hdim_vo', 'PADDED_HEAD',
    'ENABLE_DROPOUT', 'dropout_p', 'philox_seed_ptr', 'philox_offset1',
    'philox_offset2', 'philox_seed_output', 'philox_offset_output',
    'RETURN_ENCODED_SOFTMAX', 'encoded_softmax', 'CAUSAL_TYPE',
    'Window_left', 'Window_right', 'BIAS_TYPE', 'USE_ALIBI',
    'INT8', 'INT8_KV', 'USE_P_SCALE', 'PERSISTENT_TYPE',
    'persistent_atomic_counter', 'Num_CU', 'GRID_CU_MULTIP', 'Batch',
    'BLOCK_M', 'BLOCK_N', 'PRE_LOAD_V', 'NUM_XCDS',
]

_BWD_COMMON_TAIL = [
    'num_head_q', 'num_head_k', 'cu_seqlens_q', 'cu_seqlens_k', 'num_seqlens',
    'max_seqlen_q', 'max_seqlen_k', 'seq_strides_q', 'seq_strides_k',
    'hdim_qk', 'hdim_vo', 'dropout_p', 'philox_seed_ptr', 'philox_offset1',
    'philox_offset2', 'Window_left', 'Window_right', 'BLOCK_M', 'BLOCK_DMODEL',
    'BLOCK_N', 'CAUSAL_TYPE', 'ENABLE_DROPOUT', 'PADDED_HEAD', 'BIAS_TYPE',
    'NUM_XCDS',
]

BWD_DK_DV_PARAMS = [
    'Q', 'K', 'V', 'B', 'sm_scale', 'DO', 'DK', 'DV', 'L', 'D',
    'stride_qz', 'stride_qh', 'stride_qm', 'stride_qk',
    'stride_kz', 'stride_kh', 'stride_kn', 'stride_kk',
    'stride_vz', 'stride_vh', 'stride_vk', 'stride_vn',
    'stride_bz', 'stride_bh', 'stride_bm', 'stride_bn',
    'stride_doz', 'stride_doh', 'stride_dom', 'stride_dok',
    'stride_dkz', 'stride_dkh', 'stride_dkn', 'stride_dkk',
    'stride_dvz', 'stride_dvh', 'stride_dvk', 'stride_dvn',
] + _BWD_COMMON_TAIL

BWD_DQ_PARAMS = [
    'Q', 'K', 'V', 'B', 'sm_scale', 'DO', 'DQ', 'DB', 'L', 'D',
    'stride_qz', 'stride_qh', 'stride_qm', 'stride_qk',
    'stride_kz', 'stride_kh', 'stride_kn', 'stride_kk',
    'stride_vz', 'stride_vh', 'stride_vk', 'stride_vn',
    'stride_bz', 'stride_bh', 'stride_bm', 'stride_bn',
    'stride_doz', 'stride_doh', 'stride_dom', 'stride_dok',
    'stride_dqz', 'stride_dqh', 'stride_dqm', 'stride_dqk',
    'stride_dbz', 'stride_dbh', 'stride_dbm', 'stride_dbn',
] + _BWD_COMMON_TAIL


def fwd_kernel_stub():
    """Raw fwd attn_fwd stub (undescribed) — tests attach their own specs."""
    return stub('attn_fwd', FWD_PARAMS)


def bwd_dk_dv_stub():
    return stub('bwd_kernel_dk_dv', BWD_DK_DV_PARAMS)


def bwd_dq_stub():
    return stub('bwd_kernel_dq', BWD_DQ_PARAMS)


# ---- fake cited kernel (a trimmed attn_fwd) --------------------------------

from dataclasses import dataclass

import numpy as np

import aotriton.template_instantiation as ati
from aotriton.template_instantiation.describe import describe

MAIN_DTYPES = ['*fp16:16', '*bf16:16', '*fp32:16']

# The cited kernel's signature (logical args + Q strides). Trimmed from the real
# 74-param attn_fwd to the surface the cite tests exercise: the T_io dtype var, an
# encoded_softmax operand, the gap scalars/tensors debug inherits, dropout PRNG,
# causal, and the perf constexprs.
_ATTN_FWD_PARAMS = [
    'Q', 'stride_qz', 'stride_qh', 'stride_qm', 'stride_qk',
    'encoded_softmax',
    'Sm_scale',
    'Num_head_q', 'Max_seqlen_q', 'Max_seqlen_k',
    'BLOCK_DMODEL', 'BIAS_TYPE',
    'ENABLE_DROPOUT', 'dropout_p',
    'philox_seed_ptr', 'philox_offset1', 'philox_offset2',
    'philox_seed_output', 'philox_offset_output',
    'CAUSAL_TYPE',
    'BLOCK_M', 'BLOCK_N',
]


@dataclass
class _AttnFwdPerf:
    # Minimal perf schema; tunable because configs() yields one.
    BLOCK_M: np.int16 = 16
    BLOCK_N: np.int16 = 16


def _gen_configs(f):
    yield ati.tune.Config({'BLOCK_M': 16, 'BLOCK_N': 16}, num_warps=4, num_stages=1)


def _fake_disabled(f):
    # A real (non-lambda) disable so citing kernels must affirm any override.
    return f.arch == 'gfx1100' and f.choices.BLOCK_DMODEL > 256


def attn_fwd_stub():
    """A freshly-described trimmed 'attn_fwd', usable as a cite target.

    Each call returns its own described stub (specs mutate the object), so tests
    that mutate the spec (e.g. replacing disables) do not pollute each other.
    """
    k = stub('attn_fwd', _ATTN_FWD_PARAMS)
    specs = [
        ati.type_var('T_io', dtype=MAIN_DTYPES, signature_name='Q'),
        ati.type_var('T_u64', dtype=['*u64']),
        ati.tensor('Q', 'T_io', strides='stride_q?', contiguous=-1),
        ati.tensor('encoded_softmax', 'T_io', rank=4),
        ati.scalar('Sm_scale', 'fp32'),
        ati.scalar(['Num_head_q', 'Max_seqlen_q', 'Max_seqlen_k'], 'i32'),
        ati.scalar('BLOCK_DMODEL', options=[16, 64, 128]),
        ati.scalar('BIAS_TYPE', options=[0, 1]),
        ati.scalar('ENABLE_DROPOUT', options=[False, True]),
        ati.scalar('dropout_p', 'fp32'),
        ati.tensor(['philox_seed_ptr', 'philox_offset1',
                    'philox_seed_output', 'philox_offset_output'], 'T_u64', rank=0),
        ati.scalar('philox_offset2', 'u64'),
        ati.scalar('CAUSAL_TYPE', options=[0, 3]),
        ati.tune.schema(_AttnFwdPerf),
        ati.tune.configs(_gen_configs),
        # Matches real attn_fwd: the ENABLE_DROPOUT derive spans dropout_p + all
        # philox args incl. *_output. A citing kernel that lacks any target (debug
        # has no philox_*_output) does NOT inherit it (cite: ALL targets must exist).
        ati.derives(['dropout_p', 'philox_seed_ptr', 'philox_offset1',
                     'philox_offset2', 'philox_seed_output', 'philox_offset_output'],
                    to=0, when=ati.eq('ENABLE_DROPOUT', False)),
        ati.derives('encoded_softmax', to=0, when=ati.eq('CAUSAL_TYPE', 0)),
        ati.disable(when=_fake_disabled),
    ]
    describe(k, *specs, _validate=False)
    return k
