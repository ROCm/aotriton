# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI descriptions of the flash bwd_preprocess / bwd_preprocess_varlen kernels
(executive plan agent-plans/ati_aux-kernel-xref_exec0.md Step 13).

Auxiliary bwd kernels: they collaborate in the bwd metro and cite it for the
shared-operand practices, declaring only what is unique to them. Their real Triton
arguments carry several apparel renames (rev1 §4.3):
  * Delta   -> D            (the row-sum output; wires_to=)
  * D_HEAD  -> BLOCK_DMODEL  (the head-dim constexpr; wires_to=)
  * stride_on  -> stride_ok / stride_don -> stride_dok (cosmetic stride comments;
    rev1 §3.5, the access expression is identical)

Perf is schema-only (BLOCK_M=128, no configs) -> untunable.

Parity target: v3python/rules/flash/bwd_preprocess.py + op_attn_bwd.py.
"""

import os
from dataclasses import dataclass

import numpy as np

import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))

import v3python.template_instantiation as ati


def _block_dmodel_values():
    env = os.getenv('AOTRITON_FLASH_BLOCK_DMODEL',
                    default='16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 512')
    return [int(d) for d in env.split(',')]


MAIN_DTYPES = ['*fp16:16', '*bf16:16', '*fp32:16']


@dataclass
class BwdPreprocessPerf:
    BLOCK_M: np.int16 = 128


def _common_specs(varlen: bool):
    specs = [
        # Cite a bwd key kernel for the shared-operand practices (cu_seqlens_q,
        # num_seqlens, max_seqlen_q, hdim_vo). A 3-segment cite resolves via the
        # flat kernel registry when op_attn_bwd is not yet built (it is the metro
        # that calls this preprocess — citing the whole metro would be circular).
        ati.cite('op_attn_bwd.triton_split.bwd_kernel_dk_dv'),
        # main tensors: Out and DO (rank 4), Delta dressed as D (rank 2 lazy).
        ati.tensor_dtype('T_io', dtype=MAIN_DTYPES, signature_name='Out'),
        ati.tensor('Out', 'T_io', strides='stride_o?', contiguous=-1),
        ati.tensor('DO',  'T_io', strides='stride_do?', contiguous=-1),
        ati.tensor('Delta', 'LazyTensor:*fp32:16', rank=2, wires_to='D'),
        # head-dim constexpr: D_HEAD dressed as BLOCK_DMODEL (the enumerated axis).
        ati.scalar('D_HEAD', options=_block_dmodel_values(), wires_to='BLOCK_DMODEL'),
        ati.scalar('PADDED_HEAD', options=[False, True]),
        # perf: schema-only -> untunable.
        ati.tune.schema(BwdPreprocessPerf),
        ati.tune.fallback(PADDED_HEAD=False),
    ]
    return specs


def describe_bwd_preprocess(kernel):
    # cu_seqlens_q / num_seqlens / max_seqlen_q / hdim_vo are GAPS inherited from
    # the cited metro by apparel name.
    ati.describe(kernel, *_common_specs(varlen=False))
    return kernel


def describe_bwd_preprocess_varlen(kernel):
    # The varlen variant additionally takes seq_strides_q (a gap from the cite);
    # everything else is the same shape.
    ati.describe(kernel, *_common_specs(varlen=True))
    return kernel
