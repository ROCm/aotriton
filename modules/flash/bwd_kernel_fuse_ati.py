# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI description of the flash bwd_kernel_fuse kernel (executive plan
agent-plans/ati_aux-kernel-xref_exec0.md Step 14).

bwd_kernel_fuse is an ALTERNATIVE bwd backend (a single fused kernel computing
dQ/dK/dV/dB) that is NOT called by the triton_split metro. It nonetheless reuses
the same bwd operand vocabulary, so it CITES THE WHOLE METRO
(op_attn_bwd.triton_split) to inherit the merged practices and redeclares almost
nothing — citation mode (b) of rev0 §4.4.

It diverges from the metro only where it must:
  * BLOCK_DMODEL is capped at <= 256 (legacy CHOICE_FILTERS) — declared locally,
    overriding the cited 12-value axis (-> 11 values);
  * its own perf schema (BLOCK_M, BLOCK_N; no NUM_XCDS).

Parity target: v3python/rules/flash/bwd_kernel_fuse.py + op_attn_bwd.py.
"""

import os
from dataclasses import dataclass

import numpy as np

import sys as _sys
from pathlib import Path as _Path
if str(_Path(__file__).resolve().parent) not in _sys.path:
    _sys.path.insert(0, str(_Path(__file__).resolve().parent))

import v3python.template_instantiation as ati


def _block_dmodel_values_capped():
    env = os.getenv('AOTRITON_FLASH_BLOCK_DMODEL',
                    default='16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 512')
    return [int(d) for d in env.split(',') if int(d) <= 256]


@dataclass
class BwdKernelFusePerf:
    BLOCK_M: np.int16 = 16
    BLOCK_N: np.int16 = 16


def gen_autotune_configs(f):
    """Placeholder generator (one valid config); the DB path does not use it."""
    yield ati.tune.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'waves_per_eu': 1},
                          num_warps=4, num_stages=1)


def describe_bwd_kernel_fuse(kernel):
    specs = [
        # Cite the bwd metro's sub-kernels for the merged operand practices (mode b:
        # the metro does not call this kernel). fuse spans all of them — dk_dv
        # (dK/dV), dq (dQ/dB), and preprocess (Out) — so it cites the three by name;
        # the resolver merges their practices. (A single whole-metro cite
        # "op_attn_bwd.triton_split" is equivalent but needs the operator built
        # first; the per-sub-kernel cites resolve via the flat kernel registry,
        # which the registration-ordering revision will later unify.)
        ati.cite('op_attn_bwd.triton_split.bwd_kernel_dk_dv'),
        ati.cite('op_attn_bwd.triton_split.bwd_kernel_dq'),
        ati.cite('op_attn_bwd.triton_split.bwd_preprocess'),

        # Out is declared LOCALLY: fuse names its strides stride_o{z,h,m,k} while the
        # cited preprocess names them stride_o{z,h,m,n}, so its strides cannot be
        # inherited by name (the cite carries EXACT cited stride names — see rev1
        # §5). Its dtype T_io is still inherited from the cite.
        ati.tensor('Out', 'T_io', strides='stride_o?', contiguous=-1),

        # Local override: cap BLOCK_DMODEL at <= 256 (legacy CHOICE_FILTERS); this
        # replaces the cited 12-value axis with the 11 surviving values.
        ati.scalar('BLOCK_DMODEL', options=_block_dmodel_values_capped()),

        # Own perf schema (no NUM_XCDS, unlike dk_dv/dq).
        ati.tune.schema(BwdKernelFusePerf),
        ati.tune.configs(gen_autotune_configs),
        ati.tune.binning(max_seqlen_q=ati.tune.binning.le,
                         max_seqlen_k=ati.tune.binning.le),
        ati.tune.fallback(PADDED_HEAD=False),

        # No local @ati.disable: fuse inherits the cited bwd kernels' disable
        # predicate (identical correctness exclusion) through the cite.
    ]
    ati.describe(kernel, *specs)
    return kernel
