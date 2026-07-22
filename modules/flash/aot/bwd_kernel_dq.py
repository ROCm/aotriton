# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI description of the flash bwd_kernel_dq kernel (computes dQ/dB).

A KEY backward kernel: a standalone full description, the dQ/dB analogue of
bwd_kernel_dk_dv. Stacked-@ form (rev0 §5a) over ../kernel/bwd_kernel_dq.py.

Note (rev1 §3.5): the B tensor's 3rd stride is `stride_bm` in the Triton signature
but the operator/golden call it `stride_bk`; ATI emits the real name in the cosmetic
pp_args comment (the access expression is identical).
"""

from dataclasses import dataclass

import numpy as np

import itertools

import aotriton.template_instantiation as ati
from aotriton.gpu_targets import AOTRITON_ARCH_WARPSIZE
from ._common import flash_disabled, check_value


@dataclass
class BwdKernelDqPerf:
    BLOCK_M:  np.int16 = 16
    BLOCK_N:  np.int16 = 16
    NUM_XCDS: np.int8 = 1


def _bwd_disabled(f):
    """Shared flash disable predicate; bwd gfx950-bad head dims are {48, 80}."""
    return flash_disabled(f, gfx950_bad_hdims={48, 80})


def gen_autotune_configs(f):
    """Per-functional performance config generator (ported from 0.12b). Feeds the
    tuning build (AOTRITON_BUILD_FOR_TUNING); the DB path does not use it."""
    arch = f.arch
    dtype = check_value(f, ['Q'])
    HEAD_DIM = check_value(f, ['BLOCK_DMODEL'])
    CAUSAL_TYPE = check_value(f, ['CAUSAL_TYPE'])
    BIAS_TYPE = check_value(f, ['BIAS_TYPE'])
    ENABLE_DROPOUT = check_value(f, ['ENABLE_DROPOUT'])
    WAVE32 = AOTRITON_ARCH_WARPSIZE[arch] == 32
    # TODO: right sizes for fp32?
    BLOCK_SIZES = [16, 32, 64] if dtype != '*fp32:16' else [16, 32]
    WAVES_PER_EU = [1, 2, 3, 4]
    NUM_WARPS = [4, 8] if WAVE32 else [2, 4]
    NUM_STAGES = [1]
    NUM_XCDS = 8 if arch in ('gfx942', 'gfx950') else 1
    if arch == 'gfx1250':
        # aiter gfx1250-MHA-DEFAULT.json: fwd.default, plus smaller backups.
        # Confirmed crash/NaN-free against the tuning DB (~/wkdir.aiday); keep
        # num_warps=4 here — num_warps=8 with BLOCK_M=BLOCK_N=64 crashes
        # unconditionally on this kernel.
        kw = {'BLOCK_M': 64, 'BLOCK_N': 64, 'waves_per_eu': 2, 'NUM_XCDS': NUM_XCDS}
        yield ati.tune.Config(kw, num_stages=1, num_warps=4)
        kw = {'BLOCK_M': 32, 'BLOCK_N': 32, 'waves_per_eu': 2, 'NUM_XCDS': NUM_XCDS}
        yield ati.tune.Config(kw, num_stages=1, num_warps=4)
        kw = {'BLOCK_M': 16, 'BLOCK_N': 16, 'waves_per_eu': 2, 'NUM_XCDS': NUM_XCDS}
        yield ati.tune.Config(kw, num_stages=1, num_warps=4)
        def more_configs():
            for M, N in ((32, 32), (32, 16), (16, 16)):
                for waves in (1, 2, 3, 4):
                    kw = {'BLOCK_M': M,
                          'BLOCK_N': N,
                          'waves_per_eu': waves,
                          'NUM_XCDS': NUM_XCDS}
                    for nwarps in (4, 8):
                        yield ati.tune.Config(kw, num_stages=1, num_warps=nwarps)
        # HEAD_DIM=256 fp32 causal+dropout (no bias) and HEAD_DIM=64 fp32 dropout (no
        # bias, either causal state) have no shipped candidate passing every UT. Add
        # two extra block-size options at the same baseline copts (nw4/we2) rather
        # than introducing new copts.
        if dtype == '*fp32:16' and BIAS_TYPE == 0 and ENABLE_DROPOUT and (
                (HEAD_DIM == 256 and CAUSAL_TYPE != 0) or HEAD_DIM == 64):
            yield from more_configs()
            return
        return
    for M, N, waves, warps, stages in itertools.product(BLOCK_SIZES,
                                                        BLOCK_SIZES,
                                                        WAVES_PER_EU,
                                                        NUM_WARPS,
                                                        NUM_STAGES):
        if M < N:
            continue  # deduplicate
        kw = {'BLOCK_M': M, 'BLOCK_N': N, 'waves_per_eu': waves,
              'NUM_XCDS': NUM_XCDS}
        yield ati.tune.Config(kw, num_stages=stages, num_warps=warps)


# Cite-based form (ati_linker_req acceptance demo): bwd_kernel_dq declares only what
# is UNIQUE to it — its dQ/dB outputs (DQ/DB strided tensors), its perf schema/configs/
# binning/fallback, and the bias-stride derive on its own DB — and cites the metro it
# is PART OF (op_attn_bwd.triton_split) for everything else. The shared inputs
# (Q/K/V/B/DO, sm_scale, L/D, the seqlen/head scalars, dropout/philox/window, the
# BLOCK_DMODEL/CAUSAL_TYPE/ENABLE_DROPOUT/PADDED_HEAD/BIAS_TYPE constexprs) and their
# shared derives are inherited as GAPS from the metro's other sub-kernel (dk_dv),
# matched by apparel name. This is a TRUE cycle (dq cites the metro that contains dq):
# the linker resolves it via the header/extern model — dq reads the OTHER sub-kernels'
# argument surface (known from Pass 1), never its own.
@ati.start
@ati.cite('op_attn_bwd.triton_split')
@ati.tensor('DQ', 'T_io', strides='stride_dq?', contiguous=-1)
@ati.tensor('DB', 'T_io', strides='stride_db?', contiguous=-1)
@ati.tune.schema(BwdKernelDqPerf)
@ati.tune.configs(gen_autotune_configs)
@ati.tune.binning(max_seqlen_q=ati.tune.binning.le,
                  max_seqlen_k=ati.tune.binning.le)
@ati.tune.fallback(PADDED_HEAD=False)
@ati.derives('NUM_XCDS', to=8, when=lambda f: f.arch in ('gfx942', 'gfx950'))
@ati.derives('DB', to=0, when=ati.eq('BIAS_TYPE', 0))          # strides cascade
@ati.source('../kernel/bwd_kernel_dq.py')
def bwd_kernel_dq():
    pass
