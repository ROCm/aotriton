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

import aotriton.template_instantiation as ati
from ._common import flash_disabled


@dataclass
class BwdKernelDqPerf:
    BLOCK_M:  np.int16 = 16
    BLOCK_N:  np.int16 = 16
    NUM_XCDS: np.int8 = 1


def _bwd_disabled(f):
    """Shared flash disable predicate; bwd gfx950-bad head dims are {48, 80}."""
    return flash_disabled(f, gfx950_bad_hdims={48, 80})


def gen_autotune_configs(f):
    """Placeholder generator (one valid config); the DB path does not use it."""
    kw = {'BLOCK_M': 16, 'BLOCK_N': 16,
          'NUM_XCDS': 8 if f.arch in ('gfx942', 'gfx950') else 1,
          'waves_per_eu': 1}
    yield ati.tune.Config(kw, num_warps=4, num_stages=1)


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
