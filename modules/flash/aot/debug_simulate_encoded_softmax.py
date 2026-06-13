# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI description of debug_simulate_encoded_softmax (auxiliary fwd kernel).

It collaborates in the forward metro and cites the fwd metro's key kernel for the
shared operands; only its real argument `R` (the encoded-softmax output, dressed as
the operand `encoded_softmax`) and a schema-only perf are declared locally. The
dropout / PRNG / seqlen scalars are GAPS filled from the cite. Stacked-@ form
(rev0 §5a) over ../kernel/dropout_rng.py.
"""

from dataclasses import dataclass

import numpy as np

import aotriton.template_instantiation as ati


@dataclass
class DebugSimulateEncodedSoftmaxPerf:
    # Fixed perf (untunable: schema without configs). Legacy BLOCK_M=[64], BLOCK_N=[32].
    BLOCK_M: np.int16 = 64
    BLOCK_N: np.int16 = 32


@ati.kernel
@ati.cite('op_attn_fwd.triton.attn_fwd')
# R (encoded-softmax output) dressed as the operand encoded_softmax; stride-bearing
# so declared locally. Its dtype T_io is inherited from the cite.
@ati.tensor('R', 'T_io', strides='stride_r?', contiguous=-1,
            wires_to='encoded_softmax')
# Schema-only perf -> untunable; only the default perf is compiled.
@ati.tune.schema(DebugSimulateEncodedSoftmaxPerf)
# dropout_p, Num_head_q, Max_seqlen_q/k, philox_* are GAPS inherited from the cite.
@ati.source('../kernel/dropout_rng.py')
def debug_simulate_encoded_softmax():
    pass
