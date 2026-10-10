# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Fake debug_simulate_encoded_softmax: cites the fwd metro's key kernel,
declaring only its own R/encoded_softmax operand."""

from dataclasses import dataclass

import numpy as np

import aotriton.template_instantiation as ati


@dataclass
class DebugSimulateEncodedSoftmaxPerf:
    # Fixed perf (untunable: schema without configs). Legacy BLOCK_M=[64], BLOCK_N=[32].
    BLOCK_M: np.int16 = 64
    BLOCK_N: np.int16 = 32


@ati.start
@ati.cite('op_attn_fwd.triton.attn_fwd')
# The cited attn_fwd carries flash_disabled (reads CAUSAL_TYPE/BLOCK_DMODEL/BIAS_TYPE),
# which do not exist in this kernel's choice space. Replace with no_disable — this
# auxiliary kernel has no correctness exclusions of its own (rev0 §4.5).
@ati.no_disable()
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
