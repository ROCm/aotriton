# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI description of debug_simulate_encoded_softmax (executive plan
agent-plans/ati_aux-kernel-xref_exec0.md Step 11).

This is an AUXILIARY kernel: it collaborates in the forward metro and borrows the
fwd operator's instantiation practices via @ati.cite, declaring only what is unique
to it. Its real Triton argument `R` is the encoded-softmax output tensor, dressed
as the operator operand `encoded_softmax` (wires_to=). Everything else — the
dropout / PRNG / seqlen scalars — is a GAP filled from the cited attn_fwd by
apparel name. Perf is schema-only (BLOCK_M=64, BLOCK_N=32, no configs) so the
kernel is untunable and only its default perf is compiled.

Compare the legacy hand-written rules/flash/debug_simulate_encoded_softmax.py: the
`encoded_softmax` ARGUMENTS hack, the manually-copied TYPE_CHOICES, and the rank
tables all disappear — they are inherited through the cite.
"""

from dataclasses import dataclass

import numpy as np

import v3python.template_instantiation as ati


@dataclass
class DebugSimulateEncodedSoftmaxPerf:
    # Fixed perf (the kernel is untunable: schema without configs). Matches the
    # legacy PERF_CHOICES BLOCK_M=[64], BLOCK_N=[32].
    BLOCK_M: np.int16 = 64
    BLOCK_N: np.int16 = 32


def describe_debug_simulate_encoded_softmax(debug_kernel):
    specs = [
        # Cite the forward metro's key kernel for the shared operands' practices.
        ati.cite('op_attn_fwd.triton.attn_fwd'),

        # The only locally-declared argument: R (the encoded-softmax output),
        # dressed as the operator operand encoded_softmax. Stride-bearing, so it is
        # declared here with its own glob (rev0 §5), not inherited. Its dtype is
        # T_io, the shared dtype variable defined by attn_fwd and reached via cite.
        ati.tensor('R', 'T_io', strides='stride_r?', contiguous=-1,
                   wires_to='encoded_softmax'),

        # Schema-only perf -> untunable; only the default perf is compiled.
        ati.tune.schema(DebugSimulateEncodedSoftmaxPerf),

        # dropout_p, Num_head_q, Max_seqlen_q, Max_seqlen_k, philox_seed_ptr,
        # philox_offset1, philox_offset2 are GAPS — inherited from the cite by
        # apparel name. No local declarations needed.
    ]
    ati.describe(debug_kernel, *specs)
    return debug_kernel
