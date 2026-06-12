# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI description of the flash backward metro (the `triton_split` backend of
op_attn_bwd) — executive plan agent-plans/ati_aux-kernel-xref_exec0.md Step 12.

The bwd functional is implemented by three collaborating sub-kernels: a
preprocess (varlen vs padded, chosen by num_seqlens), then the two key kernels
dk_dv (dK/dV) and dq (dQ/dB). Argument wiring lives on each sub-kernel's
description; the metro body just sequences the calls.

`if params.num_seqlens > 0` lowers to the legacy ConditionalKernel condition
('num_seqlens', '> 0'); the else branch is bwd_preprocess (padded).
"""

import v3python.template_instantiation as ati


@ati.metro_kernel
def metro_bwd(params):
    if params.num_seqlens > 0:
        bwd_preprocess_varlen(params)
    else:
        bwd_preprocess(params)
    bwd_kernel_dk_dv(params)
    bwd_kernel_dq(params)
