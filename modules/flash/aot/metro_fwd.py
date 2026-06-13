# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI description of the flash forward metro (executive plan
agent-plans/ati_aux-kernel-xref_exec0.md Step 9).

A metro kernel implements one operator functional with a set of collaborating
sub-kernels. The forward metro runs attn_fwd, then — when an encoded-softmax
output tensor is provided — the debug_simulate_encoded_softmax kernel. Argument
wiring (debug's R -> encoded_softmax) lives on the sub-kernel's own wires_to=
decorator, NOT here (rev0 §4.3), so the call is a plain `kernel(params)`.

The @ati.metro_kernel transpiler parses this body (never executes it) into a
MetroPlan of Call/Cond steps; lower_plan turns that into the existing
MetroKernel/ConditionalKernel IR. The condition
`params.encoded_softmax.data_ptr() != 0` lowers to the legacy C++ string
`encoded_softmax ->data_ptr() != nullptr`.
"""

import aotriton.template_instantiation as ati


@ati.metro_kernel
def metro_fwd(params):
    attn_fwd(params)
    if params.encoded_softmax.data_ptr() != 0:
        debug_simulate_encoded_softmax(params)
