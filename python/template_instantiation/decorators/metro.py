# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The `@ati.metro_kernel` decorator (pipeline Stage 1).

Transpiles the decorated function's body (never executed) into a MetroPlan spec
(specs/metro.py) attached as fn.__ati_metro__. A metro wires an operator's
collaborating kernels with ordinary Python if/else:

    @ati.metro_kernel
    def metro_fwd(params):
        attn_fwd(params)
        if params.encoded_softmax.data_ptr() != 0:
            debug_simulate_encoded_softmax(params)
"""

from ..specs.metro import transpile


def metro_kernel(fn):
    """@ati.metro_kernel: transpile the function body (never executed) into a
    MetroPlan, attached as fn.__ati_metro__. Returns the function untouched so the
    operator builder can read the plan."""
    fn.__ati_metro__ = transpile(fn)
    return fn
