# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The `@ati.metro_kernel` decorator (pipeline Stage 1).

Innermost marker of a metro stacked-@ block: transpiles the body (never executed)
into a MetroPlan and ACCUMULATES it onto the pending list so @ati.start can
finalise the whole stack (including @ati.hints.union_precedence above it). A metro
wires an operator's collaborating kernels with ordinary Python if/else:

    @ati.start
    @ati.hints.union_precedence([kernel_a, kernel_b])   # optional, above
    @ati.metro_kernel                                    # innermost
    def metro_fwd(params):
        attn_fwd(params)
        if params.encoded_softmax.data_ptr() != 0:
            debug_simulate_encoded_softmax(params)
"""

from ..specs.metro import transpile


def metro_kernel(fn):
    """@ati.metro_kernel: transpile the function body (never executed) into a
    MetroPlan (a StackedSpec), accumulate it onto fn's pending list, and return fn
    so @ati.start above receives the original function object."""
    plan = transpile(fn)
    return plan(fn)          # MetroPlan.__call__(fn) → accumulate_spec(plan, fn) → fn
