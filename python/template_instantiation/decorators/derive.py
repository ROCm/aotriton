# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
`ati.derives` (alias `ati.overrides`) — derive an argument's or perf param's value
from functional state, plus the re-exported predicate builders (ati.eq/ne/lt/gt).
"""

from ..ir import Override, VarRef, ValueFn
from ..ir import eq, ne, lt, gt, le, ge   # predicate builders, re-exported as ati.*


def _ALWAYS(functional):
    return True


def derives(targets, *, to, when=None):
    """Derive `targets`' value from other functional state (agent-plans/ati_rev1.md
    §3.3). The single facade for both derive channels — the builder routes by
    target:
      * a kernel ARGUMENT target  -> applied in resolved[] (compiled signature),
        the former conditional/CC/CDETensor case (B, dropout_p, Hdim_qk, ...);
      * a PERF-SCHEMA target       -> applied in the perf layer (PERSISTENT_TYPE,
        NUM_XCDS), the former PROGRAMMATIC_PERFS case.

    `to` selects the value kind:
      * str          -> VarRef (copy another variable's choice, e.g. BLOCK_DMODEL)
      * callable     -> compute the value from the functional, `to(f)` — for a
                        value that is a function of functional state (e.g. NUM_XCDS
                        from arch, possibly several values 1/3/6/8). Fires
                        unconditionally unless `when` is also given.
      * anything else -> literal (TypedChoice.parse handles ints/bools/floats; `0` on a
                         tensor target is the constexpr-zero / former-CDETensor case)

    `when` is an optional predicate gating the derive: a structured ati.eq/ne/...
    over a free choice axis, or a callable `f -> bool`. Omit it (callable `to`) to
    always fire.

      ati.derives('encoded_softmax', to=0, when=ati.eq('RETURN_ENCODED_SOFTMAX', False))
      ati.derives('Hdim_qk', to='BLOCK_DMODEL', when=ati.eq('PADDED_HEAD', False))
      ati.derives('PERSISTENT_TYPE', to=2, when=ati.ne('CAUSAL_TYPE', 0))
      ati.derives('NUM_XCDS', to=lambda f: {'gfx942': 8, 'gfx950': 8}.get(f.arch, 1))
    """
    if callable(to) and not isinstance(to, VarRef):
        value = ValueFn(to)
    elif isinstance(to, str):
        value = VarRef(to)
    else:
        value = to
    predicate = when if when is not None else _ALWAYS
    return Override(targets, predicate, value)


# Back-compat alias; `derives` is the preferred facade.
overrides = derives
