# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ContextHelper — a `wires_to=` value naming a host-side C++ member function instead
of an operand rename.

`wires_to` on `@ati.tensor` / `@ati.scalar` has historically been a plain operand
name (the real→apparel wiring: this kernel argument IS that operator operand, under
a different name). `ir/triton/kdesc.py`'s `apparel_of` docstring already anticipates
a second representation: "the apparel value is a plain operand name for now; the
representation is kept opaque so it can later carry a tuple of operator params or an
expression." `ContextHelper` is that second representation.

It names a member function on the generated CONTEXT class (e.g. `FlycAttnFwdContext`)
that the author hand-implements in `modules/flash/csrc/`, exactly as
`grid_calculator()` already is:

    wires_to=ati.context_helper('flyc_num_seqlens')
      -> declares  int32_t flyc_num_seqlens() const;  on the context class
      -> author implements it by hand in modules/flash/csrc/<kernel>.cc

Use it when the kernel argument is not any operator operand under another name —
when it has to be COMPUTED from the operands the operator does have. A rename stays
a rename: `wires_to='Varlen_bits'` is the right spelling for an argument that IS
that operand, and routing such an argument through a helper would only add a
function whose body is `return params.Varlen_bits;`.

Lives in `ir/`, not `decorators/`: it is not a decorator (never stacked on a def via
`@`) — it is a *value* passed to `wires_to=`, stored on the Tensor/ScalarSpec and
read back through the apparel-wiring machinery. It is also not language-specific
(not under `ir/triton/` or `ir/flyc/`): any backend needing host-side translation for
an argument that is not a plain rename can use it.

**The contract the generated caller rests on, which this class does not enforce.**
`codegen/flyc.py:codegen_context_helper_evaluate` runs every context helper for a
description exactly once, in declaration order, before `lookup_optimal` computes its
own `arch_number`/`mod_number` or binds `perf()`. That is only sound while every
helper is a pure function of `params` and `current_gpu` alone, and helpers are
mutually independent: no helper may read `perf()` or another helper's result. That
holds for every helper written so far, but it is a property of those helpers rather
than of this type, and breaking it fails SILENTLY — a helper reading `perf()` gets a
default-constructed Pon, one reading another helper's scratch member gets that
member's zero-initialised value rather than its real result. Hence writing it down
here instead of leaving it to be rediscovered by debugging a wrong answer. The day a
helper genuinely needs `perf()` or another helper's output, replace the flat
evaluation with a topological sort over declared dependencies, or with lazy memoised
accessors. `codegen/template/flyc.cc` restates this at the
`[[context_helper_evaluate]]` site, for a reader who is looking at the C++.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ContextHelper:
    """A `wires_to=` value naming a host-side context member function, not an
    operand rename. `name` is the C++ member function name (e.g.
    `flyc_num_seqlens`), declared with no arguments and a return type taken from
    the `@ati.scalar`/`@ati.tensor` this value is attached to."""

    name: str

    def __post_init__(self):
        assert isinstance(self.name, str) and self.name, \
            f'ati.context_helper needs a non-empty member function name, got {self.name!r}'

    def __repr__(self):
        return f'ContextHelper({self.name!r})'


def context_helper(name: str) -> ContextHelper:
    """`ati.context_helper('flyc_num_seqlens')` — see module docstring."""
    return ContextHelper(name)
