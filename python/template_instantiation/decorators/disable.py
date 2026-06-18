# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
`ati.disable` — mark functionals excluded from generation (a compiler/numerical
correctness exclusion that travels with the kernel description; citeable per rev0 §4.5).
"""

import inspect

from ..specs.base import StackedSpec


def _is_callable_class_instance(when) -> bool:
    """True if `when` is an INSTANCE of a class defining __call__ (so it can extend
    a cited disable via super().__call__), as opposed to a bare function / lambda /
    method / builtin (inspect.isroutine) or a class object itself."""
    return (callable(when) and not inspect.isroutine(when)
            and not isinstance(when, type))


class DisableSpec(StackedSpec):
    """One @ati.disable(when=callable): a predicate over the functional marking it
    excluded from generation (compiler/numerical correctness exclusion). Multiple
    compose with OR. The callable reads functional state (f.arch, f.choices.<var>).
    This is the user interface to is_functional_disabled.

    `@ati.disable` is citeable (rev0 §4.5): a kernel with no local disable inherits
    the cited target's; a LOCAL disable REPLACES the cited one. To EXTEND (not
    replace) a cited disable, write it as a callable class, subclass it, and call
    super().__call__(f). `is_callable_class` records whether `when` structurally can
    do that; `override_ack` records the author's explicit affirmation that a bare
    callable intentionally overrides a cited disable (suppresses the §4.5 fatal
    error)."""

    __slots__ = ('when', 'is_callable_class', 'override_ack')

    def __init__(self, when, override_ack=False):
        assert callable(when), \
            f'@ati.disable(when=...) needs a callable f -> bool, got {when!r}'
        self.when = when
        self.is_callable_class = _is_callable_class_instance(when)
        self.override_ack = bool(override_ack)

    def holds(self, functional) -> bool:
        return bool(self.when(functional))

    def __repr__(self):
        return f'DisableSpec({getattr(self.when, "__name__", self.when)!r})'


def disable(when, *, I_understand_this_overrides_cited_disable=False):
    """Disable the functionals where `when(functional)` is truthy — a
    compiler/numerical correctness exclusion that travels with the kernel
    description (the user interface to is_functional_disabled):

      ati.disable(when=lambda f: f.choices.CAUSAL_TYPE and f.choices.BIAS_TYPE != 0)
      ati.disable(when=lambda f: f.arch == 'gfx950' and f.choices.BLOCK_DMODEL == 16)

    When the kernel also has an @ati.cite, a LOCAL disable replaces the cited one.
    A bare lambda/function cannot call super() to extend the cited predicate, so it
    silently drops it — the builder raises a FATAL error unless you affirm the
    override with `I_understand_this_overrides_cited_disable=True`. A callable-class
    instance (which can extend via super().__call__) is accepted without the flag.
    (rev0 §4.5)
    """
    return DisableSpec(when,
                       override_ack=I_understand_this_overrides_cited_disable)
