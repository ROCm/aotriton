# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
KernelSignature — the per-compiled-instance signature of a flyc kernel.

A thin, STANDALONE class calling `ir/lib/naming.py` directly — it does NOT reuse or
subclass `ir/triton/ksignature.py`'s `KernelSignature`. That class carries Triton
vocabulary (`num_warps` / `num_stages` / `waves_per_eu`, `COMPILER_OPTIONS`, the
gfx1250 double-warps workaround) that is specific to Triton's autotune model and
does not apply here.

flyc's perf vocabulary has no C struct and no grid choice among candidate
images -- every functional resolves to exactly one hsaco, whose distinguishing
knob set (all 23 `FmhaKnobs` fields, as returned by the description alongside
its deferred builder) is carried verbatim in the entry name's `#P` section as a
PON (Plain / Python Object Notation) ';'-separated 'k=v' string, read at runtime
by `class Pon` (`include/aotriton/_internal/pon.h`) rather than by a generated
struct and accessors. `copt_section` stays permanently empty -- flyc has no
compiler-option grid.
"""

from functools import cached_property

from aotriton.utils import render_pon

from ..lib import naming as lib_naming


class KernelSignature:
    """The perf + compiler-option signature of one compiled flyc kernel instance
    (one Functional). `psel_section` renders the knob dict (all 23 `FmhaKnobs`
    fields, via `asdict(knobs)`) as `k=v;k=v`; `copt_section` is always
    empty (see module docstring). Both names are the ones the Triton signature
    uses, so the two languages describe the entry name's sections alike."""

    def __init__(self, f: 'Functional', *, psels: dict | None = None, copts: dict | None = None):
        self._functional = f
        # The knob dict the description returned, kept verbatim. NEVER read back
        # from the on-disk <hsaco>.json: that file is produced later, at true
        # build time, by a different process.
        self._psels = dict(psels) if psels else {}
        # flyc has no compiler-option grid; the parameter exists so the two
        # sections stay symmetric with Triton's and with `copt_section` below.
        self._copts = dict(copts) if copts else {}

    @property
    def psel_section(self) -> str:
        return render_pon(self._psels)

    @property
    def copt_section(self) -> str:
        return render_pon(self._copts)

    @cached_property
    def hsaco_entry_name(self) -> str:
        return lib_naming.entry_name(self._functional,
                                     perf=self.psel_section,
                                     copt=self.copt_section)

    def blake2b_hash(self, package_path):
        return lib_naming.blake2b_hash(package_path, self.hsaco_entry_name)
