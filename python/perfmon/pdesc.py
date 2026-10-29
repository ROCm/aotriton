# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
PerfDescription: what a family plugin (`modules/<family>/perfmon/`) tells
perfmon about its operators.

perfmon works at operator granularity: one entry (a problem plus all its
sizes) is measured per iface, for the unforced operator and for each forced
backend. This is deliberately not tune's TuningDescription, which works on
kernels and HSACOs.

A family exports a subclass as `PerfDesc`. Family code may import this module
(and `aotriton.utils`), but never `aotriton.tune`, `aotriton.pgdef`, torch or
triton, so this module imports nothing beyond the standard library.

Entries are dataclasses (`ENTRY_CLASS`) whose fields are the family's
`PROBLEM_COLUMNS` followed by `SIZE_COLUMNS`. The helpers below convert them
to and from the payload (JSON) and the rows of the perfdb entry list
(`.metadata/problems.csv` + `entries.csv`), so every family spells its cells
the same way.
"""

import dataclasses
import typing
from dataclasses import dataclass
from typing import Any, Iterable

#: The sizes of an entry, in `entries.csv` order. Every family has these four;
#: what differs between families is the problem.
SIZE_COLUMNS = ('seqlen_q', 'seqlen_k', 'batch', 'n_heads_q')

#: The forms an LLM case's `phase` may take.
PHASES = ('fwd', 'fwd_bwd')


def canonical_cell(value) -> str:
    """One CSV cell, spelled canonically, so that matching rows by content
    does not depend on how a value was typed (0.0 vs 0, True vs 1)."""
    if isinstance(value, bool):
        return '1' if value else '0'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value or value in (float('inf'), float('-inf')):
            raise ValueError(f'not a finite number: {value!r}')
        return str(int(value)) if value.is_integer() else repr(value)
    if isinstance(value, str):
        if not value or any(c in value for c in ',\n\r"') or value != value.strip():
            raise ValueError(f'cell {value!r} cannot be written to a CSV row unquoted')
        return value
    raise TypeError(f'unsupported cell type {type(value).__name__}: {value!r}')


def coerce(value, typ):
    """`value` (from JSON or a CSV cell) as the field type `typ`."""
    if typ is bool:
        if isinstance(value, str):
            return value.strip() not in ('0', '', 'False', 'false')
        return bool(value)
    if typ is int:
        if isinstance(value, str):
            return int(value)
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f'{value!r} is not an integer')
        return int(value)
    if typ is float:
        return float(value)
    if typ is str:
        return str(value)
    raise TypeError(f'unsupported field type {typ!r}')


@dataclass(frozen=True)
class LLMCase:
    """One row of the family's workloads list: an entry with the workload's
    own sizes. `model` labels it in rendered tables and is not part of the
    entry; `pending` rows are listed but not dispatched yet."""
    model: str
    phase: str
    entry: Any
    pending: bool = False


@dataclass(frozen=True)
class PrimeTable:
    """One rendered prime table: rows by `row_label`, columns by
    `column_label`. Each cell is a pair of entries, `main` and `alt` (e.g.
    bf16 and fp16), either of which may be None; the cell shows main's TFLOPS
    and, in parentheses, the alt-minus-main difference."""
    title: str
    row_label: str
    column_label: str
    main_label: str
    alt_label: str
    columns: tuple
    rows: tuple    # ((row value, ((main entry, alt entry), ... one per column)), ...)


class PerfDescription:
    """The family interface. Subclasses set the class attributes and
    implement the methods that raise NotImplementedError."""

    #: The problem columns, `problems.csv`'s header.
    PROBLEM_COLUMNS: tuple[str, ...] = ()
    SIZE_COLUMNS: tuple[str, ...] = SIZE_COLUMNS
    #: A dataclass with fields PROBLEM_COLUMNS + SIZE_COLUMNS, in that order.
    ENTRY_CLASS: type = type(None)

    # -- what the family must provide ------------------------------------------

    def ifaces(self) -> tuple[str, ...]:
        """The operators measured, by interface name."""
        raise NotImplementedError

    def phase_ifaces(self, phase: str) -> tuple[str, ...]:
        """The ifaces an LLM case of `phase` is measured on."""
        raise NotImplementedError

    def entry_sets(self) -> dict[str, dict]:
        """{'prime': axes, 'coverage': axes}: each maps an axis name to its
        values. Dispatch offers one option per axis; the axes are the
        source of truth."""
        raise NotImplementedError

    def entries(self, axes: dict, arch: str | None = None) -> Iterable:
        """Expand `axes` into entries with their sizes (the size rule).
        Problems the size rule cannot fit are not generated."""
        raise NotImplementedError

    def llm_cases(self) -> list[LLMCase]:
        """The `llm` entry set, in the order of the family's workloads list."""
        raise NotImplementedError

    def validate_hw_feature(self, arch: str, entry) -> tuple[bool, str]:
        """(ok, reason): whether `arch` can run `entry` at all. A rejected
        entry is recorded as excluded, not silently dropped."""
        raise NotImplementedError

    def size(self, problem, seqlen_q: int, seqlen_k: int) -> tuple[int, int] | None:
        """The size rule: (batch, n_heads_q) for a grid entry, or None when the
        problem does not fit the budget at these seqlens. Dispatch only."""
        raise NotImplementedError

    def input_bytes(self, entry, iface: str) -> int:
        """The byte model: device memory the entry's buffers need on `iface`."""
        raise NotImplementedError

    def fits_vram(self, entry, iface: str, vram_total_bytes: int) -> tuple[bool, str]:
        """The worker's fit check against the GPU's actual VRAM."""
        raise NotImplementedError

    def tflops(self, iface: str, entry, time_us: float) -> float:
        """Matrix-core TFLOPS of one call taking `time_us`. Rendering only."""
        raise NotImplementedError

    def entry_pon(self, entry) -> str:
        """The entry's wire form for the runner."""
        raise NotImplementedError

    def prime_tables(self, iface: str) -> list[PrimeTable]:
        """The prime tables of `iface`, for rendering."""
        raise NotImplementedError

    def case_columns(self, entry) -> dict[str, str]:
        """{column header: text} describing an LLM case's entry in a table."""
        raise NotImplementedError

    # -- shared helpers ----------------------------------------------------------

    @classmethod
    def _field_types(cls) -> dict[str, type]:
        hints = typing.get_type_hints(cls.ENTRY_CLASS)
        return {f.name: hints[f.name] for f in dataclasses.fields(cls.ENTRY_CLASS)}

    @classmethod
    def entry_fields(cls) -> tuple[str, ...]:
        return tuple(f.name for f in dataclasses.fields(cls.ENTRY_CLASS))

    def entry_from_dict(self, d: dict):
        """An entry from its payload form (JSON), with every field coerced to
        its declared type. Unknown or missing fields are errors."""
        types = self._field_types()
        unknown = set(d) - set(types)
        if unknown:
            raise ValueError(f'unknown entry field(s) {sorted(unknown)}')
        return self.ENTRY_CLASS(**{name: coerce(d[name], typ) for name, typ in types.items()})

    def entry_to_dict(self, entry) -> dict:
        return dataclasses.asdict(entry)

    def problem_row(self, entry) -> tuple[str, ...]:
        """The entry's `problems.csv` row."""
        return tuple(canonical_cell(getattr(entry, c)) for c in self.PROBLEM_COLUMNS)

    def size_row(self, entry) -> tuple[str, ...]:
        """The entry's sizes, as in `entries.csv` (after the problem ordinal)."""
        return tuple(canonical_cell(getattr(entry, c)) for c in self.SIZE_COLUMNS)

    def entry_from_rows(self, problem_row, size_row):
        """The inverse of problem_row() + size_row()."""
        types = self._field_types()
        cells = dict(zip(self.PROBLEM_COLUMNS, problem_row))
        cells.update(zip(self.SIZE_COLUMNS, size_row))
        return self.ENTRY_CLASS(**{name: coerce(cells[name], typ) for name, typ in types.items()})
