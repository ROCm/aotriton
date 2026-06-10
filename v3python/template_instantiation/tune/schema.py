# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Perf schema (executive plan Step 3.1; agent-plans/ati_rev1.md §6).

`ati.tune.schema(AttnFwdPerf)` reads a @dataclass whose fields are the kernel's
performance parameters, each annotated with the numpy dtype that fixes its C
struct width:

    @dataclass
    class AttnFwdPerf:
        BLOCK_M:         np.int16
        BLOCK_N:         np.int16
        PRE_LOAD_V:      bool
        NUM_XCDS:        np.int8
        GRID_CU_MULTIP:  np.int8
        PERSISTENT_TYPE: np.int8

Each field becomes a PerfParam carrying the constexpr TypedChoice CLASS for that
declared dtype (np.int16 -> constexpr.int16_t, bool -> constexpr.bool_t). The
declared width is authoritative — unlike the functional GuessInt path that picks
the smallest type fitting the values, perf packing must be stable regardless of
which config values appear, matching today's
`PERF_CHOICES = { frozenset(['BLOCK_M']): np.array([...], np.int16) }` trick.
"""

import dataclasses

import numpy as np

from ...base.typed_choice import constexpr as TCC

# numpy dtype -> constexpr TypedChoice class (the perf field's C width).
_NP_TO_TCC = {
    np.int8: TCC.int8_t,
    np.int16: TCC.int16_t,
    np.int32: TCC.int32_t,
    np.int64: TCC.int64_t,
    np.uint8: TCC.uint8_t,
    np.uint16: TCC.uint16_t,
    np.uint32: TCC.uint32_t,
    np.uint64: TCC.uint64_t,
}


def _resolve_tcc(field_name, annotation):
    """Map a dataclass field annotation to its constexpr TypedChoice class."""
    if annotation is bool or annotation is np.bool_:
        return TCC.bool_t
    # numpy scalar types may be given as the type (np.int16) or dtype(np.int16).
    np_type = None
    if isinstance(annotation, type) and issubclass(annotation, np.generic):
        np_type = annotation
    else:
        try:
            np_type = np.dtype(annotation).type
        except TypeError:
            np_type = None
    if np_type in _NP_TO_TCC:
        return _NP_TO_TCC[np_type]
    raise TypeError(
        f"perf schema field {field_name!r}: annotation {annotation!r} is not a "
        f"supported perf dtype; use a numpy integer dtype (np.int8/int16/...) or "
        f"bool")


class PerfParam:
    """One performance parameter: its name and the constexpr TypedChoice class
    fixing its C struct width. A concrete value (from a tuning Config) is bound
    later via tcc(value)."""
    __slots__ = ('name', 'tcc')

    def __init__(self, name, tcc):
        self.name = name
        self.tcc = tcc          # a TCC.*_t class

    def choice_for(self, value):
        """A settled constexpr TypedChoice of this param's declared width."""
        return self.tcc(value)

    @property
    def itype(self) -> str:
        # Width-only instance to read the C itype (value is irrelevant).
        return self.tcc(0).itype

    def __repr__(self):
        return f'PerfParam({self.name!r}, {self.tcc.__name__})'


class PerfSchema:
    """The ordered perf parameters of a kernel, derived from a @dataclass."""
    __slots__ = ('dataclass', 'params')

    def __init__(self, dataclass, params):
        self.dataclass = dataclass
        self.params = params            # list[PerfParam], field order

    @property
    def names(self):
        return [p.name for p in self.params]

    def __repr__(self):
        return (f'PerfSchema({self.dataclass.__name__!r}, '
                f'{[p.name for p in self.params]})')


def build_schema(dataclass) -> PerfSchema:
    assert dataclasses.is_dataclass(dataclass), \
        f'ati.tune.schema expects a @dataclass, got {dataclass!r}'
    params = [PerfParam(f.name, _resolve_tcc(f.name, f.type))
              for f in dataclasses.fields(dataclass)]
    return PerfSchema(dataclass, params)


def schema(dataclass):
    """ati.tune.schema(PerfDataclass): build the PerfSchema. (Attaching it to a
    kernel as a stacked-@ decorator is wired in Step 3.2.)"""
    return build_schema(dataclass)
