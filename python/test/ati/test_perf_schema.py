# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit test for the @dataclass perf schema (executive plan Step 3.1).

ati.tune.schema synthesizes a perf struct: a dataclass whose fields are typed with the
constexpr TypedChoice classes (numpy replaced) and carry the python default. The struct
class IS the schema (param_names/itype_of/choice_for/default_value); an instance is a
perf bind row."""

import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

import aotriton.template_instantiation as ati
from aotriton.template_instantiation.specs.tune import PerfSchema, PerfStructBase


# Mirrors the real attn_fwd PERF_CHOICES widths: BLOCK_M/N int16,
# NUM_XCDS/GRID_CU_MULTIP/PERSISTENT_TYPE int8, PRE_LOAD_V bool. Every field declares a
# default (the untuned/empty-path value) — required by the schema.
@dataclass
class AttnFwdPerf:
    BLOCK_M:         np.int16 = 16
    BLOCK_N:         np.int16 = 16
    PRE_LOAD_V:      bool = False
    NUM_XCDS:        np.int8 = 1
    GRID_CU_MULTIP:  np.int8 = 2
    PERSISTENT_TYPE: np.int8 = 0


def _struct():
    sch = ati.tune.schema(AttnFwdPerf)
    assert isinstance(sch, PerfSchema)
    return sch.struct


def test_schema_builds_params_in_field_order():
    struct = _struct()
    assert issubclass(struct, PerfStructBase)
    assert struct.param_names() == ['BLOCK_M', 'BLOCK_N', 'PRE_LOAD_V', 'NUM_XCDS',
                                    'GRID_CU_MULTIP', 'PERSISTENT_TYPE']


def test_field_ctypes_match_perf_choices_packing():
    struct = _struct()
    itype = {n: struct.itype_of(n) for n in struct.param_names()}
    assert itype['BLOCK_M'] == 'int16_t'
    assert itype['BLOCK_N'] == 'int16_t'
    assert itype['NUM_XCDS'] == 'int8_t'
    assert itype['GRID_CU_MULTIP'] == 'int8_t'
    assert itype['PERSISTENT_TYPE'] == 'int8_t'
    assert itype['PRE_LOAD_V'] == 'bool'


def test_choice_for_keeps_declared_width():
    struct = _struct()
    # 256 fits in int16; the declared width is kept (not shrunk to int8) so the
    # struct layout is stable regardless of which config values appear.
    c = struct.choice_for('BLOCK_M', 256)
    assert c.itype == 'int16_t'
    assert c.triton_compile_signature == 256
    assert struct.choice_for('NUM_XCDS', 8).itype == 'int8_t'


def test_default_value_and_choice():
    struct = _struct()
    assert struct.default_value('BLOCK_M') == 16
    assert struct.default_choice('PRE_LOAD_V').triton_compile_signature is False


def test_bool_field():
    struct = _struct()
    c = struct.choice_for('PRE_LOAD_V', True)
    assert c.itype == 'bool'
    assert c.triton_compile_signature is True


def test_bind_row_items():
    struct = _struct()
    row = struct(**{n: struct.choice_for(n, struct.default_value(n))
                    for n in struct.param_names()})
    got = {n: tc.triton_compile_signature for n, tc in row.items()}
    assert got['BLOCK_M'] == 16
    assert got['PRE_LOAD_V'] is False


def test_unsupported_dtype_errors():
    @dataclass
    class Bad:
        x: float = 0.0        # not an integer/bool perf dtype
    try:
        ati.tune.schema(Bad)
    except TypeError as e:
        assert "'x'" in str(e)
        return
    raise AssertionError('expected unsupported-dtype TypeError')


def test_missing_default_errors():
    @dataclass
    class NoDefault:
        BLOCK_M: np.int16     # no default -> error
    try:
        ati.tune.schema(NoDefault)
    except TypeError as e:
        assert 'BLOCK_M' in str(e) and 'default' in str(e)
        return
    raise AssertionError('expected missing-default TypeError')


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} perf-schema tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
