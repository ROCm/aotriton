# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit test for the @dataclass perf schema (executive plan Step 3.1)."""

import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

import v3python.template_instantiation as ati
from v3python.template_instantiation.tune.schema import PerfSchema, PerfParam


# Mirrors the real attn_fwd PERF_CHOICES widths (v3python/rules/flash/attn_fwd.py):
# BLOCK_M/N int16, NUM_XCDS/GRID_CU_MULTIP/PERSISTENT_TYPE int8, PRE_LOAD_V bool.
@dataclass
class AttnFwdPerf:
    BLOCK_M:         np.int16
    BLOCK_N:         np.int16
    PRE_LOAD_V:      bool
    NUM_XCDS:        np.int8
    GRID_CU_MULTIP:  np.int8
    PERSISTENT_TYPE: np.int8


def test_schema_builds_params_in_field_order():
    sch = ati.tune.schema(AttnFwdPerf)
    assert isinstance(sch, PerfSchema)
    assert sch.names == ['BLOCK_M', 'BLOCK_N', 'PRE_LOAD_V', 'NUM_XCDS',
                         'GRID_CU_MULTIP', 'PERSISTENT_TYPE']
    assert all(isinstance(p, PerfParam) for p in sch.params)


def test_field_ctypes_match_perf_choices_packing():
    sch = ati.tune.schema(AttnFwdPerf)
    itype = {p.name: p.itype for p in sch.params}
    assert itype['BLOCK_M'] == 'int16_t'
    assert itype['BLOCK_N'] == 'int16_t'
    assert itype['NUM_XCDS'] == 'int8_t'
    assert itype['GRID_CU_MULTIP'] == 'int8_t'
    assert itype['PERSISTENT_TYPE'] == 'int8_t'
    assert itype['PRE_LOAD_V'] == 'bool'


def test_choice_for_keeps_declared_width():
    sch = ati.tune.schema(AttnFwdPerf)
    block_m = next(p for p in sch.params if p.name == 'BLOCK_M')
    # 256 fits in int16; the declared width is kept (not shrunk to int8) so the
    # struct layout is stable regardless of which config values appear.
    c = block_m.choice_for(256)
    assert c.itype == 'int16_t'
    assert c.triton_compile_signature == 256
    num_xcds = next(p for p in sch.params if p.name == 'NUM_XCDS')
    assert num_xcds.choice_for(8).itype == 'int8_t'


def test_bool_field():
    sch = ati.tune.schema(AttnFwdPerf)
    pv = next(p for p in sch.params if p.name == 'PRE_LOAD_V')
    c = pv.choice_for(True)
    assert c.itype == 'bool'
    assert c.triton_compile_signature is True


def test_unsupported_dtype_errors():
    @dataclass
    class Bad:
        x: float          # not an integer/bool perf dtype
    try:
        ati.tune.schema(Bad)
    except TypeError as e:
        assert "'x'" in str(e)
        return
    raise AssertionError('expected unsupported-dtype TypeError')


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} perf-schema tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
