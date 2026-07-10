# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Aux-kernel xref Step 5: perf is citeable, and schema-only kernels are untunable
(agent-plans/ati_aux-kernel-xref_rev0.md §4.3).

  * schema (@ati.tune.schema) WITHOUT configs  -> untunable: only the schema's
    default perf is compiled (translate_empty_dataframe), no autotune/LUT/DB.
  * a citing kernel that declares NO perf inherits the cited kernel's whole tune.
"""

import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import aotriton.template_instantiation as ati
from aotriton.template_instantiation.describe import describe
from registry import InterfaceRegistry, _testonly_build_kernel_description
# Cited kernel: fake attn_fwd. Citing kernel: fake debug.
from fakekernels import attn_fwd_stub, debug_stub

MAIN_DTYPES = ['*fp16:16', '*bf16:16', '*fp32:16']


@dataclass
class DebugPerf:
    BLOCK_M: np.int16 = 64
    BLOCK_N: np.int16 = 32


def _register_attn_fwd(registry):
    return _testonly_build_kernel_description(attn_fwd_stub(), family='flash',
                                    triton_kernel_name='attn_fwd',
                                    registry=registry)


def _describe_schema_only_debug(registry):
    # R declared locally; gaps + T_io from the cite; OWN schema-only perf (no
    # configs -> untunable).
    specs = [
        ati.cite('op_attn_fwd.triton.attn_fwd'),
        ati.tensor('R', 'T_io', strides='stride_r?', contiguous=-1,
                   wires_to='encoded_softmax'),
        ati.tune.schema(DebugPerf),
    ]
    debug = debug_stub()
    describe(debug, *specs, _validate=False)
    return _testonly_build_kernel_description(debug, family='flash',
                                    register=False, registry=registry)


def _describe_perfless_debug(registry):
    # No perf decorators at all -> inherit the cited attn_fwd's whole tune.
    specs = [
        ati.cite('op_attn_fwd.triton.attn_fwd'),
        ati.tensor('R', 'T_io', strides='stride_r?', contiguous=-1,
                   wires_to='encoded_softmax'),
        ati.scalar(['BLOCK_M', 'BLOCK_N'], options=[64]),  # claim the constexprs
    ]
    debug = debug_stub()
    describe(debug, *specs, _validate=False)
    return _testonly_build_kernel_description(debug, family='flash',
                                              register=False,
                                              registry=registry)


def test_schema_only_is_untunable():
    reg = InterfaceRegistry()
    _register_attn_fwd(reg)
    kdesc = _describe_schema_only_debug(reg)
    assert kdesc.is_tunable is False          # schema present, configs absent


def test_schema_only_compiles_one_default_signature():
    reg = InterfaceRegistry()
    _register_attn_fwd(reg)
    kdesc = _describe_schema_only_debug(reg)
    f = next(kdesc.gen_functionals({'gfx942': ['gfx942']}))
    lut, sigs, binning = kdesc.translate_empty_dataframe(f)
    assert len(sigs) == 1                      # exactly the default perf
    assert binning is None


def test_perfless_kernel_inherits_cited_tune():
    reg = InterfaceRegistry()
    af = _register_attn_fwd(reg)
    kdesc = _describe_perfless_debug(reg)
    # No local perf -> the cited attn_fwd's tune was adopted (it is tunable).
    assert kdesc.tune is af.tune
    assert kdesc.is_tunable is True


def test_schema_only_keeps_own_perf_not_cited():
    reg = InterfaceRegistry()
    af = _register_attn_fwd(reg)
    kdesc = _describe_schema_only_debug(reg)
    assert kdesc.tune is not af.tune          # own schema, not the cited tune
    names = list(kdesc.tune.schema.param_names())
    assert names == ['BLOCK_M', 'BLOCK_N']


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} citeable-perf tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
