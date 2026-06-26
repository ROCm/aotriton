# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit test for tune.configs/binning/fallback registration."""

import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

import aotriton.template_instantiation as ati
from aotriton.template_instantiation.describe import describe, get_kernel_spec
from aotriton.template_instantiation.ir import TypedChoice, Axis, Interface


class _IRStub(Interface):
    FAMILY = 'test'
    NAME = 'stub'
    def __init__(self, axes, overrides):
        self._axes = axes
        self._overrides = overrides
    def _axes_overrides(self):
        return self._axes, self._overrides
    # Interface abstract contract (no functional struct in this bare IR stub).
    @property
    def func_cfields(self):
        return []
    def list_functional_params(self):
        return []


def enumerate_functionals(axes, overrides, target_arch):
    return _IRStub(axes, overrides).gen_functionals(target_arch)


@dataclass
class TinyPerf:
    BLOCK_M: np.int16 = 16
    PRE_LOAD_V: bool = False


def gen_configs(f):
    # reads pinned choices through f.choices (keyed by variable name);
    # Config(kw, ...) mirrors triton.Config / aotriton.autotune.Config.
    causal = f.choices.CAUSAL_TYPE
    for bm in (16, 32):
        kw = {'BLOCK_M': bm, 'PRE_LOAD_V': (causal != 0), 'waves_per_eu': 2}
        yield ati.tune.Config(kw, num_warps=4, num_stages=1)


def _kernel():
    def k(Q, stride_qz, stride_qh, stride_qm, stride_qk,
          CAUSAL_TYPE: 'constexpr', Max_seqlen_q: 'i32', Max_seqlen_k: 'i32'):
        pass
    return k


def _describe(k):
    T = ati.type_var('T', dtype=['*fp16:16'])
    describe(k,
             ati.tensor('Q', T, strides='stride_q?'),
             ati.scalar('CAUSAL_TYPE', options=[0, 3]),
             ati.scalar('Max_seqlen_q', 'i32'),
             ati.scalar('Max_seqlen_k', 'i32'),
             ati.tune.schema(TinyPerf),
             ati.tune.configs(gen_configs),
             ati.tune.binning(Max_seqlen_q=ati.tune.binning.le,
                              Max_seqlen_k=ati.tune.binning.le),
             ati.tune.fallback(PADDED_HEAD=False),
             _validate=False)
    return get_kernel_spec(k)


def test_tune_spec_assembled():
    spec = _describe(_kernel())
    ts = spec.tune
    assert ts is not None
    assert ts.is_tunable
    assert ts.schema is not None and ts.schema.param_names() == ['BLOCK_M', 'PRE_LOAD_V']
    assert set(ts.binning) == {'Max_seqlen_q', 'Max_seqlen_k'}
    assert ts.fallback == {'PADDED_HEAD': False}


def test_binning_selectors_recorded():
    spec = _describe(_kernel())
    sel = spec.tune.binning['Max_seqlen_q']
    assert sel.key == 'le'


def test_configs_generator_runs_over_functional():
    spec = _describe(_kernel())
    # Build the IR axis set just for CAUSAL_TYPE to get a Functional to feed.
    axes = [Axis('CAUSAL_TYPE', ('CAUSAL_TYPE',),
                 [TypedChoice.parse(0), TypedChoice.parse(3)], anchor=0)]
    f = next(enumerate_functionals(axes, [], {'gfx942': ['g0']}))
    cfgs = list(spec.tune.configs(f))
    assert len(cfgs) == 2
    assert [c.perf['BLOCK_M'] for c in cfgs] == [16, 32]
    # CAUSAL_TYPE==0 for godel 0 -> PRE_LOAD_V False
    assert all(c.perf['PRE_LOAD_V'] is False for c in cfgs)
    # waves_per_eu is carried in kw but classified as a compiler option
    assert all('waves_per_eu' not in c.perf for c in cfgs)
    assert cfgs[0].copts == {'waves_per_eu': 2, 'num_warps': 4, 'num_stages': 1}


def test_config_api_matches_old_autotune_config():
    # Same call shape as v3python.rules.flash.attn_fwd: Config(kw, num_stages=, num_warps=)
    kw = {'BLOCK_M': 128, 'BLOCK_N': 64, 'PRE_LOAD_V': True, 'waves_per_eu': 3}
    c = ati.tune.Config(kw, num_warps=8, num_stages=2)
    assert c.kwargs == kw
    assert c.num_warps == 8 and c.num_stages == 2
    assert c.waves_per_eu == 3


def test_config_separates_perf_and_copts():
    kw = {'BLOCK_M': 128, 'BLOCK_N': 64, 'PRE_LOAD_V': True, 'waves_per_eu': 3}
    c = ati.tune.Config(kw, num_warps=8, num_stages=2)
    # perf = constexpr kernel args (kw minus the compiler option waves_per_eu)
    assert c.perf == {'BLOCK_M': 128, 'BLOCK_N': 64, 'PRE_LOAD_V': True}
    assert c.copts == {'waves_per_eu': 3, 'num_warps': 8, 'num_stages': 2}


def test_waves_per_eu_is_compiler_option_not_kernel_arg():
    # triton.Config puts waves_per_eu in kwargs (as if a kernel arg); AOTriton
    # classifies it as a compiler option, so it must never appear in .perf.
    c = ati.tune.Config({'BLOCK_M': 64, 'waves_per_eu': 4})
    assert 'waves_per_eu' not in c.perf
    assert c.copts['waves_per_eu'] == 4


def test_binning_rejects_non_selector():
    try:
        ati.tune.binning(Max_seqlen_q='le')      # raw string, not a selector
    except AssertionError:
        return
    raise AssertionError('expected selector-type assertion')


def test_no_tune_decorators_leaves_none():
    def k(Q, stride_qz, stride_qh, stride_qm, stride_qk):
        pass
    T = ati.type_var('T', dtype=['*fp16:16'])
    describe(k, ati.tensor('Q', T, strides='stride_q?'), _validate=False)
    assert get_kernel_spec(k).tune is None


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} tune-registration tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
