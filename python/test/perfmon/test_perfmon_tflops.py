# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Golden values and the `flash.js` guard for modules/flash/perfmon/tflops.py,
the Python port of the visperf formulas. No GPU, torch or database."""

import importlib
import re
from pathlib import Path

import pytest

_FLASH_JS = Path(__file__).resolve().parents[3] / 'modules' / 'flash' / 'visperf' / 'static' / 'flash.js'


def _tflops():
    from aotriton.perfmon.registry import load_family_perfmon
    mod = load_family_perfmon('flash')
    return importlib.import_module('.tflops', package=mod.__name__)


# -- golden values, computed by hand -------------------------------------------------

def test_fwd_non_causal():
    # valid = 128*128; per matmul 2*valid*64; total 2*that*3*5; median 1 ms
    assert _tflops().attn_fwd_tflops(128, 128, 64, False, 1.0, 3, 5) == pytest.approx(0.06291456)


def test_fwd_causal_square():
    # valid = 16384 - (16384 - 128) / 2 = 8256
    assert _tflops().attn_fwd_tflops(128, 128, 64, True, 1.0, 3, 5) == pytest.approx(0.03170304)


def test_fwd_causal_more_queries_than_keys():
    # sq > sk: valid = (sk^2 + sk) / 2 = 8256
    assert _tflops().attn_fwd_tflops(256, 128, 128, True, 2.0, 2, 4) == pytest.approx(0.016908288)


def test_fwd_causal_fewer_queries_than_keys():
    # sq < sk: valid = sq*sk - (sq^2 - sq) / 2 = 128*256 - 8128 = 24640
    flops = 2 * (2 * 24640 * 64)
    assert _tflops().attn_fwd_tflops(128, 256, 64, True, 1.0, 1, 1) == pytest.approx(flops / 1e-3 / 1e12)


def test_bwd_is_the_forward_with_the_time_divided_by_2p5():
    t = _tflops()
    fwd = t.attn_fwd_tflops(128, 128, 64, False, 1.0, 3, 5)
    assert t.attn_bwd_tflops(128, 128, 64, False, 1.0, 3, 5) == pytest.approx(fwd * 2.5)
    assert t.attn_bwd_tflops(128, 128, 64, False, 1.0, 3, 5) == pytest.approx(0.1572864)


def test_valid_elements():
    t = _tflops()
    assert t.attn_valid_elements(10, 10, False) == 100
    assert t.attn_valid_elements(10, 10, True) == 55          # the lower triangle with its diagonal
    assert t.attn_valid_elements(10, 4, True) == 10           # (16 + 4) / 2


def test_description_converts_microseconds_and_picks_the_iface():
    from aotriton.perfmon.registry import make_perf_desc
    desc = make_perf_desc('flash')
    entry = [e for e in desc.entries({**desc.entry_sets()['prime'], 'hdim': [64], 'causal': [0],
                                      'seqlen': [128], 'T_io': ['bf16'], 'dropout_p': [0.0]})][0]
    t = _tflops()
    expected = t.attn_fwd_tflops(128, 128, 64, False, 1.0, entry.batch, entry.n_heads_q)
    assert desc.tflops('op_attn_fwd', entry, 1000.0) == pytest.approx(expected)
    assert desc.tflops('op_attn_bwd', entry, 1000.0) == pytest.approx(expected * 2.5)


# -- the guard: the port must follow flash.js ------------------------------------------------

def _js_body(source: str, name: str) -> str:
    m = re.search(r'function\s+' + re.escape(name) + r'\s*\([^)]*\)\s*\{', source)
    assert m, f'{name} not found in flash.js'
    depth, i = 1, m.end()
    while depth:
        depth += {'{': 1, '}': -1}.get(source[i], 0)
        i += 1
    return source[m.end():i - 1]


_EXPECTED = {
    '_attnValidElements': '''
  if (causal) {
    return seqlen_q <= seqlen_k
      ? seqlen_q * seqlen_k - (seqlen_q * seqlen_q - seqlen_q) / 2
      : (seqlen_k * seqlen_k + seqlen_k) / 2;
  }
  return seqlen_q * seqlen_k;
''',
    'attnFwdTflops': '''
  const valid = _attnValidElements(seqlen_q, seqlen_k, causal);
  const flops_per_matmul = 2 * valid * hdim;
  const total_flops = 2 * flops_per_matmul * batch * n_heads;
  return total_flops / (median_ms * 1e-3) / 1e12;
''',
    'attnBwdTflops': '''
  // backward = 2.5x forward FLOPs (2.0 bwd matmuls + 0.5 recompute)
  return attnFwdTflops(seqlen_q, seqlen_k, hdim, causal, median_ms / 2.5, batch, n_heads);
''',
}


@pytest.mark.parametrize('name', sorted(_EXPECTED))
def test_flash_js_formula_text_unchanged(name):
    body = _js_body(_FLASH_JS.read_text(), name)
    assert body == _EXPECTED[name], (
        f'flash.js {name} changed: re-port modules/flash/perfmon/tflops.py and update this text')
