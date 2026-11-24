# SPDX-License-Identifier: Apache-2.0

"""The flyc forward's per-call rules (`variants`, read by `tuning_fwd`).

No GPU and no FlyDSL: the rules, the tuning policy and the AOT/JIT entries are
all pure Python. The rule semantics themselves are ATI's, tested in
`python/test/test_runtime_rules.py`.

For every gfx950 forward functional, `verify` checks that what AOT would compile
and select agrees with what JIT applies at every cell of the common refinement.
"""

import itertools

import pytest

import ati_bootstrap  # noqa: F401  (aotriton from this checkout)
import tuning_fwd
from aotriton.template_instantiation.ir.runtime_rules import builds, verify
from fmha_meta import CAUSAL_BOTTOM_RIGHT, CAUSAL_NONE, CAUSAL_WINDOW, RUNTIME_VALUES, FmhaInputMetadata


# ---------------------------------------------------------------------------
# The gfx950 forward policy
# ---------------------------------------------------------------------------

def _functionals():
    axes = itertools.product((32, 64, 96, 128, 160, 192, 224, 256, 384, 512),
                             (False, True),
                             (CAUSAL_NONE, CAUSAL_BOTTOM_RIGHT, CAUSAL_WINDOW),
                             (False, True),
                             (False, True))
    for block_dmodel, padded_head, causal_type, bias, dropout in axes:
        if bias and causal_type != CAUSAL_NONE:
            continue
        yield FmhaInputMetadata(num_heads=1,
                                dtype_str='bf16',
                                causal_type=causal_type,
                                block_dmodel=block_dmodel,
                                padded_head=padded_head,
                                bias=bias,
                                dropout=dropout)


@pytest.mark.parametrize('meta', list(_functionals()), ids=lambda m: (
    f'd{m.block_dmodel}{"p" if m.padded_head else ""}-c{m.causal_type}'
    f'{"-bias" if m.bias else ""}{"-drop" if m.dropout else ""}'))
def test_gfx950_fwd_aot_builds_agree_with_jit(meta):
    found = verify(tuning_fwd.variants('gfx950', meta), RUNTIME_VALUES)
    result = tuning_fwd.aot_builds('gfx950', meta)
    if isinstance(result, tuple):
        assert len(found) == 1 and found[0][0] == [{}]
        return
    result = list(result)
    assert [boxes for boxes, _, _ in result] == [boxes for boxes, _ in found]
    # Every build resolves, and no two share knobs (one image each).
    knobs = [tuple(sorted(k.items())) for _, _, k in result]
    assert len(set(knobs)) == len(knobs)


def test_gfx950_fwd_first_build_is_todays():
    """The code generator ships the first build until it emits the selector."""
    meta = FmhaInputMetadata(num_heads=1,
                             dtype_str='bf16',
                             causal_type=CAUSAL_NONE,
                             block_dmodel=128,
                             padded_head=True)
    (boxes, _, knobs), *rest = tuning_fwd.aot_builds('gfx950', meta)
    assert knobs['hdim_qk_floor'] == 0 and knobs['exact_kv_tiles'] is True
    assert len(rest) == 3


def _jit(head_dim, causal_type, **values):
    meta = FmhaInputMetadata(num_heads=8,
                             dtype_str='bf16',
                             causal_type=causal_type,
                             block_dmodel=None,
                             head_dim=head_dim)
    return tuning_fwd.plan('gfx950', meta, **values)[1]


def test_gfx950_fwd_jit_padded_head_walk():
    assert _jit(100, CAUSAL_NONE, max_seqlen_k=128).exact_kv_tiles is True
    assert _jit(100, CAUSAL_NONE, max_seqlen_k=129).exact_kv_tiles is False
    assert _jit(100, CAUSAL_NONE, max_seqlen_k=129).hdim_qk_floor == 96
    # Windows and other tiles have no seqlen rule, so need no seqlen.
    assert _jit(100, CAUSAL_WINDOW).exact_kv_tiles is True
    assert _jit(128, CAUSAL_NONE).hdim_qk_floor == 0


def test_gfx950_fwd_jit_needs_the_values_its_rules_read():
    with pytest.raises(ValueError, match="'max_seqlen_k'"):
        _jit(100, CAUSAL_NONE)
    # Pinning the knob a rule decides makes the rule moot.
    meta = FmhaInputMetadata(num_heads=8,
                             dtype_str='bf16',
                             causal_type=CAUSAL_NONE,
                             block_dmodel=None,
                             head_dim=100)
    _, knobs = tuning_fwd.plan('gfx950', meta, overrides={'exact_kv_tiles': False})
    assert knobs.exact_kv_tiles is False


def test_gfx1201_fwd_has_one_build():
    meta = FmhaInputMetadata(num_heads=1,
                             dtype_str='bf16',
                             causal_type=CAUSAL_NONE,
                             block_dmodel=128,
                             padded_head=True)
    assert isinstance(tuning_fwd.aot_builds('gfx1201', meta), tuple)
