# SPDX-License-Identifier: Apache-2.0

"""Per-call knob rules (`fmha_variants`) and the gfx950 forward's use of them.

No GPU and no FlyDSL: the rules, the tuning policy and the AOT/JIT entries are
all pure Python.

The policy tests check, for every gfx950 forward functional, that what AOT
would compile and select (`builds`: one override set per box) agrees with what
JIT applies (`overrides_at`) at every cell of the common refinement -- one
point per cell is exhaustive, since every rule bound is a cut.
"""

import itertools

import pytest

import tuning_fwd
from fmha_meta import CAUSAL_BOTTOM_RIGHT, CAUSAL_NONE, CAUSAL_WINDOW, FmhaInputMetadata
from fmha_variants import RUNTIME_VALUES, builds, check, contains, derive, eq, ge, gt, le, lt, overrides_at

INT32_MIN, INT32_MAX = RUNTIME_VALUES['seqlen_k']


# ---------------------------------------------------------------------------
# The rule module
# ---------------------------------------------------------------------------

def test_predicates_are_inclusive_intervals():
    assert derive('x', 1, when=le('seqlen_k', 128)).box == {'seqlen_k': (INT32_MIN, 128)}
    assert derive('x', 1, when=lt('seqlen_k', 128)).box == {'seqlen_k': (INT32_MIN, 127)}
    assert derive('x', 1, when=gt('seqlen_k', 128)).box == {'seqlen_k': (129, INT32_MAX)}
    assert derive('x', 1, when=ge('seqlen_k', 128)).box == {'seqlen_k': (128, INT32_MAX)}
    assert derive('x', 1, when=eq('seqlen_k', 128)).box == {'seqlen_k': (128, 128)}
    assert derive('x', 1, when=()).box == {}


def test_a_tuple_of_predicates_is_their_conjunction():
    rule = derive('x', 1, when=(gt('seqlen_k', 64), le('seqlen_k', 256), ge('seqlen_q', 8)))
    assert rule.box == {'seqlen_k': (65, 256), 'seqlen_q': (8, INT32_MAX)}
    with pytest.raises(ValueError, match='holds for no call'):
        derive('x', 1, when=(gt('seqlen_k', 64), le('seqlen_k', 64)))


def test_unknown_runtime_value_is_rejected():
    with pytest.raises(ValueError, match='not a runtime value'):
        le('seqlen', 3)


def test_coupled_targets_take_one_value_each():
    rule = derive(('num_waves', 'block_m'), (8, 128), when=())
    assert rule.overrides == {'num_waves': 8, 'block_m': 128}
    with pytest.raises(ValueError, match='2 targets but 1 values'):
        derive(('num_waves', 'block_m'), (8,), when=())


def test_overlapping_rules_on_one_target_are_rejected():
    with pytest.raises(ValueError, match='overlapping'):
        check([derive('x', 1, when=gt('seqlen_k', 128)), derive('x', 2, when=gt('seqlen_k', 192))])
    # Different targets may overlap; touching boxes are disjoint.
    check([derive('x', 1, when=gt('seqlen_k', 128)), derive('y', 2, when=gt('seqlen_k', 192))])
    check([derive('x', 1, when=le('seqlen_k', 128)), derive('x', 2, when=gt('seqlen_k', 128))])


def test_a_target_no_rule_matches_keeps_its_default():
    rules = [derive('x', 1, when=gt('seqlen_k', 128))]
    assert overrides_at(rules, {'seqlen_k': 128}) == {}
    assert overrides_at(rules, {'seqlen_k': 129}) == {'x': 1}
    assert builds(rules) == [([{'seqlen_k': (INT32_MIN, 128)}], {}),
                             ([{'seqlen_k': (129, INT32_MAX)}], {'x': 1})]


def test_a_missing_runtime_value_a_rule_reads_raises():
    with pytest.raises(ValueError, match="'seqlen_k'"):
        overrides_at([derive('x', 1, when=gt('seqlen_k', 128))], {'seqlen_q': 1})


def test_builds_refine_rules_on_different_targets():
    rules = [derive('x', 1, when=gt('seqlen_k', 128)),
             derive('y', 1, when=gt('seqlen_k', 192)),
             derive('z', 1, when=le('seqlen_q', 16))]
    found = builds(rules)
    assert [over for _, over in found] == [{'z': 1}, {}, {'x': 1, 'z': 1}, {'x': 1},
                                           {'x': 1, 'y': 1, 'z': 1}, {'x': 1, 'y': 1}]
    _assert_builds_agree_with_jit(rules)


def test_equal_overrides_merge_into_one_build():
    # x is pinned on both sides of seqlen_q 16, so seqlen_q splits nothing.
    rules = [derive('x', 1, when=le('seqlen_q', 16)), derive('x', 1, when=gt('seqlen_q', 16))]
    assert builds(rules) == [([{}], {'x': 1})]


def _cells(rules):
    """One point per cell of the common refinement of `rules`."""
    names = sorted({n for r in rules for n in r.box})
    axes = []
    for n in names:
        lo, hi = RUNTIME_VALUES[n]
        cuts = {lo, hi + 1} | {b for r in rules if n in r.box for b in (r.box[n][0], r.box[n][1] + 1)}
        axes.append(sorted(cuts)[:-1])
    for point in itertools.product(*axes):
        yield dict(zip(names, point))


def _assert_builds_agree_with_jit(rules):
    found = builds(rules)
    for values in _cells(rules):
        serving = [over for boxes, over in found if any(contains(b, values) for b in boxes)]
        assert len(serving) == 1, f'{values} is served by {len(serving)} builds'
        assert serving[0] == overrides_at(rules, values), values


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
    rules = tuning_fwd.variants('gfx950', meta)
    _assert_builds_agree_with_jit(rules)
    result = tuning_fwd.aot_builds('gfx950', meta)
    found = builds(rules)
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
    assert _jit(100, CAUSAL_NONE, seqlen_k=128).exact_kv_tiles is True
    assert _jit(100, CAUSAL_NONE, seqlen_k=129).exact_kv_tiles is False
    assert _jit(100, CAUSAL_NONE, seqlen_k=129).hdim_qk_floor == 96
    # Windows and other tiles have no seqlen rule, so need no seqlen.
    assert _jit(100, CAUSAL_WINDOW).exact_kv_tiles is True
    assert _jit(128, CAUSAL_NONE).hdim_qk_floor == 0


def test_gfx950_fwd_jit_needs_the_values_its_rules_read():
    with pytest.raises(ValueError, match="'seqlen_k'"):
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
