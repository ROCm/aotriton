# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Functionals sharing a tuning entry must have identical tuning candidates.

@ati.tune.fallback(PADDED_HEAD=False) folds both PADDED_HEAD functionals onto
one database row, and one tuning entry runs the same impl_index on both (its
irregular-hdim test cases land on PADDED_HEAD=True). If the two candidate
lists differ, impl_index silently means different kernels -- or no kernel --
depending on the test case.

That happened on gfx1100: #220 widened attn_fwd's search space for
PADDED_HEAD=False only, so every candidate past the padded twin's 64th was
rejected as broken, and the shipped table could only pick from the first 64.

The last two tests link the real modules/ tree (no GPU needed): the guard must
accept every shipped description and must reject the #220 condition.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aotriton.codegen.autotune import (
    SharedTuningEntryMismatch,
    check_shared_tuning_entries,
    tuning_entry_key,
)

_MODULES = Path(__file__).resolve().parents[2] / 'modules'


def _fake_functional(arch='gfx1100', padded=False, dtype='"*fp16:16"', fallback=None):
    tc = lambda text: SimpleNamespace(infotext=text)
    meta = SimpleNamespace(partially_tuned_functionals={'PADDED_HEAD': False} if fallback is None else fallback)
    return SimpleNamespace(
        arch=arch,
        meta_object=meta,
        compact_choices={'Q': tc(dtype), 'PADDED_HEAD': tc('true' if padded else 'false')},
        tunecc_signature=f'{arch}-{dtype}-padded{int(padded)}')


_A = ('BLOCK_M=64;BLOCK_N=32', 'num_stages=1')
_B = ('BLOCK_M=32;BLOCK_N=16', 'num_stages=1')
_C = ('BLOCK_M=16;BLOCK_N=16', 'num_stages=1')


def test_fallback_keys_collapse_onto_one_entry():
    assert tuning_entry_key(_fake_functional(padded=False)) == tuning_entry_key(_fake_functional(padded=True))


def test_non_fallback_keys_stay_distinct():
    no_fallback = {}
    assert (tuning_entry_key(_fake_functional(padded=False, fallback=no_fallback))
            != tuning_entry_key(_fake_functional(padded=True, fallback=no_fallback)))
    assert tuning_entry_key(_fake_functional(dtype='"*bf16:16"')) != tuning_entry_key(_fake_functional())
    assert tuning_entry_key(_fake_functional(arch='gfx942')) != tuning_entry_key(_fake_functional())


def test_identical_twins_pass():
    check_shared_tuning_entries('k', [(_fake_functional(padded=False), [_A, _B, _C]),
                                      (_fake_functional(padded=True), [_A, _B, _C])])


def test_shorter_twin_is_rejected():
    with pytest.raises(SharedTuningEntryMismatch, match='first divergence at impl_index 2'):
        check_shared_tuning_entries('k', [(_fake_functional(padded=False), [_A, _B, _C]),
                                          (_fake_functional(padded=True), [_A, _B])])


def test_reordered_twin_is_rejected():
    with pytest.raises(SharedTuningEntryMismatch, match='first divergence at impl_index 0'):
        check_shared_tuning_entries('k', [(_fake_functional(padded=False), [_A, _B]),
                                          (_fake_functional(padded=True), [_B, _A])])


def test_different_entries_may_differ():
    check_shared_tuning_entries('k', [(_fake_functional(dtype='"*fp16:16"'), [_A, _B, _C]),
                                      (_fake_functional(dtype='"*bf16:16"'), [_A])])


def _real_tunable_kernels():
    from aotriton.codegen.linker import Linker
    kernels = Linker(_MODULES).link_all_families()[0]
    return [k for k in kernels if k.is_tunable]


def _real_candidates(kdesc, arch):
    for f in kdesc.gen_functionals({arch: [f'{arch}_mod0']}):
        if kdesc.is_functional_disabled(f):
            continue
        yield f, [(s.psel_section, s.copt_section) for s in kdesc.gen_signatures_for_tuning(f)]


def _all_archs():
    from aotriton.gpu_targets import AOTRITON_ARCH_WARPSIZE
    return sorted(AOTRITON_ARCH_WARPSIZE)


def test_real_descriptions_share_candidates():
    kernels = _real_tunable_kernels()
    assert kernels, 'no tunable kernel linked; the check below would pass vacuously'
    for kdesc in kernels:
        for arch in _all_archs():
            check_shared_tuning_entries(kdesc.NAME, _real_candidates(kdesc, arch))


def test_guard_rejects_padded_head_dependent_search_space():
    """Reintroduce #220's `PADDED_HEAD is False` condition: the guard must fire."""
    attn_fwd = next(k for k in _real_tunable_kernels() if k.NAME == 'attn_fwd')
    configs = attn_fwd._built.tune.configs
    module_globals = configs.__globals__
    original = module_globals['_use_extended_search']

    def padded_head_dependent(f, *args):
        return original(f, *args) and f.choices.PADDED_HEAD is False

    module_globals['_use_extended_search'] = padded_head_dependent
    try:
        with pytest.raises(SharedTuningEntryMismatch, match='attn_fwd'):
            check_shared_tuning_entries('attn_fwd', _real_candidates(attn_fwd, 'gfx1100'))
    finally:
        module_globals['_use_extended_search'] = original
