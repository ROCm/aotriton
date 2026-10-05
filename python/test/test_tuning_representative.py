# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tuning candidates are generated from the tuning representative.

@ati.tune.fallback(PADDED_HEAD=False) folds both PADDED_HEAD functionals onto
one database row and one tuning entry. gen_autotune_configs() therefore sees
`Interface.tuning_representative(f)` -- `f` with every fallback axis pinned --
so the functionals sharing an entry get the same candidates, in the same order,
whatever the description does with a fallback axis.

Before, gen_autotune_configs() saw `f` itself. #220 widened gfx1100 attn_fwd
for PADDED_HEAD=False only; the padded twin kept 64 of the 496 candidates and
the tuner, which runs one impl_index on both twins, could only pick among the
first 64.

These tests link the real modules/ tree (no GPU needed).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_MODULES = Path(__file__).resolve().parents[2] / 'modules'


def _tunable_kernels():
    from aotriton.codegen.linker import Linker
    kernels = Linker(_MODULES).link_all_families()[0]
    return [k for k in kernels if k.is_tunable]


def _all_archs():
    from aotriton.gpu_targets import AOTRITON_ARCH_WARPSIZE
    return sorted(AOTRITON_ARCH_WARPSIZE)


def _functionals(kdesc, arch):
    return [f for f in kdesc.gen_functionals({arch: [f'{arch}_mod0']})
            if not kdesc.is_functional_disabled(f)]


def _candidates(kdesc, f):
    return [(s.psel_section, s.copt_section) for s in kdesc.gen_signatures_for_tuning(f)]


def _assert_shared_entries_match(kdesc, arch):
    by_entry = {}
    for f in _functionals(kdesc, arch):
        rep = kdesc.tuning_representative(f)
        cands = _candidates(kdesc, f)
        if rep.godel_number in by_entry:
            first, first_cands = by_entry[rep.godel_number]
            assert cands == first_cands, (
                f'{kdesc.NAME}: {f.tunecc_signature} and {first.tunecc_signature} share '
                f'a tuning entry but have {len(cands)} vs {len(first_cands)} candidates')
        else:
            by_entry[rep.godel_number] = (f, cands)


def test_representative_pins_fallback_axes():
    attn_fwd = next(k for k in _tunable_kernels() if k.NAME == 'attn_fwd')
    assert attn_fwd.partially_tuned_functionals == {'PADDED_HEAD': False}
    fs = _functionals(attn_fwd, 'gfx1100')
    padded = [f for f in fs if 'PADDED_HEAD=True' in f.tunecc_signature]
    assert padded
    for f in padded:
        rep = attn_fwd.tuning_representative(f)
        assert rep.tunecc_signature == f.tunecc_signature.replace('PADDED_HEAD=True', 'PADDED_HEAD=False')
        assert attn_fwd.tuning_representative(rep) is rep
        assert any(rep.godel_number == g.godel_number for g in fs)


def test_shared_entries_have_identical_candidates():
    kernels = _tunable_kernels()
    assert kernels, 'no tunable kernel linked; the check below would pass vacuously'
    for kdesc in kernels:
        for arch in _all_archs():
            _assert_shared_entries_match(kdesc, arch)


def test_padded_head_dependent_search_space_cannot_split_an_entry():
    """Reintroduce #220's `PADDED_HEAD is False` condition: twins still match."""
    attn_fwd = next(k for k in _tunable_kernels() if k.NAME == 'attn_fwd')
    module_globals = attn_fwd._built.tune.configs.__globals__
    original = module_globals['_use_extended_search']

    def padded_head_dependent(f, *args):
        return original(f, *args) and f.choices.PADDED_HEAD is False

    module_globals['_use_extended_search'] = padded_head_dependent
    try:
        _assert_shared_entries_match(attn_fwd, 'gfx1100')
        widened = [f for f in _functionals(attn_fwd, 'gfx1100')
                   if 'PADDED_HEAD=True' in f.tunecc_signature
                   and "Q='*bf16:16';BLOCK_DMODEL=128;" in f.tunecc_signature
                   and 'ENABLE_DROPOUT=False;CAUSAL_TYPE=0;BIAS_TYPE=0' in f.tunecc_signature]
        assert len(widened) == 1
        assert len(_candidates(attn_fwd, widened[0])) == 496
    finally:
        module_globals['_use_extended_search'] = original
