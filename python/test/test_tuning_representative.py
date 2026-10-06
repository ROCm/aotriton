# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tuning candidates are generated from the tuning representative.

@ati.tune.fallback(PADDED_HEAD=False) folds both PADDED_HEAD functionals onto
one database row and one tuning entry, and the tuner runs the same impl_index
on both. KernelDescription.gen_autotune_configs(f) therefore returns the
configs of the representative (`f` with every fallback axis pinned), so twins
get the same candidates by construction, and raises FallbackAxisRead when the
generator gives `f` different configs (as #220 did on gfx1100 attn_fwd).

These tests link the real modules/ tree (no GPU needed).
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aotriton.template_instantiation.ir.interface import FallbackAxisRead

_MODULES = Path(__file__).resolve().parents[2] / 'modules'


def _tunable_kernels():
    from aotriton.codegen.linker import Linker
    return [k for k in Linker(_MODULES).link_all_families()[0] if k.is_tunable]


def _attn_fwd():
    return next(k for k in _tunable_kernels() if k.NAME == 'attn_fwd')


def _functionals(kdesc, arch):
    return [f for f in kdesc.gen_functionals({arch: [f'{arch}_mod0']})
            if not kdesc.is_functional_disabled(f)]


def _assert_shared_entries_match(kdesc, arch, build_dir):
    """Compare the candidate lists the code generator emits for each functional."""
    from aotriton.codegen.autotune import AutotuneCodeGenerator
    args = SimpleNamespace(build_for_tuning=True, build_dir=build_dir)
    first = {}
    for f in _functionals(kdesc, arch):
        sigs = AutotuneCodeGenerator(args, f, None, (), None).all_signatures
        cands = [(s.psel_section, s.copt_section) for s in sigs]
        g, c0 = first.setdefault(kdesc.tuning_representative(f).godel_number, (f, cands))
        assert cands == c0, f'{kdesc.NAME}: {f.tunecc_signature} and {g.tunecc_signature} differ'


def test_representative_pins_fallback_axes():
    attn_fwd = _attn_fwd()
    padded = [f for f in _functionals(attn_fwd, 'gfx1100') if 'PADDED_HEAD=True' in f.tunecc_signature]
    assert padded
    for f in padded:
        rep = attn_fwd.tuning_representative(f)
        assert rep.tunecc_signature == f.tunecc_signature.replace('PADDED_HEAD=True', 'PADDED_HEAD=False')
        assert attn_fwd.tuning_representative(rep) is rep


def test_representative_does_not_need_prior_enumeration():
    """godel strides are assigned by tuning_representative itself, not left over
    from an earlier gen_functionals() on the same Interface."""
    attn_fwd = _attn_fwd()
    fs = _functionals(attn_fwd, 'gfx1100')
    padded = next(f for f in fs if 'PADDED_HEAD=True' in f.tunecc_signature)
    unpadded = next(f for f in fs if f.tunecc_signature == padded.tunecc_signature.replace(
        'PADDED_HEAD=True', 'PADDED_HEAD=False'))
    for ax in attn_fwd._built.axes:
        ax.godel_stride = None  # as if gen_functionals() had never run
    assert attn_fwd.tuning_representative(padded).godel_number == unpadded.godel_number


def test_shared_entries_have_identical_candidates(tmp_path):
    from aotriton.gpu_targets import AOTRITON_ARCH_WARPSIZE
    kernels = _tunable_kernels()
    assert kernels, 'no tunable kernel linked; the check below would pass vacuously'
    for kdesc in kernels:
        seen = set()
        for arch in sorted(AOTRITON_ARCH_WARPSIZE):
            # Archs with the same configs give the same verdict: the code
            # generator's per-functional steps do not depend on the arch.
            fingerprint = tuple((f.godel_number, tuple(map(repr, kdesc.gen_autotune_configs(f))))
                                for f in _functionals(kdesc, arch))
            if fingerprint not in seen:
                seen.add(fingerprint)
                _assert_shared_entries_match(kdesc, arch, tmp_path)


@pytest.mark.parametrize('padded', [lambda f: f.choices.PADDED_HEAD,
                                    lambda f: f.compact_choices.get('PADDED_HEAD').triton_compile_signature])
def test_config_generator_cannot_depend_on_a_fallback_axis(padded):
    """#220's mistake -- a search space that depends on PADDED_HEAD -- fails loudly,
    however the generator reads the axis."""
    attn_fwd = _attn_fwd()
    tune = attn_fwd._built.tune
    original = tune.configs
    tune.configs = lambda f: [] if padded(f) else list(original(f))
    try:
        twin = next(f for f in _functionals(attn_fwd, 'gfx1100') if 'PADDED_HEAD=True' in f.tunecc_signature)
        with pytest.raises(FallbackAxisRead, match='PADDED_HEAD'):
            attn_fwd.gen_autotune_configs(twin)
    finally:
        tune.configs = original


@pytest.mark.parametrize('key, value', [('PADDED_HEADS', False), ('PADDED_HEAD', 0)])
def test_fallback_must_name_an_axis_value(key, value):
    attn_fwd = _attn_fwd()
    fallback = attn_fwd._built.tune.fallback
    saved = dict(fallback)
    fallback[key] = value
    try:
        with pytest.raises(ValueError, match=key):
            attn_fwd.tuning_representative(_functionals(attn_fwd, 'gfx1100')[0])
    finally:
        fallback.clear()
        fallback.update(saved)
