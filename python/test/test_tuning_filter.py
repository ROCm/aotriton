# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU regressions for scoped tuning builds against the real flash descriptions."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from aotriton.codegen.autotune import AutotuneCodeGenerator
from aotriton.codegen.common import hsaco_dir, hsaco_ondisk_name
from aotriton.codegen.linker import Linker
from aotriton.codegen.tuning_filter import TuningFilter
from aotriton.generate import parse
from aotriton.gpu_targets import cluster_gpus
from aotriton.utils import RegistryRepository


REPO = Path(__file__).resolve().parents[2]
PROFILE = REPO / '.devcontainer/tuning/gfx908-inference.json'


@pytest.fixture
def flash():
    kernels, *_ = Linker(REPO / 'modules').link_all_families()
    selector = TuningFilter.from_file(PROFILE)
    fwd = next(k for k in kernels if k.NAME == 'attn_fwd')
    functionals = [f for f in fwd.gen_functionals(cluster_gpus(['gfx908_mod0']))
                   if not fwd.is_functional_disabled(f)]
    selected = next(f for f in functionals if selector.matches(f))
    return selector, fwd, functionals, selected


def generator(tmp_path, functional, selector, *, dataframe=None, tuning=True,
              second_pass=False):
    args = SimpleNamespace(build_dir=tmp_path, build_for_tuning=tuning,
                           build_for_tuning_second_pass=second_pass,
                           tuning_filter=selector,
                           _should_raise_for_lut=lambda _: True)
    return AutotuneCodeGenerator(args, functional, dataframe, (), RegistryRepository())


def names(signatures):
    return [sig.hsaco_entry_name for sig in signatures]


def test_inference_profile_selects_only_intended_functionals(flash):
    selector, fwd, functionals, _ = flash
    selected = [f for f in functionals if selector.matches(f)]
    assert len(functionals) == 432
    assert len(selected) == 16
    assert {(f.choices.T_io, f.choices.BLOCK_DMODEL, f.choices.CAUSAL_TYPE,
             f.choices.PADDED_HEAD) for f in selected} == {
        (dtype, dim, causal, padded)
        for dtype in ('*fp16:16', '*bf16:16')
        for dim in (64, 128) for causal in (0, 3) for padded in (False, True)
    }
    assert all(not f.choices.ENABLE_DROPOUT and f.choices.BIAS_TYPE == 0
               for f in selected)
    assert not any(selector.matches(f) for f in
                   fwd.gen_functionals(cluster_gpus(['gfx942_mod0'])))


def test_scoped_build_retains_default_before_candidates(tmp_path, flash):
    selector, fwd, _, f = flash
    expected_lut, defaults, _ = fwd.translate_empty_dataframe(f)
    candidates = list(fwd.gen_signatures_for_tuning(f))
    gen = generator(tmp_path, f, selector)
    assert len(candidates) == 80
    assert names(gen.all_signatures) == names(defaults + candidates)
    np.testing.assert_array_equal(gen._lut_tensor, expected_lut)
    assert gen._binning_dict is None


def test_unmatched_kernel_keeps_normal_images(tmp_path, flash):
    selector, fwd, functionals, _ = flash
    f = next(f for f in functionals if not selector.matches(f))
    defaults = fwd.translate_empty_dataframe(f)[1]
    gen = generator(tmp_path, f, selector)
    assert names(gen.all_signatures) == names(defaults)


def test_candidates_are_deduplicated_against_base_and_each_other(tmp_path, flash, monkeypatch):
    selector, fwd, _, f = flash
    defaults = fwd.translate_empty_dataframe(f)[1]
    candidate = next(fwd.gen_signatures_for_tuning(f))
    monkeypatch.setattr(fwd, 'gen_signatures_for_tuning',
                        lambda _: iter(defaults + [candidate, candidate]))
    gen = generator(tmp_path, f, selector)
    assert names(gen.all_signatures) == names(defaults + [candidate])


@pytest.mark.parametrize('matched', [False, True])
def test_database_lut_and_image_indices_are_preserved(tmp_path, flash, monkeypatch, matched):
    selector, fwd, functionals, selected = flash
    f = selected if matched else next(f for f in functionals if not selector.matches(f))
    signatures = fwd.translate_empty_dataframe(f)[1] + [next(fwd.gen_signatures_for_tuning(f))]
    rows = []
    for length, sig in zip((1024, 8192), signatures):
        row = {'gpu': f.database_gpus[0], 'inputs$Max_seqlen_q': 1,
               'inputs$Max_seqlen_k': length}
        # SQLite tuning columns store boolean knobs as integers.
        row.update({f'tuned_kernel${name}': int(value) if isinstance(value, bool) else value
                    for name, value in sig.perf_cdict.items()})
        row.update({f'compiler_options${name}': value for name, value in sig.copt_dict.items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    expected_lut, expected_sigs, expected_bins = fwd.translate_dataframe(f, df.copy())
    # This test exercises DB translation and preservation; the family-specific
    # completeness policy is independent of the selector and this two-row fixture.
    checked = []
    def sancheck(functional, lut):
        checked.append(functional)
        np.testing.assert_array_equal(lut, expected_lut)
        return True, [], []
    monkeypatch.setattr(fwd, 'sancheck_lut_tensor', sancheck)
    gen = generator(tmp_path, f, selector, dataframe=df)
    assert checked == [f]
    np.testing.assert_array_equal(gen._lut_tensor, expected_lut)
    assert list(gen._binning_dict) == list(expected_bins)
    assert names(gen.all_signatures[:len(expected_sigs)]) == names(expected_sigs)
    assert len(gen.all_signatures) == (81 if matched else 2)
    assert len(set(names(gen.all_signatures))) == len(gen.all_signatures)


def test_second_pass_keeps_failed_candidate_at_same_index(tmp_path, flash):
    selector, fwd, _, f = flash
    first = generator(tmp_path, f, selector)
    failed = first.all_signatures[1]
    image = hsaco_dir(tmp_path, fwd) / hsaco_ondisk_name(fwd, failed)
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b'')
    image.with_suffix('.json').write_text('{"compile_status": "Crash"}')
    second = generator(tmp_path, f, selector, second_pass=True)
    assert names(second.all_signatures) == names(first.all_signatures)


def test_unrestricted_tuning_behavior_is_unchanged(tmp_path, flash):
    _, fwd, _, f = flash
    gen = generator(tmp_path, f, None)
    assert names(gen.all_signatures) == names(list(fwd.gen_signatures_for_tuning(f)))


def test_normal_build_behavior_is_unchanged(tmp_path, flash):
    _, fwd, _, f = flash
    gen = generator(tmp_path, f, None, tuning=False)
    assert names(gen.all_signatures) == names(fwd.translate_empty_dataframe(f)[1])


def test_cli_requires_tuning_mode(monkeypatch, capsys):
    monkeypatch.setattr('sys.argv', ['generate', '--tuning_filter', str(PROFILE)])
    with pytest.raises(SystemExit) as error:
        parse()
    assert error.value.code == 2
    assert '--tuning_filter requires --build_for_tuning' in capsys.readouterr().err


def test_cli_loads_selector(monkeypatch):
    monkeypatch.setattr('sys.argv', ['generate', '--build_for_tuning',
                                   '--tuning_filter', str(PROFILE)])
    args = parse()
    assert args.tuning_filter == TuningFilter.from_file(PROFILE)
