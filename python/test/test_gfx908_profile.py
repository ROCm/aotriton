# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""CPU checks of the measured MI100 database and its generated C++ dispatch."""

import io
import itertools
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from aotriton.autotune import BinningExact
from aotriton.codegen.autotune import AutotuneCodeGenerator
from aotriton.codegen.linker import Linker
from aotriton.codegen.tuning_filter import TuningFilter
from aotriton.database.sqlite import Factory
from aotriton.database_compose import compose_database
from aotriton.gpu_targets import cluster_gpus
from aotriton.utils import RegistryRepository


REPO = Path(__file__).resolve().parents[2]
DATABASE = REPO / 'modules/flash/database'
SHARD = DATABASE / 'amd/gfx908'
SELECTOR = REPO / '.devcontainer/tuning/gfx908-inference.json'


@pytest.fixture
def profile(tmp_path):
    kernels, *_ = Linker(REPO / 'modules').link_all_families()
    fwd = next(k for k in kernels if k.NAME == 'attn_fwd')
    functionals = [f for f in fwd.gen_functionals(cluster_gpus(['gfx908_mod0']))
                   if not fwd.is_functional_disabled(f)]
    root = tmp_path / 'flash/database'
    root.mkdir(parents=True)
    for archive in DATABASE.glob('*.sqlite3.tar.xz'):
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter='data')
    shard_dir = root / 'amd/gfx908'
    shard_dir.mkdir(parents=True)
    with tarfile.open(SHARD / 'attn_fwd.sqlite3.tar.xz') as tar:
        tar.extractall(shard_dir, filter='data')
    with sqlite3.connect(root / 'tuning_database.sqlite3') as conn:
        compose_database(SimpleNamespace(decomposed=root, vendors=['amd']), conn)
    factory = Factory(root.parent)
    entries = json.loads((SHARD / 'attn_fwd.json').read_text())['entries']
    yield fwd, functionals, factory, entries
    factory._conn.close()


def selected_entries(f, entries):
    c = f.choices
    dtype = {'*fp16:16': 'float16', '*bf16:16': 'bfloat16'}.get(c.T_io)
    return [e for e in entries if e['dtype'] == dtype
            and e['block_dmodel'] == c.BLOCK_DMODEL
            and e['padded_head'] == c.PADDED_HEAD
            and (3 if e['causal'] else 0) == c.CAUSAL_TYPE
            and not c.ENABLE_DROPOUT and c.BIAS_TYPE == 0]


def generator(tmp_path, f, factory, *, tuning=False, selector=None):
    args = SimpleNamespace(build_dir=tmp_path, build_for_tuning=tuning,
                           build_for_tuning_second_pass=False,
                           tuning_filter=selector, _should_raise_for_lut=lambda _: True)
    df, sql = factory.create_view(f)
    return AutotuneCodeGenerator(args, f, df, sql, RegistryRepository())


def names(sigs):
    return [s.hsaco_entry_name for s in sigs]


def test_database_matches_measurements_and_compilable_candidates(tmp_path, profile):
    fwd, functionals, factory, entries = profile
    rows = factory._conn.execute('SELECT COUNT(*) FROM FLASH$attn_fwd').fetchone()[0]
    assert rows == len(entries) == 24
    covered = 0
    extra_images = 0
    for f in functionals:
        measured = selected_entries(f, entries)
        gen = generator(tmp_path, f, factory)
        defaults = fwd.translate_empty_dataframe(f)[1]
        assert names(gen.all_signatures[:1]) == names(defaults)
        if not measured:
            assert names(gen.all_signatures) == names(defaults)
            assert fwd.exact_tuning(f) is None
            continue
        covered += 1
        extra_images += len(gen.all_signatures) - 1
        assert all(isinstance(b, BinningExact) for b in gen._binning_dict.values())
        assert fwd.sancheck_lut_tensor(f, gen._lut_tensor)[0]
        assert (gen._lut_tensor >= 0).all()
        candidates = set(names(defaults + list(fwd.gen_signatures_for_tuning(f))))
        assert set(names(gen.all_signatures)) <= candidates
        bins = [b.representatives for b in gen._binning_dict.values()]
        for e in measured:
            index = gen._lut_tensor[0, bins[0].index(e['seqlen_q']),
                                    bins[1].index(e['seqlen_k'])]
            sig = gen.all_signatures[index]
            assert sig.perf_cdict == e['psels']
            assert sig.copt_dict == e['copts']
            assert {(m['batch'], tuple(m['heads'])) for m in e['measurements']} == {
                (1, (32, 32)), (1, (32, 8))}
        measured_keys = {(e['seqlen_q'], e['seqlen_k']) for e in measured}
        for iq, ik in itertools.product(range(len(bins[0])), range(len(bins[1]))):
            if (bins[0][iq], bins[1][ik]) not in measured_keys:
                assert gen._lut_tensor[0, iq, ik] == 0
    assert covered == 16
    assert extra_images == 16


def test_scoped_and_unrestricted_tuning_with_installed_profile(tmp_path, profile):
    fwd, functionals, factory, entries = profile
    selector = TuningFilter.from_file(SELECTOR)
    for f in functionals:
        if not selected_entries(f, entries):
            continue
        normal = generator(tmp_path, f, factory)
        scoped = generator(tmp_path, f, factory, tuning=True, selector=selector)
        full = generator(tmp_path, f, factory, tuning=True)
        assert names(scoped.all_signatures[:len(normal.all_signatures)]) == names(normal.all_signatures)
        np.testing.assert_array_equal(scoped._lut_tensor, normal._lut_tensor)
        assert len(scoped.all_signatures) == 81
        assert names(full.all_signatures) == names(list(fwd.gen_signatures_for_tuning(f)))
        assert full._binning_dict is None
        assert not full._lut_tensor.any()


def test_missing_profile_values_and_other_architectures_still_fail_sanity(profile):
    fwd, functionals, _, entries = profile
    f = next(f for f in functionals if selected_entries(f, entries))
    assert not fwd.sancheck_lut_tensor(f, np.full((1, 2, 2), -1))[0]
    assert not fwd.sancheck_lut_tensor(f, np.zeros((1, 2)))[0]
    assert not fwd.sancheck_lut_tensor(f, np.zeros((1, 0, 2)))[0]
    for gpu in ('gfx90a_mod0', 'gfx942_mod0', 'gfx1100_mod0'):
        other = next(fwd.gen_functionals(cluster_gpus([gpu])))
        assert fwd.exact_tuning(other) is None
        assert not fwd.sancheck_lut_tensor(other, np.zeros((1, 2, 2)))[0]


def test_generated_cpp_exact_coverage_and_tuning_override(tmp_path, profile):
    """Execute the actual emitted LUT and template selection code on the CPU.

    No ROCm headers, library, device discovery, or kernel execution is involved.
    Compile both production and tuning branches, including debug assertions.
    """
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler required for generated lookup regression')
    fwd, functionals, factory, entries = profile
    source = ['#include <cassert>', '#include <cstdint>',
              '#define AOTRITON_LOG(...)',
              f'struct {fwd.param_class_name} {{',
              'int Batch, Num_head_q, Num_head_k, Hdim_qk, Hdim_vo, Varlen_bits;',
              'int Max_seqlen_q, Max_seqlen_k;', '};']
    tests = []
    for i, f in enumerate(functionals):
        measured = selected_entries(f, entries)
        if not measured:
            continue
        gen = generator(tmp_path, f, factory)
        out = io.StringIO()
        gen.write_autotune_src(out)
        ctype, cshape, cdata = gen.codegen_format_lut(gen._lut_tensor)
        functions = gen._parent_repo.get_data('lut_function')
        [(lookup, item)] = functions.items()
        # Use exactly the LUT declaration and lookup function emitted by codegen.
        source += [f'namespace case_{i} {{', f'{ctype} lut{cshape} = {cdata};',
                   f'{item.ret} lookup {lookup};']
        # Extract selection through the bounds check, before device/image access.
        body = out.getvalue().split('int CURRENT_ENTRY_PUBLIC', 1)[1].split('{', 1)[1]
        body = body.split('    if (kernel_index < 0', 1)[0]
        lut_function = next(iter(functions.values())).name
        body = body.replace(lut_function, 'lookup')
        for tuning in (0, 1):
            source += [f'#define AOTRITON_BUILD_FOR_TUNING {tuning}',
                       f'int choose_{tuning}({fwd.param_class_name}& params, int override_index = -1) {{',
                       f'struct {{ {fwd.param_class_name}* params; int _has_preferred_kernel; }}',
                       'context{&params, override_index}; int mod_number = 0;', body,
                       'return kernel_index; }', '#undef AOTRITON_BUILD_FOR_TUNING']
        source.append('}')
        bins = [b.representatives for b in gen._binning_dict.values()]
        hdim = measured[0]['hdim']
        known = {(e['seqlen_q'], e['seqlen_k']) for e in measured}
        pairs = set(itertools.product(*bins))
        # Adjacent and out-of-range lengths must not inherit a nearby winner.
        for q, k in known:
            pairs.update((q+dq, k+dk) for dq, dk in ((-1, 0), (1, 0), (0, -1), (0, 1)))
        pairs.update(((1, 2147483647), (2147483647, 8192), (-1, -1)))
        tests.append('{')
        for q, k in sorted(pairs):
            expected = (int(gen._lut_tensor[0, bins[0].index(q), bins[1].index(k)])
                        if (q, k) in known else 0)
            for heads_k in (8, 32):
                tests += [f'{fwd.param_class_name} p_{q if q >= 0 else "neg"}_{k if k >= 0 else "neg"}_{heads_k}'
                          f'{{1, 32, {heads_k}, {hdim}, {hdim}, 0, {q}, {k}}};']
                p = f'p_{q if q >= 0 else "neg"}_{k if k >= 0 else "neg"}_{heads_k}'
                tests += [f'assert(case_{i}::choose_0({p}) == {expected});',
                          f'assert(case_{i}::choose_1({p}) == {expected});',
                          f'assert(case_{i}::choose_1({p}, 0) == 0);',
                          f'assert(case_{i}::choose_1({p}, 1) == 1);']
        e = measured[0]
        for field, value in (('Batch', 2), ('Num_head_q', 16), ('Num_head_k', 1),
                             ('Hdim_qk', hdim-1), ('Hdim_vo', hdim-1),
                             ('Varlen_bits', 1), ('Varlen_bits', 65536)):
            tests += ['{', f'{fwd.param_class_name} p{{1, 32, 8, {hdim}, {hdim}, 0, '
                      f'{e["seqlen_q"]}, {e["seqlen_k"]}}};', f'p.{field} = {value};',
                      f'assert(case_{i}::choose_0(p) == 0);',
                      f'assert(case_{i}::choose_1(p) == 0);', '}']
        tests.append('}')
    source += ['int main() {', *tests, '}']
    cpp = tmp_path / 'gfx908_lookup.cc'
    cpp.write_text('\n'.join(source))
    binary = tmp_path / 'gfx908_lookup'
    compiled = subprocess.run([compiler, '-std=c++20', '-Wall', '-Wextra',
                               str(cpp), '-o', str(binary)], capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    executed = subprocess.run([str(binary)], capture_output=True, text=True)
    assert executed.returncode == 0, executed.stderr
