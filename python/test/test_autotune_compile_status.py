# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for handling failed kernel artifacts during second-pass codegen.

The generator is subclassed to test the status and LUT logic without a GPU,
compiler, database, or the rest of its construction pipeline.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from aotriton.codegen.autotune import AutotuneCodeGenerator
from aotriton.codegen.common import (
    NoCompiledKernel,
    hsaco_dir,
    hsaco_ondisk_name,
)


class _Sig:
    def __init__(self, name):
        self.hsaco_entry_name = name

    def __repr__(self):
        return f'_Sig({self.hsaco_entry_name!r})'


class _Kdesc:
    FAMILY = 'fakefamily'
    NAME = 'fake_kernel'
    TUNE_NAME = 'autotune'


class _Functional:
    arch = 'gfx1103'
    name = 'fake_functional'
    tunecc_signature = 'fake_kernel-fakefunctional'
    meta_object = _Kdesc()
    optimized_for = ['gfx1103']  # codegen_format_lut labels one block per GPU


class _Args:
    def __init__(self, build_dir=None, *, second_pass=False):
        self.build_dir = build_dir
        self.build_for_tuning_second_pass = second_pass


class _Kernel:
    def __init__(self, unique_path):
        self.unique_path = Path(unique_path)


class _Completed:
    returncode = 0


class _Generator(AutotuneCodeGenerator):
    def __init__(self, sigs, lut_tensor, compiled):
        self._sigs = list(sigs)
        self._lut_tensor = np.asarray(lut_tensor)
        self._lut_ctype_hint = None
        self._f = _Functional()
        self._args = _Args()
        self._compiled = dict(zip(sigs, compiled))

    def hsaco_compile_successful(self, ksig):
        return self._compiled[ksig]


def _sigs(n):
    return [_Sig(f'kernel-{i}') for i in range(n)]


_COMPLETE = json.dumps({'compile_status': 'Complete'})
_TIMEOUT = json.dumps({'compile_status': 'Timeout'})


@pytest.mark.parametrize('image,status,usable,reported', [
    # Image and status are required independently. Malformed metadata is reported;
    # missing artifacts are treated as failed without a corruption warning.
    (b'\x7fELF', _COMPLETE,           True,  False),
    (b'\x7fELF', _TIMEOUT,            False, False),
    (None,       _COMPLETE,           False, False),
    (b'\x7fELF', None,                False, False),
    (b'',        _COMPLETE,           False, False),
    (b'\x7fELF', '{"compile_status":', False, True),  # ValueError
    (b'\x7fELF', '{}',                False, True),     # KeyError
])
def test_a_kernel_is_usable_only_with_an_image_and_a_complete_status(
        tmp_path, capsys, image, status, usable, reported):
    """Lay out one kernel the way compile.py does, then read its status back."""
    sig = _Sig('kernel-under-test')
    image_dir = hsaco_dir(tmp_path, _Kdesc)
    image_dir.mkdir(parents=True, exist_ok=True)
    path = image_dir / hsaco_ondisk_name(_Kdesc, sig)
    if image is not None:
        path.write_bytes(image)
    if status is not None:
        path.with_suffix('.json').write_text(status)

    gen = _Generator.__new__(_Generator)
    gen._f = _Functional()
    gen._args = _Args(tmp_path)
    assert AutotuneCodeGenerator.hsaco_compile_successful(gen, sig) is usable

    err = capsys.readouterr().err
    if reported:
        assert path.with_suffix('.json').name in err
    else:
        assert err == '', 'a kernel not built yet is not a corrupt one'


def test_only_the_cells_of_a_failed_kernel_move_and_they_move_to_the_busiest_survivor(capsys):
    """Repoint failed cells without changing unaffected cells or signature order.

    Signature order fixes offsets into the first-pass shim's packed strings.
    """
    sigs = _sigs(5)
    #           0 failed and busiest overall, 4 failed but unreferenced
    compiled = [False, True, True, True, False]
    lut = np.array([[-1] * 4 + [0] * 10 + [1] * 2 + [2] * 5 + [3] * 3])
    gen = _Generator(sigs, lut.copy(), compiled)
    gen.repoint_failed_lut_cells()
    new = gen._lut_tensor

    assert len(gen._sigs) == len(sigs)
    for got, want in zip(gen._sigs, sigs):
        assert got is want, 'the signature list did not survive the pass untouched'

    failed = [i for i, good in enumerate(compiled) if not good]
    assert not set(new.ravel().tolist()) & set(failed), 'a failed kernel is still reachable'

    kept = ~np.isin(lut, failed)
    assert (new[kept] == lut[kept]).all(), 'a cell that did not have to move was moved'

    moved_to = set(new[~kept].tolist())
    assert len(moved_to) == 1
    cells = {i: int((lut == i).sum()) for i, good in enumerate(compiled) if good}
    assert moved_to == {max(cells, key=cells.get)}

    err = capsys.readouterr().err
    for i in failed:
        assert sigs[i].hsaco_entry_name in err
    # Named once, for the one failure that actually cost something.
    assert err.count(sigs[moved_to.pop()].hsaco_entry_name) == 1



def test_repointing_does_not_narrow_the_generated_lut_type():
    """Keep the first-pass LUT type after its highest index is repointed."""
    sigs = _sigs(128)
    compiled = [True] * len(sigs)
    compiled[-1] = False
    gen = _Generator(sigs, [[0, 1, 2, 3, 4, 5, len(sigs) - 1]], compiled)
    assert gen.codegen_format_lut(gen._lut_tensor)[0] == 'int16_t'
    gen.repoint_failed_lut_cells()
    assert int(gen._lut_tensor.max()) < 127, 'the narrowing this test guards against'
    assert gen.codegen_format_lut(gen._lut_tensor)[0] == 'int16_t'


def test_a_functional_with_nothing_left_to_dispatch_to_fails_the_build():
    """Fail when no compiled signature remains as a replacement."""
    sigs = _sigs(3)
    gen = _Generator(sigs, [[0, 1, 2]], [False] * len(sigs))

    with pytest.raises(NoCompiledKernel) as excinfo:
        gen.repoint_failed_lut_cells()
    rendered = str(excinfo.value)
    assert _Functional.tunecc_signature in rendered
    assert _Functional.arch in rendered
    for sig in sigs:
        assert sig.hsaco_entry_name in rendered


def test_a_default_only_functional_reports_a_compile_failure(capsys):
    """Report a failed default when there is no alternative to select."""
    only = _sigs(1)
    gen = _Generator(only, [[0]], [False])
    gen.warn_if_default_failed()

    warned = capsys.readouterr().err
    assert _Functional.arch in warned
    assert _Functional.tunecc_signature in warned
    assert only[0].hsaco_entry_name in warned

    gen = _Generator(only, [[0]], [True])
    gen.warn_if_default_failed()
    assert capsys.readouterr().err == ''


def test_second_pass_does_not_append_tuning_manifest(tmp_path):
    """Only the first pass records generated tuning-table sources."""
    functional = _Functional()
    args = _Args(tmp_path, second_pass=False)
    gen = _Generator.__new__(_Generator)
    gen._args = args
    gen._f = functional

    path = gen.get_cc_file(functional)
    manifest = path.parent / 'manifest.nsv'
    before = manifest.read_bytes()

    args.build_for_tuning_second_pass = True
    assert gen.get_cc_file(functional) == path
    assert manifest.read_bytes() == before


@pytest.mark.parametrize('has_df,tuning,second_pass,noimage,expect', [
    # A tuned build repoints LUT cells; a tuning build drops signatures instead.
    (True,  False, True,  False, 'repoint'),
    (True,  False, False, False, 'nothing'),  # first pass: no compile status yet
    (True,  False, True,  True,  'nothing'),  # nothing to read under --noimage_mode
    (True,  True,  True,  False, 'drop'),
    (True,  True,  True,  True,  'nothing'),  # build-tune.sh --shim: every signature
                                              # would look failed, leaving empty arrays
    (True,  True,  False, False, 'nothing'),
    # Without tuning data there is one default signature and nothing to repoint
    # it to, so the untuned path only warns.
    (False, False, True,  False, 'warn'),
])
def test_the_second_pass_runs_the_branch_that_matches_the_build(
        monkeypatch, has_df, tuning, second_pass, noimage, expect):
    """Pin the helper call sites and no-image guards in __init__."""
    from aotriton.codegen import autotune as A
    from aotriton.codegen import basetune as B

    sigs = _sigs(2)
    lut = np.array([[0, 1]])

    class _BothKdesc(_Kdesc):
        is_tunable = True

        def translate_dataframe(self, f, df):
            return lut.copy(), list(sigs), {}

        def translate_empty_dataframe(self, f):
            return lut.copy(), list(sigs), {}

        def gen_signatures_for_tuning(self, f):
            return iter(sigs)

        def sancheck_lut_tensor(self, f, lut_tensor):
            return True, [], []

    class _NonEmptyDf:
        empty = False

    functional = _Functional()
    functional.meta_object = _BothKdesc()

    args = _Args()
    args.build_for_tuning = tuning
    args.build_for_tuning_second_pass = second_pass
    args.noimage_mode = noimage

    def _base_init(self, args, f, df, parent_repo):
        self._args, self._f, self._df, self._parent_repo = args, f, df, parent_repo

    monkeypatch.setattr(B.BaseTuneCodeGenerator, '__init__', _base_init)
    # The stand-in signatures are not KernelSignature instances, and __init__
    # ends by asserting they are.
    monkeypatch.setattr(A, 'KernelSignature', _Sig)
    # The first signature is the one the compiler rejected.
    monkeypatch.setattr(A.AutotuneCodeGenerator, 'hsaco_compile_successful',
                        lambda self, ksig: ksig is not sigs[0])

    repointed = []
    monkeypatch.setattr(A.AutotuneCodeGenerator, 'repoint_failed_lut_cells',
                        lambda self: repointed.append(True))
    warned = []
    monkeypatch.setattr(A.AutotuneCodeGenerator, 'warn_if_default_failed',
                        lambda self: warned.append(True))

    gen = A.AutotuneCodeGenerator(args, functional,
                                  _NonEmptyDf() if has_df else None, (), None)

    assert bool(repointed) is (expect == 'repoint')
    assert bool(warned) is (expect == 'warn')
    if expect == 'drop':
        assert gen._sigs == sigs[1:]
    else:
        assert gen._sigs == sigs


def test_the_second_pass_leaves_the_build_manifests_alone(tmp_path, monkeypatch):
    """Leave first-pass Bare.* files unchanged when workers write no shards.

    The seeded shards differ from those files, so an accidental merge is visible.
    """
    from aotriton.codegen import root as R

    items = ['flash/triton/attn_fwd', 'flash/triton/bwd_kernel_dq']
    names = ['Bare.shim', 'Bare.compile', 'Bare.cluster', 'Affine.cluster', 'Bare.flatzip']
    for item in items:
        shard_dir = tmp_path / 'Bare.shards' / R._shard_path(item)
        shard_dir.mkdir(parents=True)
        for name in names:
            (shard_dir / name).write_text(f'stale-shard:{item}:{name}\n')
    for name in names:
        (tmp_path / name).write_text(''.join(f'{item}:{name}\n' for item in items))
    before = {name: (tmp_path / name).read_text() for name in names}

    gen = R.RootGenerator.__new__(R.RootGenerator)
    gen._args = _Args(tmp_path)
    gen._args.selective = None
    gen._args.build_for_tuning_second_pass = True
    gen._dispatcher_operators = []
    gen._affine_kernels = []
    gen._triton_kernels = [_Kernel(item) for item in items]

    spawned = []
    monkeypatch.setattr(R.subprocess, 'run',
                        lambda cmd, **kw: spawned.append(cmd) or _Completed())
    monkeypatch.setattr(R.sys, 'argv', ['generate', '--build_for_tuning_second_pass'])

    gen.launch_workers()

    assert len(spawned) == len(items), 'every item still gets a second-pass worker'
    for name in names:
        assert (tmp_path / name).read_text() == before[name]
