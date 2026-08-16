# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase 1 step 7 (modular-tune.md): non-GPU regression coverage for the
relocated tuning infrastructure (`python/tune/` -> `aotriton.tune`,
`modules/flash/tune/` loaded by path).

None of this needs a GPU or a database connection. Two of the three checks
below (`FlashEntry`/`FlashInputMetadata` and anything touching
`modules/flash/tune/flash/module.py`) transitively need `dacite`; most of
`aotriton.tune.pq`/`aotriton.tune.localq` need `psycopg`. Neither is in the
base `requirements.txt` (only `requirements-tuning.txt` pulls them in, and no
`.ci/*.sh` installs it before running this suite), so every check here that
needs one of those two packages skips cleanly via `pytest.importorskip`/a
per-module ImportError probe instead of hard-failing when they are absent.
"""

import ast
import importlib
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for conftest-adjacent helpers, if any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULES_DIR = _REPO_ROOT / 'modules'


# --- (a) F6: modules/<family> stays a plain directory; the by-path loader ---
#     resolves modules/flash/tune by path, not through sys.path, so a stray
#     top-level `flash.py` module cannot shadow it.

def test_family_loader_ignores_stray_sys_path_module():
    from aotriton.tune.registry import load_family_tune

    with tempfile.TemporaryDirectory() as tmp:
        decoy = Path(tmp) / 'flash.py'
        decoy.write_text('DECOY = True\n')
        sys.path.insert(0, tmp)
        try:
            # Sanity: prove the collision risk is real -- a naive `import flash`
            # from this sys.path *does* pick up the decoy.
            sys.modules.pop('flash', None)
            decoy_mod = importlib.import_module('flash')
            assert getattr(decoy_mod, 'DECOY', False) is True

            # The by-path loader must not be fooled by the same sys.path entry:
            # it resolves modules/flash/tune/__init__.py directly by file path
            # (submodule_search_locations=[tune_dir]), never falling back to a
            # global `sys.path` lookup for the top-level name 'flash'.
            family_pkg = load_family_tune('flash', modules_dir=_MODULES_DIR)
            assert not hasattr(family_pkg, 'DECOY')
            expected_init = (_MODULES_DIR / 'flash' / 'tune' / '__init__.py').resolve()
            assert Path(family_pkg.__file__).resolve() == expected_init
        finally:
            sys.path.remove(tmp)
            sys.modules.pop('flash', None)


def test_load_tune_module_resolves_flash_and_flash_op_to_distinct_submodules():
    pytest.importorskip('dacite')
    from aotriton.tune.registry import load_tune_module

    flash_mod = load_tune_module('flash', modules_dir=_MODULES_DIR)
    flash_op_mod = load_tune_module('flash_op', modules_dir=_MODULES_DIR)
    assert flash_mod is not flash_op_mod
    assert hasattr(flash_mod, 'TuneDesc') and hasattr(flash_mod, 'ImplSelector')
    assert hasattr(flash_op_mod, 'TuneDesc') and hasattr(flash_op_mod, 'ImplSelector')
    # Phase 1: two independent descriptions, not yet unified (Phase 2).
    assert flash_mod.TuneDesc is not flash_op_mod.TuneDesc


# --- (b) F5: the codegen-side FlashEntry copy (modules/flash/aot/flash_entry.py) ---
#     must stay byte-identical, in `as_text()` output, to the tuning-side
#     FlashEntry (modules/flash/tune/flash/module.py) it was split off from.

def _load_codegen_flash_entry():
    import importlib.util
    path = _MODULES_DIR / 'flash' / 'aot' / 'flash_entry.py'
    spec = importlib.util.spec_from_file_location('_test_codegen_flash_entry', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FlashEntry


def test_flash_entry_as_text_matches_codegen_copy():
    pytest.importorskip('dacite')
    from aotriton.tune.registry import load_flash_entry_module

    TuningFlashEntry = load_flash_entry_module(modules_dir=_MODULES_DIR).FlashEntry
    CodegenFlashEntry = _load_codegen_flash_entry()

    kwargs = dict(dtype='bfloat16', hdim=128, seqlen_q=256, seqlen_k=512,
                  causal=True, dropout_p=0.5, bias_type=1)
    a = TuningFlashEntry(**kwargs)
    b = CodegenFlashEntry(**kwargs)
    assert a.as_text() == b.as_text()

    # Also cover a tuple-valued hdim (hdim_qk != hdim_v) and defaults.
    a2 = TuningFlashEntry(hdim=(64, 128))
    b2 = CodegenFlashEntry(hdim=(64, 128))
    assert a2.as_text() == b2.as_text()


# --- (c) pq/localq stay torch-free at import time -----------------------------

def _iter_pq_localq_module_names():
    for pkg in ('pq', 'localq'):
        root = _REPO_ROOT / 'python' / 'tune' / pkg
        for path in sorted(root.rglob('*.py')):
            if path.name == '__init__.py':
                dotted = path.parent.relative_to(_REPO_ROOT / 'python').as_posix().replace('/', '.')
            else:
                dotted = path.relative_to(_REPO_ROOT / 'python').with_suffix('').as_posix().replace('/', '.')
            yield f'aotriton.{dotted}'


_PQ_LOCALQ_MODULES = sorted(set(_iter_pq_localq_module_names()))


@pytest.mark.parametrize('module_name', _PQ_LOCALQ_MODULES)
def test_pq_localq_module_imports_without_torch(module_name):
    assert 'torch' not in sys.modules, (
        f'torch was already imported before importing {module_name}; '
        'this test cannot tell whether the module itself pulled it in.')
    try:
        importlib.import_module(module_name)
    except ImportError as e:
        # Optional tuning-only deps (psycopg, dacite) may genuinely be absent
        # in this environment (see module docstring); anything else is a real
        # failure.
        missing = getattr(e, 'name', None) or ''
        if missing.split('.')[0] in ('psycopg', 'dacite'):
            pytest.skip(f'{module_name}: optional dependency {missing!r} not installed')
        raise
    assert 'torch' not in sys.modules, f'{module_name} imported torch at import time'


def test_pq_localq_module_list_is_not_empty():
    # Guards against the discovery glob above silently matching nothing (e.g.
    # if python/tune/pq or python/tune/localq were ever renamed/moved again).
    assert len(_PQ_LOCALQ_MODULES) >= 20
    assert 'aotriton.tune.pq.queue' in _PQ_LOCALQ_MODULES
    assert 'aotriton.tune.localq.broker' in _PQ_LOCALQ_MODULES
    assert 'aotriton.tune.pq.vis_descriptors' in _PQ_LOCALQ_MODULES
    assert 'aotriton.tune.pq.vis_descriptors.flash' in _PQ_LOCALQ_MODULES


def main():
    raise SystemExit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
    main()
