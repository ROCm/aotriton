# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase 1 step 7 / Phase 2 step 14 (modular-tune.md): non-GPU regression
coverage for the relocated and unified tuning infrastructure (`python/tune/`
-> `aotriton.tune`, `modules/flash/tune/` loaded by path).

None of this needs a GPU or a database connection. `entry.py`/`desc.py`/
`tdesc.py`/`utils.py` import `dacite` lazily (only inside the methods that
actually call `from_dict()`, e.g. `ENTRY_CLASS.from_dict()`, `run_single_test`)
so merely loading the flash family / constructing a `TuneDesc` never requires
`dacite` to be installed; `aotriton.tune.pq`/`aotriton.tune.localq` still need
`psycopg` for real DB access. Neither is in the base `requirements.txt` (only
`requirements-tuning.txt` pulls them in, and no `.ci/*.sh` installs it before
running this suite), so every check here that needs one of those two packages
skips cleanly via `pytest.importorskip`/a per-module ImportError probe instead
of hard-failing when they are absent.
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


def test_load_tune_module_flash_op_no_longer_exists():
    # Phase 2 (modular-tune.md §4.3): flash/flash_op collapse into a single
    # 'flash' family with a `level` axis ('kernel' | 'op'). 'flash_op' must no
    # longer resolve as an independent module name/queue key/CLI subcommand.
    from aotriton.tune.registry import load_tune_module, available_module_names

    assert available_module_names() == ['flash']
    with pytest.raises(ImportError):
        load_tune_module('flash_op', modules_dir=_MODULES_DIR)


# --- Phase 2 step 14 (modular-tune.md): ImplSelector DSL + fetch_tasks -------
#     highest-risk-area coverage (op.attn_fwd=1 is surface syntax only,
#     iface_name collisions across levels, fetch_tasks tuning_mode required).

def test_implselector_parse_text_roundtrip_prefixed():
    from aotriton.tune.tdesc import ImplSelector

    sel = ImplSelector.parse_text('op.attn_fwd=1')
    assert sel == ImplSelector(tuning_level='op', iface_name='attn_fwd', impl_index=1)
    # iface_name must be bare -- the 'op.' prefix is surface syntax only, never
    # part of the stored/queried interface name (highest-risk area #1).
    assert sel.iface_name == 'attn_fwd'
    assert sel.as_text() == 'op.attn_fwd=1'


def test_implselector_parse_text_roundtrip_unprefixed():
    from aotriton.tune.tdesc import ImplSelector

    sel = ImplSelector.parse_text('attn_fwd=3')
    assert sel == ImplSelector(tuning_level='kernel', iface_name='attn_fwd', impl_index=3)
    assert sel.as_text() == 'attn_fwd=3'


def test_implselector_unprefixed_defaults_to_kernel_level():
    from aotriton.tune.tdesc import ImplSelector

    sel = ImplSelector.parse_text('bwd_kernel_dq=0')
    assert sel.tuning_level == 'kernel'
    # Default-constructed selector (no tuning_level given) is also 'kernel'.
    assert ImplSelector(iface_name='attn_fwd', impl_index=0).tuning_level == 'kernel'


@pytest.mark.parametrize('malformed', [
    'attn_fwd',            # missing '='
    'attn_fwd=',           # missing index
    'attn_fwd=notanint',   # non-integer index
    'op.attn_fwd=1=2',     # too many '=' for str.split('=') with no maxsplit
    '',                    # empty string
])
def test_implselector_parse_text_rejects_malformed_input(malformed):
    from aotriton.tune.tdesc import ImplSelector

    with pytest.raises((ValueError, IndexError)):
        ImplSelector.parse_text(malformed)


def test_cross_level_attn_fwd_iface_name_collision():
    # 'attn_fwd' is a valid bare iface_name at BOTH flash's kernel level and
    # its op level (highest-risk area #2: iface_name collides across levels).
    # list_impls() and plain dataclass construction (unlike ENTRY_CLASS.from_dict())
    # need neither torch nor dacite, so this runs without a GPU or dacite installed.
    from aotriton.tune.registry import make_tune_desc

    entry_kwargs = dict(dtype='float16', hdim=64, seqlen_q=128, seqlen_k=128,
                         causal=False, dropout_p=0.0, bias_type=0)

    kernel_tune = make_tune_desc('flash', level='kernel', modules_dir=_MODULES_DIR)
    op_tune = make_tune_desc('flash', level='op', modules_dir=_MODULES_DIR)

    entry = kernel_tune.ENTRY_CLASS(**entry_kwargs)
    kernel_impls = kernel_tune.list_impls(entry)
    op_impls = op_tune.list_impls(entry)

    assert 'attn_fwd' in kernel_impls
    assert 'attn_fwd' in op_impls
    # Bare names only -- never the DSL's 'op.' prefix (highest-risk area #1).
    assert all('.' not in name for name in kernel_impls)
    assert all('.' not in name for name in op_impls)

    # The two levels' 'attn_fwd' selectors are distinguished only by the
    # ImplSelector.tuning_level field / DSL prefix, never by iface_name itself.
    from aotriton.tune.tdesc import ImplSelector
    kernel_sel = ImplSelector(tuning_level='kernel', iface_name='attn_fwd', impl_index=0)
    op_sel = ImplSelector(tuning_level='op', iface_name='attn_fwd', impl_index=0)
    assert kernel_sel.iface_name == op_sel.iface_name == 'attn_fwd'
    assert kernel_sel.as_text() == 'attn_fwd=0'
    assert op_sel.as_text() == 'op.attn_fwd=0'
    assert kernel_sel != op_sel


def test_flash_tune_bogus_level_raises():
    # TuningDescription.__init__ validates `level` against the subclass's
    # LEVELS dict and must reject anything else instead of silently building
    # a half-initialized description (modular-tune.md §5 step 14).
    from aotriton.tune.registry import make_tune_desc

    with pytest.raises(ValueError):
        make_tune_desc('flash', level='bogus', modules_dir=_MODULES_DIR)


def test_schema_sql_ddl_parses_and_iface_name_columns_carry_tuning_level():
    # No live PG in this environment/CI job -- fall back to a structural check
    # of the checked-in DDL: (1) balanced parens/quotes so it is at least
    # lexically well-formed, and (2) every table/index that has an
    # `iface_name` column also carries `tuning_level`, since `iface_name`
    # collides across tuning levels (highest-risk area #2) and any table or
    # lookup index keyed on it alone would silently mix kernel-level and
    # op-level rows.
    schema_sql = (_REPO_ROOT / 'python' / 'tune' / 'pq' / 'schema.sql').read_text()
    mat_views_sql = (_REPO_ROOT / 'python' / 'tune' / 'pq' / 'materialized_views.sql').read_text()

    for name, sql in (('schema.sql', schema_sql), ('materialized_views.sql', mat_views_sql)):
        # Lexical sanity: parens balance (ignoring $$ plpgsql bodies' own
        # dollar-quoting is not needed here -- schema.sql's only $$ body,
        # create_arch_partition, is itself paren-balanced).
        assert sql.count('(') == sql.count(')'), f'{name}: unbalanced parentheses'

    # CREATE TABLE blocks: split on 'CREATE TABLE' and check each block up to
    # its closing ');' for the iface_name/tuning_level pairing.
    import re
    for name, sql in (('schema.sql', schema_sql), ('materialized_views.sql', mat_views_sql)):
        for m in re.finditer(r'CREATE TABLE.*?\);', sql, re.DOTALL):
            block = m.group(0)
            if 'iface_name' in block:
                assert 'tuning_level' in block, (
                    f'{name}: a CREATE TABLE block has iface_name but no '
                    f'tuning_level column:\n{block}')

    # CREATE INDEX statements: any index over iface_name that is NOT already
    # anchored on task_id must also cover tuning_level. A task_id already
    # implies exactly one tuning_level via its task_queue row (see
    # pq/results.py's get_task_results docstring), so an (task_id,
    # iface_name, ...) index is unambiguous without tuning_level; but an
    # index meant to be queried by iface_name alone (no task_id/task_queue
    # join) must carry tuning_level or it would silently mix kernel-level and
    # op-level rows (highest-risk area #2).
    for name, sql in (('schema.sql', schema_sql), ('materialized_views.sql', mat_views_sql)):
        for m in re.finditer(r'CREATE (?:UNIQUE )?INDEX.*?;', sql, re.DOTALL):
            stmt = m.group(0)
            if 'iface_name' in stmt and 'task_id' not in stmt:
                assert 'tuning_level' in stmt, (
                    f'{name}: an index covers iface_name (without task_id) but not '
                    f'tuning_level:\n{stmt}')


def test_fetch_tasks_requires_tuning_mode_keyword():
    # F16 (modular-tune.md): a kernel worker must never claim an op task and
    # vice versa. fetch_tasks() must have NO default for tuning_mode so a
    # caller that forgets to pass it fails fast at the call site instead of
    # silently defaulting to 'kernel'.
    pytest.importorskip('psycopg')
    import inspect
    from aotriton.tune.pq.queue import TaskQueue

    sig = inspect.signature(TaskQueue.fetch_tasks)
    tuning_mode_param = sig.parameters['tuning_mode']
    assert tuning_mode_param.kind == inspect.Parameter.KEYWORD_ONLY
    assert tuning_mode_param.default is inspect.Parameter.empty

    # A conn is never touched before the missing-keyword TypeError fires.
    task_queue = TaskQueue(conn=None)
    with pytest.raises(TypeError):
        task_queue.fetch_tasks('gfx942', batch_size=1)


def test_fetch_tasks_sql_filters_on_tuning_level():
    # The SQL itself must filter on the denormalized tuning_level column
    # (not a `module LIKE '%_op'` string-suffix pattern -- the pre-Phase-2
    # anti-pattern this replaces).
    pytest.importorskip('psycopg')
    import inspect
    from aotriton.tune.pq.queue import TaskQueue

    source = inspect.getsource(TaskQueue.fetch_tasks)
    assert 'tuning_level = %s' in source
    assert "LIKE '%_op'" not in source
    assert "NOT LIKE '%_op'" not in source


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
    # as_text()/plain construction need neither torch nor dacite (only
    # ENTRY_CLASS.from_dict() does -- see entry.py's lazy dacite import).
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
