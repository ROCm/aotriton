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
import json
import subprocess
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
    #
    # Revision note 3 (modular-tune.md): there is only ONE TuneDesc -- no
    # per-level construction -- and list_impls() reports both levels' impls
    # together, DSL-spelled (bare for kernel, 'op.'-prefixed for op).
    from aotriton.tune.registry import make_tune_desc

    entry_kwargs = dict(dtype='float16', hdim=64, seqlen_q=128, seqlen_k=128,
                         causal=False, dropout_p=0.0, bias_type=0)

    tune = make_tune_desc('flash', modules_dir=_MODULES_DIR)

    entry = tune.ENTRY_CLASS(**entry_kwargs)
    impls = tune.list_impls(entry)

    assert 'attn_fwd' in impls          # kernel level: bare, unprefixed
    assert 'op.attn_fwd' in impls       # op level: 'op.'-prefixed DSL name
    assert 'op.attn_bwd' in impls
    # No accidental double-prefixing / stray dots beyond the single 'op.' marker.
    assert all(name.count('.') <= 1 for name in impls)

    # The two levels' 'attn_fwd' selectors are distinguished only by the
    # ImplSelector.tuning_level field / DSL prefix, never by iface_name itself.
    from aotriton.tune.tdesc import ImplSelector
    kernel_sel = ImplSelector(tuning_level='kernel', iface_name='attn_fwd', impl_index=0)
    op_sel = ImplSelector(tuning_level='op', iface_name='attn_fwd', impl_index=0)
    assert kernel_sel.iface_name == op_sel.iface_name == 'attn_fwd'
    assert kernel_sel.as_text() == 'attn_fwd=0'
    assert op_sel.as_text() == 'op.attn_fwd=0'
    assert kernel_sel != op_sel
    # list_impls()'s DSL names round-trip through the same split/dsl_name
    # primitives ImplSelector itself uses.
    assert kernel_sel.dsl_name in impls
    assert op_sel.dsl_name in impls


def test_flash_tune_get_impl_dispatches_on_prefix():
    # Revision note 3: get_impl() takes a plain DSL name (never an
    # ImplSelector) and dispatches internally on the 'op.' prefix, lazily
    # importing whichever provider module (level_kernel.py / level_op.py)
    # owns that level. Neither provider module needs torch/pyaotriton at
    # import time for its list_impls()/get_impl() top-level dispatch machinery
    # -- but the actual impl objects they build (_build_kernel_dict /
    # _build_op_dict) DO need pyaotriton, so resolving either name here is
    # expected to fail with ImportError in this no-pyaotriton environment.
    # That failure -- not a silent success -- is exactly what this test
    # checks: both prefixes route to *some* provider and fail for the
    # *expected* reason (missing pyaotriton), not because the dispatch itself
    # is broken.
    from aotriton.tune.registry import make_tune_desc

    tune = make_tune_desc('flash', modules_dir=_MODULES_DIR)
    try:
        import pyaotriton  # noqa: F401
        pytest.skip('pyaotriton is installed; dispatch-failure path not exercisable here')
    except ImportError:
        pass

    with pytest.raises(ImportError) as kernel_exc:
        tune.get_impl('attn_fwd')
    assert 'attn_fwd' in str(kernel_exc.value)

    with pytest.raises(ImportError) as op_exc:
        tune.get_impl('op.attn_fwd')
    assert 'op.attn_fwd' in str(op_exc.value)


def test_flash_tune_get_impl_unknown_prefix_raises_clearly():
    from aotriton.tune.registry import make_tune_desc

    tune = make_tune_desc('flash', modules_dir=_MODULES_DIR)
    with pytest.raises(ValueError, match='bogus_level'):
        tune.get_impl('bogus_level.attn_fwd')


def test_flash_tune_get_impl_unknown_name_raises_clearly():
    from aotriton.tune.registry import make_tune_desc

    tune = make_tune_desc('flash', modules_dir=_MODULES_DIR)
    try:
        import pyaotriton  # noqa: F401
    except ImportError:
        pytest.skip('needs pyaotriton to reach the per-name KeyError past the library import')
    with pytest.raises(KeyError):
        tune.get_impl('not_a_real_iface')


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


# The probe has to run in a child interpreter. `import torch` is a no-op once
# torch is in sys.modules, so "did this module import torch" is only answerable
# in a process that has never imported it -- and the whole suite shares one
# process, in which test_gpu_utils_amdsmi.py imports aotriton.tune.gpu_utils
# (which legitimately needs torch) long before this file is collected.
# Asserting on the parent's sys.modules made the result depend on collection
# order, and skipping the precondition instead would defeat the check outright.
_TORCH_PROBE = r"""
import importlib, json, sys
sys.path[:0] = [p for p in json.loads(sys.argv[1]) if p not in sys.path]
out = {}
for name in json.loads(sys.argv[2]):
    had_torch = 'torch' in sys.modules
    try:
        importlib.import_module(name)
    except ImportError as e:
        dep = (getattr(e, 'name', None) or '').split('.')[0]
        out[name] = 'missing:' + dep if dep in ('psycopg', 'dacite') else 'error:' + repr(e)
        continue
    out[name] = 'ok' if had_torch or 'torch' not in sys.modules else 'torch'
print(json.dumps(out))
"""


@pytest.fixture(scope='session')
def torch_free_probe():
    """module name -> 'ok' | 'torch' | 'missing:<dep>' | 'error:<repr>'.

    One child imports every module in turn, attributing torch to whichever
    import first pulled it in. For the question being asked that is as strong
    as one child per module -- if no module imports torch, no sequential order
    of them does either -- and costs one interpreter start rather than ~25.
    """
    proc = subprocess.run(
        [sys.executable, '-c', _TORCH_PROBE,
         json.dumps(sys.path), json.dumps(_PQ_LOCALQ_MODULES)],
        capture_output=True, text=True)
    assert proc.returncode == 0, (
        f'torch-free probe crashed (rc={proc.returncode}); the traceback names '
        f'the module that raised:\n{proc.stdout}\n{proc.stderr}')
    return json.loads(proc.stdout)


@pytest.mark.parametrize('module_name', _PQ_LOCALQ_MODULES)
def test_pq_localq_module_imports_without_torch(module_name, torch_free_probe):
    status = torch_free_probe[module_name]
    if status.startswith('missing:'):
        # Optional tuning-only deps (psycopg, dacite) may genuinely be absent
        # in this environment -- see the module docstring.
        pytest.skip(f'{module_name}: optional dependency '
                    f'{status.split(":", 1)[1]!r} not installed')
    assert status == 'ok', (
        f'{module_name} imported torch at import time, directly or through '
        f'one of its imports -- the probe names the first module to pull it '
        f'in, which may not be the one holding the offending import'
        if status == 'torch' else f'{module_name}: {status}')


def test_pq_localq_module_list_is_not_empty():
    # Guards against the discovery glob above silently matching nothing (e.g.
    # if python/tune/pq or python/tune/localq were ever renamed/moved again).
    assert len(_PQ_LOCALQ_MODULES) >= 20
    assert 'aotriton.tune.pq.queue' in _PQ_LOCALQ_MODULES
    assert 'aotriton.tune.localq.broker' in _PQ_LOCALQ_MODULES
    assert 'aotriton.tune.pq.vis_descriptors' in _PQ_LOCALQ_MODULES
    # Per-family descriptors used to live here as
    # aotriton.tune.pq.vis_descriptors.<family> (e.g. `...flash`); they now
    # live at modules/<family>/visperf/__init__.py, loaded by path via
    # aotriton.tune.registry.load_family_visperf (modular-tune.md §3d.1), so
    # this glob-based discovery no longer sees them -- vis_descriptors/
    # holds only the registry shim now.


# --- (d) Phase 3 (modular-tune.md §3d/§5): visperf relocation + registry- ---
#     driven discovery. Covers load_family_visperf, the /family_static
#     Flask route's path-traversal guard, and export_visperf's multi-family
#     JS concatenation.

def test_load_family_visperf_flash_descriptor_dims_match_js():
    # DESCRIPTOR['dims'] (consumed by aotriton.tune.pq.visperf._build_query)
    # and modules/flash/visperf/static/flash.js's FLASH_DESCRIPTOR.dims /
    # matrixAxes must agree on dimension names, or the webui's column
    # picker and the SQL it queries would silently disagree.
    import re
    from aotriton.tune.registry import load_family_visperf

    mod = load_family_visperf('flash', modules_dir=_MODULES_DIR)
    desc = mod.DESCRIPTOR
    assert desc['id'] == 'flash'
    assert desc['kernel_table'] == 'best_tuning_results'
    assert desc['name_col'] == 'iface_name'

    dim_aliases = {alias for _expr, alias in desc['dims']}
    expected = {'dtype', 'hdim', 'seqlen_q', 'seqlen_k', 'causal', 'bias_type', 'dropout'}
    assert expected <= dim_aliases
    assert set(desc['matrix_axes']) <= dim_aliases

    js_path = _MODULES_DIR / 'flash' / 'visperf' / 'static' / 'flash.js'
    js_text = js_path.read_text(encoding='utf-8')
    js_dim_keys = set(re.findall(r"key:\s*'(\w+)'", js_text))
    m = re.search(r"matrixAxes:\s*\{\s*row:\s*'(\w+)',\s*col:\s*'(\w+)'\s*\}", js_text)
    assert m, "flash.js: matrixAxes not found in expected {row: '...', col: '...'} form"
    js_matrix_axes = {m.group(1), m.group(2)}

    # JS declares one 'dims' entry per non-matrix axis; matrixAxes covers the
    # other two. Together they must equal the Python side's dim aliases.
    assert js_dim_keys | js_matrix_axes == dim_aliases
    assert js_matrix_axes == set(desc['matrix_axes'])


def test_family_static_route_guards_traversal_and_unknown_family():
    # modular-tune.md §3d.3: the registry (tasks.DESCRIPTORS) is the
    # path-traversal whitelist for the <family> URL segment; werkzeug's own
    # safe_join (inside send_from_directory) covers <filename>.
    pytest.importorskip('flask')
    pytest.importorskip('psycopg')
    from werkzeug.exceptions import NotFound

    tune_dir = str(_REPO_ROOT / '.tune')
    inserted = tune_dir not in sys.path
    if inserted:
        sys.path.insert(0, tune_dir)
    try:
        from webui import routes
    finally:
        if inserted:
            sys.path.remove(tune_dir)

    from flask import Flask
    app = Flask(__name__)
    app.register_blueprint(routes.bp)

    with app.test_request_context():
        # Legitimate family + file: served successfully. send_from_directory
        # streams the file, so the response is in direct-passthrough mode and
        # get_data() refuses to materialise it; read the body off the iterable
        # instead of forcing a sequence conversion.
        resp = routes.family_static('flash', 'flash.js')
        assert resp.status_code == 200
        body = b''.join(resp.response)
        resp.close()
        assert b'registerDescriptor' in body

        # Unregistered family: 404s via the registry whitelist, before ever
        # touching the filesystem (e.g. '..' is never a registered family).
        with pytest.raises(NotFound):
            routes.family_static('not_a_real_family', 'flash.js')

        # Traversal attempt against an otherwise-*valid* family: safe_join
        # inside send_from_directory must still reject it.
        with pytest.raises(NotFound):
            routes.family_static('flash', '../../../../../../etc/passwd')


def test_build_export_html_concatenates_every_family_js_once(tmp_path):
    # modular-tune.md §3d.4: export_visperf's placeholder becomes
    # __FAMILY_JS__, concatenating every registered family's JS exactly
    # once. perf.js and the template are aotriton.tune.pq package data (F12
    # is closed by packaging them, not by an explicit repo_root); only the
    # per-family JS under modules/ still needs an explicit modules_dir.
    pytest.importorskip('psycopg')
    from aotriton.tune.pq.export_visperf import build_export_html
    from aotriton.tune.registry import available_module_names

    modules_dir = tmp_path / 'modules'

    families = available_module_names()
    assert families, 'expected at least one registered family (flash)'
    for family in families:
        static_dir = modules_dir / family / 'visperf' / 'static'
        static_dir.mkdir(parents=True)
        (static_dir / f'{family}.js').write_text(
            f'/* {family.upper()}_JS_MARKER */', encoding='utf-8')

    html = build_export_html({}, modules_dir=modules_dir)

    # perf.js comes from the package, not the fixture -- confirm the real
    # engine got inlined (it's ~1500 lines; a distinctive top-of-file symbol
    # is enough to prove it's not empty/placeholder text).
    assert 'registerDescriptor' in html

    for family in families:
        marker = f'/* {family.upper()}_JS_MARKER */'
        assert html.count(marker) == 1, (
            f'{family}: expected exactly one occurrence of {marker!r}')

    # No leftover placeholder tokens of any kind. `// __FAMILY_JS__` /
    # `// __PERF_JS__` are matched as whole lines -- the exact form of
    # build_export_html's substitution key -- rather than as a bare
    # substring: the real, packaged perf.js legitimately mentions the bare
    # name '__FAMILY_JS__' inside a prose comment describing this very
    # mechanism, which a substring check would misflag as a leftover
    # placeholder.
    import re
    for placeholder_line in ('// __FAMILY_JS__', '// __PERF_JS__'):
        pattern = re.compile(r'^\s*' + re.escape(placeholder_line) + r'\s*$', re.MULTILINE)
        assert not pattern.search(html), f'leftover placeholder line {placeholder_line!r} in output'
    for placeholder in ('__PERF_DATA__', '__INITIAL_PARAMS__', '__PLOTLY_CDN__'):
        assert placeholder not in html, f'leftover placeholder {placeholder!r} in output'


def test_build_export_html_fails_loudly_on_missing_family_js(tmp_path):
    # F12: a missing asset must raise, not silently emit a JS-less page.
    # perf.js/the template are package data and always present in this
    # checkout's editable install; what's still a caller-supplied root is
    # modules_dir, so an empty one (no family JS at all) is what exercises
    # the loud-failure path here.
    pytest.importorskip('psycopg')
    from aotriton.tune.pq.export_visperf import build_export_html

    modules_dir = tmp_path / 'modules'  # deliberately empty: no family JS at all
    modules_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        build_export_html({}, modules_dir=modules_dir)


def main():
    raise SystemExit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
    main()


# ---------------------------------------------------------------------------
# pytest node-ID parsing: family-neutral split vs family-specific translation
# ---------------------------------------------------------------------------

def test_parse_node_id_is_family_neutral():
    from aotriton.tune.pytest_node import parse_node_id
    node = parse_node_id(
        'modules/flash/tests/test_backward.py::test_regular_bwd'
        '[Split-False-l1-dtype2-0.0-CausalOff-256-8192-hdim8-5-3]')
    assert node.path == 'modules/flash/tests/test_backward.py'
    assert node.test == 'test_regular_bwd'
    assert len(node.params) == 11 and node.params[3] == 'dtype2'
    # The family comes from the path, so the splitter never names one itself.
    assert node.family == 'flash'
    assert parse_node_id('tests/t.py::test_x[a-b]').family is None


def test_parse_node_id_rejects_malformed():
    from aotriton.tune.pytest_node import parse_node_id
    for bad in ('no brackets', 'p.py::test_x', 'p.py[a-b]'):
        with pytest.raises(ValueError):
            parse_node_id(bad)


def test_flash_entry_from_pytest_node():
    from aotriton.tune.pytest_node import parse_node_id
    from aotriton.tune.registry import load_family_tune
    pe = load_family_tune('flash', modules_dir=_MODULES_DIR).pytest_entry

    # Shapes round UP to the nearest tuning-database axis: 8 -> 16, 8192 stays.
    e = pe.entry_from_pytest_node(parse_node_id(
        'modules/flash/tests/test_backward.py::test_regular_bwd'
        '[Split-False-l1-dtype2-0.0-CausalOff-256-8192-hdim8-5-3]'))
    assert e == {'dtype': 'float32', 'hdim': 16, 'seqlen_q': 8192,
                 'seqlen_k': 256, 'causal': False, 'dropout_p': 0.0,
                 'bias_type': 0}

    # _common_test.fmt_hdim renders a tuple as 'hdim(a,b)', not 'hdimAxB'.
    e = pe.entry_from_pytest_node(parse_node_id(
        'modules/flash/tests/test_backward.py::test_regular_bwd'
        '[Split-False-l1-dtype0-0.0-CausalOff-256-256-hdim(64,128)-8-2]'))
    assert e['hdim'] == (64, 128)

    # test_op_bwd_with_matrix_bias has no causal parameter and fixes bias_type.
    e = pe.entry_from_pytest_node(parse_node_id(
        'modules/flash/tests/test_backward.py::test_op_bwd_with_matrix_bias'
        '[Split-False-l1-dtype0-0.0-129-257-hdim48-3-1]'))
    assert e['causal'] is False and e['bias_type'] == 1

    with pytest.raises(ValueError):
        pe.entry_from_pytest_node(parse_node_id('modules/flash/tests/t.py::test_nope[a-b]'))
    with pytest.raises(ValueError):   # too few params for the layout
        pe.entry_from_pytest_node(parse_node_id(
            'modules/flash/tests/test_backward.py::test_regular_bwd[a-b-c]'))

    # test_irregulars exists in both test files with different parameter
    # positions (the forward form has no leading BWDOP), so the layout is
    # keyed by file too. An unknown file must raise, never fall back to the
    # backward layout and resolve a real but wrong entry.
    with pytest.raises(ValueError):
        pe.entry_from_pytest_node(parse_node_id(
            'modules/flash/tests/test_forward.py::test_irregulars'
            '[BiasOn-False-l1-dtype2-0.5-CausalOff-300-900-hdim100-4-2]'))


def test_entry_filter_is_the_only_clause_builder():
    # CLAUDE.md: pq owns database access. Four copies of this clause builder
    # had drifted across .tune/; three of them mishandled a tuple value.
    from aotriton.tune.pq.queue import entry_filter
    entry = {'dtype': 'float16', 'hdim': 64, 'causal': True, 'dropout_p': 0.0}

    sql, params = entry_filter(entry)
    # bool must be tested before int -- bool is a subclass of int in Python,
    # so a reordering silently casts causal to ::integer.
    assert "(task_config->'entry'->>'causal')::boolean = %s" in sql
    assert "(task_config->'entry'->>'hdim')::integer = %s" in sql
    assert "(task_config->'entry'->>'dropout_p')::float = %s" in sql
    assert "task_config->'entry'->>'dtype' = %s" in sql
    assert params == ['float16', 64, True, 0.0]

    # A composite value is compared as a JSON array via -> , not ->>.
    sql, params = entry_filter({'hdim': (64, 128)})
    assert sql == "task_config->'entry'->'hdim' = %s::jsonb"
    assert params == ['[64, 128]']

    # Optional row filters lead, in a fixed order.
    sql, params = entry_filter({'hdim': 64}, arch='gfx942',
                               tuning_level='op', module='flash')
    assert sql.startswith("task_config->>'arch' = %s AND tuning_level = %s "
                          "AND module = %s AND ")
    assert params[:3] == ['gfx942', 'op', 'flash']


def test_no_raw_tuning_db_sql_outside_pq():
    # Regression guard for the CLAUDE.md rule: .tune/ must reach the tuning
    # database through aotriton.tune.pq, never with its own SQL.
    import re
    # .tune/bin/psql is an interactive psql wrapper; SQL in its help text is
    # the point of the tool, not a bypass of the pq layer.
    EXEMPT = {'.tune/bin/psql'}
    offenders = []
    pat = re.compile(r"FROM (task_queue|tuning_results|best_tuning_results"
                     r"|most_accurate_tuning_results)\b|task_config->'entry'")
    for sub in ('webui', 'libexec', 'bin', 'remote'):
        d = _REPO_ROOT / '.tune' / sub
        if not d.is_dir():
            continue
        for f in d.rglob('*'):
            if not f.is_file() or f.suffix in {'.html', '.js', '.css'}:
                continue
            if str(f.relative_to(_REPO_ROOT)) in EXEMPT:
                continue
            try:
                text = f.read_text(encoding='utf-8', errors='ignore')
            except OSError:
                continue
            for n, line in enumerate(text.splitlines(), 1):
                if pat.search(line) and not line.lstrip().startswith(('#', '--', '*')):
                    offenders.append(f'{f.relative_to(_REPO_ROOT)}:{n}')
    assert not offenders, 'raw tuning-DB SQL outside pq: ' + ', '.join(offenders)


# ---------------------------------------------------------------------------
# visperf: the Python and JavaScript halves of a family descriptor must agree
# ---------------------------------------------------------------------------

def _js_list(src: str, key: str, *, where: str) -> list[str]:
    """Extract `key: ['a', 'b']` (or `new Set([...])`) from a JS object.

    Deliberately strict: a miss raises rather than returning empty, so a
    reshaped descriptor fails this test instead of passing it vacuously.
    """
    import re
    m = re.search(rf'\b{key}:\s*(?:new Set\()?\[([^\]]*)\]', src)
    assert m, f'{where}: could not find `{key}: [...]`'
    return [t.strip().strip('\'"') for t in m.group(1).split(',') if t.strip()]


def _js_scalar(src: str, key: str, *, where: str) -> str:
    import re
    m = re.search(rf"\b{key}:\s*'([^']*)'", src)
    assert m, f'{where}: could not find `{key}: <string>`'
    return m.group(1)


def _js_dim_keys(src: str, *, where: str) -> list[str]:
    """The `key:` of each entry in the JS `dims: [ {...}, ... ]` array."""
    import re
    m = re.search(r'\bdims:\s*\[(.*?)\n  \]', src, re.S)
    assert m, f'{where}: could not find the `dims: [...]` array'
    keys = re.findall(r"\bkey:\s*'([^']+)'", m.group(1))
    assert keys, f'{where}: `dims` array contained no `key:` entries'
    return keys


def _visperf_families():
    """(family, DESCRIPTOR, js_source) for every family that has a visperf block."""
    from aotriton.tune.registry import available_module_names, load_family_visperf
    out = []
    for family in available_module_names():
        js = _MODULES_DIR / family / 'visperf' / 'static' / f'{family}.js'
        try:
            desc = load_family_visperf(family, modules_dir=_MODULES_DIR).DESCRIPTOR
        except Exception:
            assert not js.is_file(), (
                f'{family}: has {js} but no loadable visperf DESCRIPTOR')
            continue
        assert js.is_file(), f'{family}: has a DESCRIPTOR but no {js}'
        out.append((family, desc, js.read_text(encoding='utf-8')))
    return out


def test_visperf_descriptors_agree_across_languages():
    """Every registered family's Python and JS descriptors must line up.

    They are consumed together -- Python builds the query, JS labels the
    dropdown and reads the result -- so a name present in one and not the
    other produces an empty chart with no error anywhere. That is exactly how
    op-mode perf pages broke once: the JS still asked for '<name>_op' after
    storage moved to bare iface_names.
    """
    families = _visperf_families()
    assert families, 'no family exposes a visperf descriptor; test is vacuous'

    for family, desc, js in families:
        where = f'{family}.js'
        assert _js_scalar(js, 'id', where=where) == desc['id']
        assert _js_scalar(js, 'label', where=where) == desc['label']

        # The names the UI offers must be the names the query filters on.
        assert _js_list(js, 'kernels', where=where) == list(desc['kernels'])
        assert _js_list(js, 'opsList', where=where) == list(desc['ops'])
        # `ops` is a Set used for membership; same members as opsList.
        assert set(_js_list(js, 'ops', where=where)) == set(desc['ops'])

        # No '<name>_op'-style spelling may survive on either side: the level
        # rides in the DSL prefix, never in the interface name.
        for name in list(desc['kernels']) + list(desc['ops']):
            assert not name.endswith('_op'), f'{family}: {name!r} is not a bare iface_name'

        # Matrix axes and the remaining dims together cover the Python dims.
        row = _js_scalar(js, 'row', where=where)
        col = _js_scalar(js, 'col', where=where)
        assert (row, col) == tuple(desc['matrix_axes'])
        py_aliases = {alias for _, alias in desc['dims']}
        assert set(_js_dim_keys(js, where=where)) | {row, col} == py_aliases, (
            f'{family}: JS dims + matrixAxes do not cover the Python dims aliases')


def test_export_assets_resolve_per_tree(tmp_path):
    """The three trees have three lifetimes, so three resolution rules.

    python/ is installed and its template is package data; .tune/ and modules/
    are checkout-only and must be named by the caller. An installed package
    cannot infer where a checkout is, and pretending otherwise is what once
    produced a silently JS-less export.
    """
    pytest.importorskip('psycopg')
    from aotriton.tune.pq import export_visperf as ev

    # The template is package data -- next to the module, in the wheel.
    assert ev._TEMPLATE.is_file()
    assert ev._TEMPLATE.parent == Path(ev.__file__).parent

    # perf.js is NOT: it belongs to .tune/, which is never installed.
    assert not (Path(ev.__file__).parent / 'static' / 'perf.js').exists(), (
        'perf.js is back inside the package; it belongs to .tune/')
    assert (_REPO_ROOT / '.tune' / ev._PERF_JS_RELPATH).is_file()

    # An explicit tune_root is honoured, and a missing perf.js raises rather
    # than producing an export with no renderer in it.
    fake_tune = tmp_path / '.tune'
    (fake_tune / 'webui' / 'static' / 'js').mkdir(parents=True)
    (fake_tune / 'webui' / 'static' / 'js' / 'perf.js').write_text('/*STUB*/')
    fake_modules = tmp_path / 'modules'
    for family in ['flash']:
        d = fake_modules / family / 'visperf' / 'static'
        d.mkdir(parents=True)
        (d / f'{family}.js').write_text('/*FAM*/')
    html = ev.build_export_html({}, tune_root=fake_tune, modules_dir=fake_modules)
    assert '/*STUB*/' in html and '/*FAM*/' in html

    with pytest.raises(FileNotFoundError):
        ev.build_export_html({}, tune_root=tmp_path / 'nope',
                             modules_dir=fake_modules)


def test_default_tune_root_matches_the_checkout():
    from aotriton.tune.registry import default_tune_root
    assert (default_tune_root() / 'webui' / 'static' / 'js' / 'perf.js').is_file()


def test_export_inlines_perf_js_before_family_js(tmp_path):
    """perf.js defines registerDescriptor(); the family scripts call it as
    they load. If the template emits them the other way round, a standalone
    export throws ReferenceError while parsing the first family script and
    renders nothing -- with the strings all present, so a substring check
    passes anyway. Assert the order.
    """
    pytest.importorskip('psycopg')
    from aotriton.tune.pq.export_visperf import build_export_html

    tune = tmp_path / '.tune'
    (tune / 'webui' / 'static' / 'js').mkdir(parents=True)
    (tune / 'webui' / 'static' / 'js' / 'perf.js').write_text(
        'function registerDescriptor(d){} /*ENGINE*/')
    mods = tmp_path / 'modules'
    d = mods / 'flash' / 'visperf' / 'static'
    d.mkdir(parents=True)
    (d / 'flash.js').write_text('registerDescriptor({id:"flash"}); /*FAMILY*/')

    html = build_export_html({}, tune_root=tune, modules_dir=mods)
    assert html.index('/*ENGINE*/') < html.index('/*FAMILY*/'), (
        'family descriptor JS is inlined before the engine that defines '
        'registerDescriptor()')
