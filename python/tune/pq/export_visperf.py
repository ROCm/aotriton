# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Export a self-contained performance visualization HTML file.

The output is a single .html file with all chart data inlined as JSON and
all JS logic inlined from perf.js plus every registered family's
modules/<family>/visperf/static/<family>.js (modular-tune.md §3d.4). The
2-D heatmap and level-1 drilldown work fully offline; the 3-D mesh3d view
requires Plotly.js, which is loaded from CDN (no inlined copy — Plotly is
~4.5 MB and would bloat the export). Level-2 (psel × copt) drilldown is
not included; it queries the live PostgreSQL backend.

Also packaged as a .zip download by the /api/perf/export_zip route in
.tune/webui/routes.py.

Usage:
    python -m aotriton.tune.pq.export_visperf --workdir <workdir> --output perf.html
"""

import argparse
import json
import logging
import re
from pathlib import Path

import psycopg

from .visperf import query_all_best_results
from ..utils import get_db_connection_params, default_repo_root
from ..registry import default_modules_dir, available_module_names

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# visperf_template.html IS package data (setup.py's package_data covers
# *.html under aotriton.tune.pq), so it is safe to locate relative to this
# file even in a non-editable install.
_HERE = Path(__file__).parent
_TEMPLATE = _HERE / 'visperf_template.html'

# .tune/webui/static/js/perf.js and modules/<family>/visperf/static/*.js are
# NOT part of the installed `aotriton` package (webui assets and modules/
# both live outside python/); build_export_html() takes the checkout root
# and the modules root as explicit parameters (F12) rather than deriving
# them from a fixed `.parent` hop count off this file -- that fixed-depth
# scheme silently resolves to `site-packages/.tune/...` under a
# non-editable install and would emit a JS-less page with no error. The
# defaults below (default_repo_root() / default_modules_dir()) cover normal
# CLI usage; callers -- including tests -- can override either explicitly.

# CDN URL with exact semver pin.
PLOTLY_CDN = (
    'https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js'
)


def _read_required(path: Path, what: str) -> str:
    """Read a required asset, failing loudly (F12) instead of the historical
    silent-JS-less-page failure mode: a missing file must raise here, not
    quietly produce a broken export."""
    if not path.is_file():
        raise FileNotFoundError(
            f'{what} not found at {path}. export_visperf needs a source '
            f'checkout (or AOTRITON_ROOT / AOTRITON_MODULES_DIR pointing at '
            f'one) -- this asset is not shipped inside the installed '
            f'`aotriton` package.')
    return path.read_text(encoding='utf-8')


def _family_js_paths(modules_dir: Path) -> dict:
    """One `modules/<family>/visperf/static/<family>.js` path per family
    registered in `aotriton.tune.registry` (§3d.4). Self-registration
    (flash.js's own `registerDescriptor(FLASH_DESCRIPTOR)` at its bottom,
    and every other family's JS analogously) makes concatenation order
    irrelevant, so iterating `available_module_names()`'s sorted order is
    fine."""
    modules_dir = Path(modules_dir)
    return {
        family: modules_dir / family / 'visperf' / 'static' / f'{family}.js'
        for family in available_module_names()
    }


def _to_column_store(data: dict) -> dict:
    """Convert per-kernel rows from list-of-dicts to a column-store form.

    Each {arch, kernel, axes, rows: [{col: val, ...}]} becomes
         {arch, kernel, axes, cols: [...], rows: [[...]]}.
    Rehydrated client-side in visperf_template.html's fetchData override.
    """
    for arch_data in data.values():
        for kdata in arch_data.values():
            rows = kdata.get('rows') or []
            if not rows:
                kdata['cols'] = []
                kdata['rows'] = []
                continue
            cols = list(rows[0].keys())
            kdata['cols'] = cols
            kdata['rows'] = [[r.get(c) for c in cols] for r in rows]
    return data


def _json_for_script(obj) -> str:
    """json.dumps suitable for embedding directly inside <script>...</script>.

    The HTML parser ends a <script> element at the first ``</`` (followed by
    a valid tag name) — most importantly ``</script>``, but also ``<!--`` and
    ``<script`` open new parsing states. ``json.dumps`` leaves ``/`` and
    ``<`` untouched, so a string value containing ``</script>`` would break
    out of the tag.

    Replacing ``<`` with the JSON-equivalent ``\\u003c`` is the canonical
    fix: it neutralizes all three problem sequences, the JSON parser
    decodes the escape back to ``<`` transparently, and it requires no
    changes on the consuming JS side (vs. e.g. base64).
    """
    return json.dumps(obj, separators=(',', ':')).replace('<', '\\u003c')


def build_export_html(data: dict, url_params: dict | None = None, *,
                       repo_root: Path | None = None,
                       modules_dir: Path | None = None) -> str:
    """Build a self-contained HTML string from pre-fetched data.

    data:        {arch: {kernel: {arch, kernel, axes, rows}}}
    url_params:  optional dict of URL search params to pre-set on first load
                 (e.g. arch, kernel, display, scale, az_mode, col_dims, row_dims).
    repo_root:   checkout root containing `.tune/webui/static/js/perf.js`;
                 defaults to `aotriton.tune.utils.default_repo_root()`.
    modules_dir: `modules/` root containing every registered family's
                 `<family>/visperf/static/<family>.js`; defaults to
                 `aotriton.tune.registry.default_modules_dir()`.

    Both roots are explicit parameters rather than module-level globals
    (F12) so a caller -- including a test -- can point them at a throwaway
    fixture instead of whatever checkout this process happens to run from.
    """
    repo_root = Path(repo_root) if repo_root is not None else default_repo_root()
    modules_dir = Path(modules_dir) if modules_dir is not None else default_modules_dir()

    perf_js_path = repo_root / '.tune' / 'webui' / 'static' / 'js' / 'perf.js'
    perf_js = _read_required(perf_js_path, 'perf.js')
    family_js = '\n'.join(
        _read_required(path, f"visperf JS for family {family!r}")
        for family, path in _family_js_paths(modules_dir).items()
    )
    template = _read_required(_TEMPLATE, 'visperf_template.html')

    data = _to_column_store(data)
    substitutions = {
        '__PERF_DATA__':     _json_for_script(data),
        '__INITIAL_PARAMS__': _json_for_script(url_params or {}),
        '__PLOTLY_CDN__':    PLOTLY_CDN,
        '// __FAMILY_JS__':  family_js,
        '// __PERF_JS__':    perf_js,
    }
    # Single-pass substitution so a value that happens to contain another
    # placeholder token (e.g. inlined JS mentioning `__PERF_DATA__`) cannot
    # be re-substituted in a later step.
    pattern = re.compile('|'.join(re.escape(k) for k in substitutions))
    return pattern.sub(lambda m: substitutions[m.group(0)], template)


def export_visperf(conn, output_path: Path, *,
                    repo_root: Path | None = None,
                    modules_dir: Path | None = None) -> None:
    """Generate self-contained perf.html from live database."""
    logger.info('Querying all best results (all arches, all kernels)…')
    data = query_all_best_results(conn)
    total = sum(len(kd['rows']) for ad in data.values() for kd in ad.values())
    logger.info('Fetched %d rows across %d arches', total, len(data))

    html = build_export_html(data, repo_root=repo_root, modules_dir=modules_dir)
    output_path.write_text(html, encoding='utf-8')
    logger.info('Written %d bytes to %s', len(html), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--workdir', help='Project workdir containing config.rc')
    src.add_argument('--host', help='PostgreSQL host')
    parser.add_argument('--port', type=int, default=5432)
    parser.add_argument('--user')
    parser.add_argument('--password')
    parser.add_argument('--output', required=True, type=Path,
                        help='Output HTML file path (e.g. perf.html)')
    args = parser.parse_args()

    if args.workdir:
        conn_params = get_db_connection_params(Path(args.workdir))
    else:
        conn_params = {'host': args.host, 'port': args.port}
        if args.user:     conn_params['user'] = args.user
        if args.password: conn_params['password'] = args.password

    with psycopg.connect(**conn_params, autocommit=True) as conn:
        export_visperf(conn, args.output)


if __name__ == '__main__':
    main()
