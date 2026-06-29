# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Export a self-contained performance visualization HTML file.

The output is a single .html file with all chart data inlined as JSON and
all JS logic inlined from perf.js / vis_descriptors/flash.js. The 2-D
heatmap and level-1 drilldown work fully offline; the 3-D mesh3d view
requires Plotly.js, which is loaded from CDN (no inlined copy — Plotly is
~4.5 MB and would bloat the export). Level-2 (psel × copt) drilldown is
not included; it queries the live PostgreSQL backend.

Also packaged as a .zip download by the /api/perf/export_zip route in
.tune/webui/routes.py.

Usage:
    python -m v3python.tune.pq.export_visperf --workdir <workdir> --output perf.html
"""

import argparse
import json
import logging
import re
from pathlib import Path

import psycopg

from .visperf import query_all_best_results
from ..utils import get_db_connection_params

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# Paths to JS sources (relative to this file's parent = v3python/tune/pq/).
_HERE = Path(__file__).parent
_WEBUI_JS = _HERE.parent.parent.parent / '.tune' / 'webui' / 'static' / 'js'
_FLASH_JS = _WEBUI_JS / 'vis_descriptors' / 'flash.js'
_PERF_JS  = _WEBUI_JS / 'perf.js'
_TEMPLATE = _HERE / 'visperf_template.html'

# CDN URL with exact semver pin.
PLOTLY_CDN = (
    'https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js'
)


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


def build_export_html(data: dict, url_params: dict | None = None) -> str:
    """Build a self-contained HTML string from pre-fetched data.

    data:       {arch: {kernel: {arch, kernel, axes, rows}}}
    url_params: optional dict of URL search params to pre-set on first load
                (e.g. arch, kernel, display, scale, az_mode, col_dims, row_dims).
    """
    flash_js = _FLASH_JS.read_text(encoding='utf-8')
    perf_js  = _PERF_JS.read_text(encoding='utf-8')
    template = _TEMPLATE.read_text(encoding='utf-8')

    data = _to_column_store(data)
    substitutions = {
        '__PERF_DATA__':     _json_for_script(data),
        '__INITIAL_PARAMS__': _json_for_script(url_params or {}),
        '__PLOTLY_CDN__':    PLOTLY_CDN,
        '// __FLASH_JS__':   flash_js,
        '// __PERF_JS__':    perf_js,
    }
    # Single-pass substitution so a value that happens to contain another
    # placeholder token (e.g. inlined JS mentioning `__PERF_DATA__`) cannot
    # be re-substituted in a later step.
    pattern = re.compile('|'.join(re.escape(k) for k in substitutions))
    return pattern.sub(lambda m: substitutions[m.group(0)], template)


def export_visperf(conn, output_path: Path) -> None:
    """Generate self-contained perf.html from live database."""
    logger.info('Querying all best results (all arches, all kernels)…')
    data = query_all_best_results(conn)
    total = sum(len(kd['rows']) for ad in data.values() for kd in ad.values())
    logger.info('Fetched %d rows across %d arches', total, len(data))

    html = build_export_html(data)
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
