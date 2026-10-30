# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The Perfmon compare page's data (perfmon-after-def-rev0.md section 8.6).

Everything numeric is aotriton.perfmon.perfdb's: reading the committed
`modules/<family>/perfdb/` trees, joining two sides (compare_files), the
geometric mean and the number format. Live measurements are read through
aotriton.perfmon.pq.iter_latest; this module has no SQL.

A side is "tag:<release tag>" (an exported perfdb directory) or
"live:<subject>" (the newest measurements in PostgreSQL, e.g. HEAD, which is
never exported).
"""

import logging
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from aotriton.perfmon import perfdb
from aotriton.perfmon import pq as perfmon_pq
from aotriton.perfmon.export import row_line
from aotriton.perfmon.registry import default_perfdb_dir, make_perf_desc
from aotriton.pgdef.utils import default_modules_dir
from aotriton.pgdef.workdir import get_db_connection_params

logger = logging.getLogger(__name__)

DEFAULT_FAMILY = 'flash'
DEFAULT_TOP = 20
#: Per-line ratios shown on the page; the CSV download has them all.
MAX_SHOWN_LINES = 500


class CompareError(ValueError):
    """A bad selection; the message is shown to the user."""


def list_families() -> list[str]:
    """Families that have a perfdb directory (DEFAULT_FAMILY when none)."""
    modules = Path(default_modules_dir())
    found = sorted(p.parent.name for p in modules.glob('*/perfdb') if p.is_dir())
    return found or [DEFAULT_FAMILY]


def perfdb_root(family: str) -> Path:
    if family not in list_families():
        raise CompareError(f'no perfdb for family {family!r}')
    return default_perfdb_dir(family)


def parse_side(text: str) -> tuple[str, str]:
    """('tag', '0.13b') or ('live', 'aotriton-head+rocm10.1.0')."""
    origin, sep, name = (text or '').partition(':')
    if not sep or origin not in ('tag', 'live') or not name:
        raise CompareError(f'bad side {text!r}: expected tag:<tag> or live:<subject>')
    return origin, name


def list_live_subjects(workdir, family: str) -> list[dict]:
    """Selections with measurements in PostgreSQL (aotriton.perfmon.pq.list_selections);
    empty when the database is unreachable."""
    try:
        with psycopg.connect(**get_db_connection_params(Path(workdir)), row_factory=dict_row) as conn:
            return perfmon_pq.list_selections(conn, module=family)
    except Exception as e:
        logger.warning('perfmon compare: no live subjects: %s', e)
        return []


def selector_data(workdir, family: str) -> dict:
    """What the compare form offers: tags with their arches and variants, and
    the live subjects."""
    root = perfdb_root(family)
    tags = perfdb.list_tags(root)
    arches = {tag: perfdb.list_arches(root, tag) for tag in tags}
    variants = {tag: {arch: sorted(perfdb.variants_present(root, tag, arch)) for arch in arches[tag]}
                for tag in tags}
    return {'tags': tags, 'arches': arches, 'variants': variants,
            'live': list_live_subjects(workdir, family)}


def _live_files(workdir, family: str, desc, entries, arch: str, subject: str) -> dict:
    """The live measurements of one subject as perfdb files, entries joined by
    ordinal. Entries that are not in the entry list have no ordinal and are
    skipped: they cannot be joined with a committed tag."""
    files: dict[str, perfdb.PerfdbFile] = {}
    with psycopg.connect(**get_db_connection_params(Path(workdir)), row_factory=dict_row) as conn:
        for row in perfmon_pq.iter_latest(conn, arch=arch, subjects=[subject], module=family):
            ordinal = entries.ordinal_of(desc, desc.entry_from_dict(row['entry']), add=False)
            if ordinal is None:
                continue
            name = perfdb.file_name(row['iface'], row['backend'])
            f = files.get(name)
            if f is None:
                f = files[name] = perfdb.new_file(row['iface'], row['backend'])
            try:
                f.set_measurement_keys({'timing_method': row.get('timing_method'),
                                        'iterations': row.get('iterations'),
                                        'l2_flush': row.get('l2_flush'),
                                        'unit': perfdb.UNIT},
                                       where=f'{subject}/{arch}/{name}')
            except perfdb.PerfdbError:
                continue
            f.merge(row['variant'], {ordinal: row_line(row)})
    return files


def _load_side(workdir, family, root, desc, entries, side: tuple[str, str], arch: str) -> dict:
    origin, name = side
    if origin == 'tag':
        return perfdb.load_arch(root, name, arch)
    return _live_files(workdir, family, desc, entries, arch, name)


def _label(entries, ordinal: int) -> str:
    try:
        problem, sizes = entries.rows_of(ordinal)
    except IndexError:
        return f'entry {ordinal}'
    pairs = list(zip(entries.problem_columns, problem)) + list(zip(entries.size_columns, sizes))
    return ' '.join(f'{c}={v}' for c, v in pairs)


def compare(workdir, family: str, side_a: str, side_b: str, arch: str,
            variant: str | None = None, top: int = DEFAULT_TOP) -> dict:
    """Compare two sides of one arch (A old, B new; ratio is A/B, above 1 means
    B is faster).

    Returns {'lines': [...], 'files': [...], 'geomean', 'worst': [...],
    'refused': [...]}: lines are dicts (file, variant, entry, label, median_a,
    median_b, ratio) in the order perfdb.compare_files gives them, files are
    {file, lines, geomean}, worst the `top` lowest ratios.
    """
    a, b = parse_side(side_a), parse_side(side_b)
    if not arch:
        raise CompareError('choose an arch')
    root = perfdb_root(family)
    desc = make_perf_desc(family)
    entries = perfdb.EntryList.load(root, desc.PROBLEM_COLUMNS, desc.SIZE_COLUMNS)
    files_a = _load_side(workdir, family, root, desc, entries, a, arch)
    files_b = _load_side(workdir, family, root, desc, entries, b, arch)
    ratios, refused = perfdb.compare_files(files_a, files_b, variant=variant or None)
    lines = [{'file': name, 'variant': v, 'entry': entry, 'label': _label(entries, entry),
              'median_a': ma, 'median_b': mb, 'ratio': r}
             for name, v, entry, ma, mb, r in ratios]
    per_file = []
    for name in sorted({line['file'] for line in lines}):
        mine = [line['ratio'] for line in lines if line['file'] == name]
        per_file.append({'file': name, 'lines': len(mine), 'geomean': perfdb.geomean(mine)})
    return {'arch': arch,
            'lines': lines,
            'files': per_file,
            'geomean': perfdb.geomean(line['ratio'] for line in lines),
            'worst': sorted(lines, key=lambda line: line['ratio'])[:max(top, 0)],
            'refused': refused,
            'only_a': sorted(set(files_a) - set(files_b)),
            'only_b': sorted(set(files_b) - set(files_a))}


def lines_csv(result: dict) -> str:
    """Every per-line ratio of a compare() result, as CSV text."""
    out = ['arch,file,variant,entry,median_a,median_b,a/b']
    for line in result['lines']:
        out.append(','.join([result['arch'], line['file'], line['variant'], str(line['entry']),
                             perfdb.fmt4(line['median_a']), perfdb.fmt4(line['median_b']),
                             perfdb.fmt4(line['ratio'])]))
    return '\n'.join(out) + '\n'


fmt4 = perfdb.fmt4
