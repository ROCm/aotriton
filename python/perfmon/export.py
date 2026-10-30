# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Export: PostgreSQL -> the repository's perfdb (`modules/<family>/perfdb/`).

    python -m aotriton.perfmon.export --workdir <workdir> --module flash \\
        --arch gfx942 [--variant <name>] [--subject ...] [--output modules/flash/perfdb] \\
        [--entry_set prime llm] [--verify]
    python -m aotriton.perfmon.export --workdir <workdir> -l [--module M] [--arch A] \\
        [--variant V] [--subject S ...]

The core is `export_rows()`, which takes the newest measurement of every point
(as `aotriton.perfmon.pq.iter_latest` yields them) and merges them into the
tree; it touches no database. The command line fetches the rows, calls it,
renders the markdown and, with `--verify`, reads the files back.

Merge rules: a line in the database replaces the file's line for the same
entry; lines not in the database are kept, so exporting a partial campaign
never erases data. A `head` subject is refused (HEAD is compared live, never
committed). Two subjects of one release tag (two ROCm versions) cannot share
`<tag>/`, which has no ROCm level: that is an error.
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from aotriton.pgdef import paths

from . import perfdb
from .perfdb import PerfdbError

APP = 'perfmon'
HEAD_TAG = 'head'
GIB = 1 << 30

#: Statistic columns of a database row, in perfdb.STAT_COLUMNS order.
_STAT_FIELDS = tuple(f'{c}_us' for c in perfdb.STAT_COLUMNS)


@dataclass
class ExportResult:
    """What an export did, for the summary line."""
    written: int = 0            # lines written from database rows
    changed: int = 0            # lines that differ from the file's before
    no_data: int = 0            # NotOK, crash, or an OK row without statistics
    excluded: int = 0           # lines marked excluded (database or arch rule)
    files: list = field(default_factory=list)       # paths of data files that changed
    tags: list = field(default_factory=list)
    appended: dict = field(default_factory=dict)    # {'problems': n, 'entries': n}
    warnings: list = field(default_factory=list)
    skipped: int = 0            # rows left out: their timing method differs from the file's


def _day(created_at) -> str:
    if created_at is None:
        return ''
    if isinstance(created_at, str):
        return created_at[:10]
    return created_at.date().isoformat()


def _gate_text(value) -> str:
    if isinstance(value, bool):
        return 'on' if value else 'off'
    return '' if value is None else str(value)


def _device_id_text(value) -> str | None:
    """The runner reports the PCI device id as a number; spell it as lspci does."""
    return f'0x{value:04x}' if isinstance(value, int) and not isinstance(value, bool) else value


def _vram_gib(platform: dict) -> str:
    vram = platform.get('vram_total_bytes')
    return '' if not vram else perfdb.fmt4(round(int(vram) / GIB, 1))


def platform_config(row: dict, rocm: str) -> dict:
    """The `platform.csv` columns (but `measured`) of one measurement row."""
    platform = row.get('platform') or {}
    harness = row.get('harness') or {}
    # TODO(perfmon): the power caps, RAM speed and amdgpu version are not in the
    # report yet (they come from amd-smi and the webui's host inventory); the
    # columns stay empty until the fleet work provides them.
    return {
        'variant': row['variant'],
        'gpu_device_id': _device_id_text(platform.get('pci_device_id')),
        'vram_gib': _vram_gib(platform),
        'power_cap_w': platform.get('power_cap_w'),
        'power_cap_default_w': platform.get('power_cap_default_w'),
        'gpus_per_node': platform.get('gpus_per_node'),
        'thermal_gate': _gate_text(platform.get('thermal_gate')),
        'cpu': platform.get('cpu'),
        'ram_speed': platform.get('ram_speed'),
        'os': platform.get('os'),
        'kernel': platform.get('kernel'),
        'amdgpu': platform.get('amdgpu'),
        'rocm': rocm,
        'harness': harness.get('commit') if isinstance(harness, dict) else harness,
    }


def row_line(row: dict):
    """The perfdb line of a database row: the six statistics, or a marker."""
    status = row['status']
    if status == 'excluded':
        return perfdb.EXCLUDED
    if status == 'OK':
        values = [row.get(f) for f in _STAT_FIELDS]
        if all(v is not None for v in values):
            return tuple(values)
    return perfdb.NO_DATA


def group_by_tag(rows: list[dict]) -> dict[str, list[dict]]:
    """{tag: rows}. Refuses `head`, and a tag measured with two ROCm versions
    (the tag's directory has no ROCm level)."""
    by_tag: dict[str, list[dict]] = {}
    rocm_of: dict[str, tuple[str, str]] = {}
    for row in rows:
        tag, rocm = perfdb.subject_tag(row['subject'])
        if tag == HEAD_TAG:
            raise PerfdbError(f'{row["subject"]}: head is never committed (it is compared live); '
                              'select release subjects with --subject')
        if rocm_of.setdefault(tag, (rocm, row['subject']))[0] != rocm:
            raise PerfdbError(f'tag {tag} has subjects of two ROCm versions: {rocm_of[tag][1]} and '
                              f'{row["subject"]}; export one of them with --subject')
        by_tag.setdefault(tag, []).append(row)
    return by_tag


def export_rows(root: Path, desc, arch: str, rows, excluded_entries=()) -> ExportResult:
    """Merge `rows` (newest measurement per point, dicts with the columns of
    perfmon.latest_measurement) into the tree at `root`.

    `excluded_entries` are (entry, ifaces) pairs the arch cannot run (ifaces
    None: every iface); each is written as excluded to every data file of the
    (tag, variant) groups present in `rows` that has no line for it yet. They
    were never dispatched, so the database has no row for them.
    """
    root = Path(root)
    out = ExportResult()
    rows = list(rows)
    by_tag = group_by_tag(rows)
    entries = perfdb.EntryList.load(root, desc.PROBLEM_COLUMNS, desc.SIZE_COLUMNS)
    excluded = list(excluded_entries)
    for tag in sorted(by_tag, key=perfdb.tag_sort_key):
        tag_rows = by_tag[tag]
        rocm = perfdb.subject_tag(tag_rows[0]['subject'])[1]
        files = perfdb.load_arch(root, tag, arch)
        touched: set[str] = set()
        seen: dict[tuple, set] = {}     # (file name, variant) -> entry ordinals with a database row
        configs = []
        skipped_why: dict[str, str] = {}    # file -> first mismatch message
        for row in tag_rows:
            name = perfdb.file_name(row['iface'], row['backend'])
            f = files.get(name)
            if f is None:
                f = files[name] = perfdb.new_file(row['iface'], row['backend'])
            try:
                f.set_measurement_keys({'timing_method': row.get('timing_method'),
                                        'iterations': row.get('iterations'),
                                        'l2_flush': row.get('l2_flush'),
                                        'unit': perfdb.UNIT},
                                       where=f'{tag}/{arch}/{name}')
            except PerfdbError as e:
                # A file keeps one method: leave this row out, do not abort
                # the export of the others.
                out.skipped += 1
                skipped_why.setdefault(f'{tag}/{arch}/{name}', str(e))
                continue
            entry = desc.entry_from_dict(row['entry'])
            ordinal = entries.ordinal_of(desc, entry, add=True)
            line = row_line(row)
            if line == perfdb.NO_DATA:
                out.no_data += 1
            elif line == perfdb.EXCLUDED:
                out.excluded += 1
            out.written += 1
            if f.merge(row['variant'], {ordinal: line}):
                out.changed += 1
                touched.add(name)
            seen.setdefault((name, row['variant']), set()).add(ordinal)
            configs.append((platform_config(row, rocm), _day(row.get('created_at'))))

        for message in skipped_why.values():
            out.warnings.append(f'rows skipped: {message}')

        # Excluded lines, for the files and variants this export covers.
        variants = sorted({r['variant'] for r in tag_rows})
        for entry, ifaces in excluded:
            ordinal = entries.ordinal_of(desc, entry, add=True)
            for name, f in sorted(files.items()):
                if ifaces is not None and f.iface not in ifaces:
                    continue
                for variant in variants:
                    if ordinal in seen.get((name, variant), ()) or ordinal in f.blocks.get(variant, {}):
                        continue
                    f.merge(variant, {ordinal: perfdb.EXCLUDED})
                    out.excluded += 1
                    out.changed += 1
                    touched.add(name)

        d = perfdb.data_dir(root, tag, arch)
        for name in sorted(touched):
            if perfdb.write_perfdb(d / name, files[name]):
                out.files.append(d / name)
        platform_path = d / perfdb.PLATFORM_CSV
        merged = perfdb.merge_platform(perfdb.read_platform(platform_path), configs)
        d.mkdir(parents=True, exist_ok=True)
        perfdb.write_if_changed(platform_path, perfdb.platform_text(merged))
        for variant, configurations in sorted(perfdb.platform_conflicts(merged).items()):
            out.warnings.append(f'{tag}/{arch}: variant {variant!r} has machines with different '
                                f'(device id, default power cap): {configurations}')
        out.tags.append(tag)
    entries.save(root)
    out.appended = dict(entries.appended)
    if out.skipped:
        out.warnings.append(f'{out.skipped} rows skipped in total: their timing method differs from '
                            'the file\'s (a file keeps one method)')
    return out


def verify(root: Path, desc, arch: str, rows) -> list[str]:
    """Read the tree back and check every database row: returns the problems
    found (empty: all good)."""
    root = Path(root)
    entries = perfdb.EntryList.load(root, desc.PROBLEM_COLUMNS, desc.SIZE_COLUMNS)
    bad = []
    cache: dict[str, dict] = {}
    for row in rows:
        tag, _ = perfdb.subject_tag(row['subject'])
        if tag not in cache:
            cache[tag] = perfdb.load_arch(root, tag, arch)
        files = cache[tag]
        name = perfdb.file_name(row['iface'], row['backend'])
        f = files.get(name)
        method = row.get('timing_method')
        if f is not None and method is not None and f.metadata.get('timing_method') not in (None, str(method)):
            continue        # export skipped this row: a file keeps one method
        ordinal = entries.ordinal_of(desc, desc.entry_from_dict(row['entry']))
        where = f'{tag}/{arch}/{name} {row["variant"]} entry {ordinal}'
        if ordinal is None or f is None:
            bad.append(f'{where}: missing')
            continue
        expected = perfdb.canonical_line(row_line(row))
        found = f.blocks.get(row['variant'], {}).get(ordinal)
        if found != expected:
            bad.append(f'{where}: file has {found}, database {expected}')
    return bad


def excluded_for_sets(desc, arch: str, entry_sets) -> list[tuple]:
    """(entry, ifaces) of the entries of `entry_sets` that `arch` rejects,
    re-derived from the family's rule the way dispatch derives them."""
    out = []
    for name in entry_sets:
        if name == 'llm':
            for case in desc.llm_cases():
                if case.pending:
                    continue
                if not desc.validate_hw_feature(arch, case.entry)[0]:
                    out.append((case.entry, tuple(desc.phase_ifaces(case.phase))))
        else:
            for entry in desc.entries(desc.entry_sets()[name], arch):
                if not desc.validate_hw_feature(arch, entry)[0]:
                    out.append((entry, None))
    return out


def format_listing(selections: list[dict], *, given: dict, exported) -> list[str]:
    """The `-l` table. Levels (module, arch, variant, subject) that were given
    as options are not printed; `exported(selection)` says whether the
    perfdb already has a block for it."""
    levels = [lv for lv in ('module', 'arch', 'variant', 'subject') if not given.get(lv)]
    header = levels + ['measured', 'OK', 'NotOK', 'crash', 'newest', 'exported']
    table = [header]
    for s in selections:
        table.append([str(s[lv]) for lv in levels]
                     + [str(s['measured']), str(s['ok']), str(s['notok']), str(s['crash']),
                        _day(s['newest']) or '-', 'yes' if exported(s) else 'no'])
    widths = [max(len(r[i]) for r in table) for i in range(len(header))]
    numeric = {header.index(c) for c in ('measured', 'OK', 'NotOK', 'crash')}
    lines = []
    for r in table:
        cells = [c.rjust(widths[i]) if i in numeric else c.ljust(widths[i]) for i, c in enumerate(r)]
        lines.append('  '.join(cells).rstrip())
    return lines


def _is_exported(selection: dict, output: Path | None) -> bool:
    from .registry import default_perfdb_dir
    try:
        tag, _ = perfdb.subject_tag(selection['subject'])
    except PerfdbError:
        return False
    root = output or default_perfdb_dir(selection['module'])
    return selection['variant'] in perfdb.variants_present(root, tag, selection['arch'])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='python -m aotriton.perfmon.export',
                                     description='Move perfmon measurements from PostgreSQL into '
                                                 'the repository (perfdb), or list what can be exported.')
    parser.add_argument('--workdir',
                        type=Path,
                        default=None,
                        help='Working directory (config.rc); default: AOTRITON_WORKDIR')
    parser.add_argument('--module',
                        default=None,
                        help='Family (required unless -l)')
    parser.add_argument('--arch',
                        default=None,
                        help='Architecture (required unless -l)')
    parser.add_argument('--variant',
                        default=None,
                        help='Only this variant (default: every variant)')
    parser.add_argument('--subject',
                        nargs='+',
                        default=None,
                        help='Only these subjects')
    parser.add_argument('--output',
                        type=Path,
                        default=None,
                        help='perfdb directory (default: modules/<module>/perfdb)')
    parser.add_argument('--entry_set',
                        nargs='+',
                        choices=('prime', 'coverage', 'llm'),
                        default=['prime', 'llm'],
                        help='Entry sets whose arch-rejected entries are written as excluded')
    parser.add_argument('--verify',
                        action='store_true',
                        help='Read the files back and check every exported line')
    parser.add_argument('-l',
                        '--list',
                        dest='list_only',
                        action='store_true',
                        help='Write nothing; list the selections in PostgreSQL that match the given options')
    return parser


def _connect(args):
    import psycopg

    from aotriton.pgdef.pq.conn import params_from_env
    from aotriton.pgdef.workdir import load_config
    workdir = args.workdir or paths.workdir()
    if not workdir.is_dir():
        sys.exit(f'Error: working directory does not exist: {workdir}')
    load_config(workdir)
    return psycopg.connect(**params_from_env(), autocommit=True)


def _cmd_list(args) -> int:
    from . import pq
    conn = _connect(args)
    try:
        selections = pq.list_selections(conn,
                                        module=args.module,
                                        arch=args.arch,
                                        variant=args.variant,
                                        subjects=args.subject)
    finally:
        conn.close()
    if not selections:
        print('no measurements match')
        return 0
    given = {'module': args.module, 'arch': args.arch, 'variant': args.variant, 'subject': args.subject}
    for line in format_listing(selections, given=given, exported=lambda s: _is_exported(s, args.output)):
        print(line)
    return 0


def _cmd_export(args) -> int:
    from . import pq
    from .registry import default_perfdb_dir, make_perf_desc
    if not args.module or not args.arch:
        sys.exit('--module and --arch are required (or use -l to list what can be exported)')
    desc = make_perf_desc(args.module)
    root = args.output or default_perfdb_dir(args.module)
    conn = _connect(args)
    try:
        selection = dict(arch=args.arch, variant=args.variant, subjects=args.subject, module=args.module)
        rows = list(pq.iter_latest(conn, **selection))
        superseded = pq.count_superseded(conn, **selection)
    finally:
        conn.close()
    if not rows:
        sys.exit('no measurements match; see -l')
    failed = sum(1 for r in rows if r['status'] in ('NotOK', 'crash'))
    print(f'{len(rows)} measurements, {superseded} superseded by newer ones, {failed} NotOK or crash')
    try:
        result = export_rows(root, desc, args.arch, rows, excluded_for_sets(desc, args.arch, args.entry_set))
    except PerfdbError as e:
        sys.exit(f'Error: {e}')
    for warning in result.warnings:
        print(f'warning: {warning}', file=sys.stderr)
    print(f'{result.changed} lines changed in {len(result.files)} files of {", ".join(result.tags)}; '
          f'{result.appended["problems"]} problems and {result.appended["entries"]} entries appended')
    for path in perfdb.render(root, desc):
        print(f'wrote {path}')
    if args.verify:
        bad = verify(root, desc, args.arch, rows)
        for line in bad:
            print(f'verify: {line}', file=sys.stderr)
        if bad:
            return 1
        print(f'verified {len(rows)} lines')
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    return _cmd_list(args) if args.list_only else _cmd_export(args)


if __name__ == '__main__':
    sys.exit(main())
