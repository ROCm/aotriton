# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Live comparison: the `head` subject in PostgreSQL against committed releases.

HEAD is never committed to the perfdb (export refuses it); it is compared live
instead, and the output is a list of flagged entries for the tuning-database
fault hunt of the next cycle. Two checks, both with the noise rule of
`perfdb compare` (a change of the median time above NOISE_FRACTION, 3%, for
the same GPU; use `--noise 0.06` when the two sides ran on different GPUs):

    regression             a point of the head subject that is slower than the
                           same point in a reference (a committed tag, or
                           another subject in the database) by more than the
                           noise
    forced_beats_unforced  within one subject, a forced backend is faster than
                           the unforced operator by more than the noise: the
                           tuning database picked a worse variant for that
                           cell than what the backend can do

    python -m aotriton.perfmon.live --workdir <workdir> --module flash \\
        --arch gfx942 gfx950 --against 0.14.2b 0.13b \\
        [--against_subject <subject> ...] [--subject <head subject>] \\
        [--check regression forced_beats_unforced] [--variant V] [--noise 0.03] \\
        [--format csv|markdown] [--output_file F] [--summary_only] [--top 10] \\
        [--perfdb modules/flash/perfdb] [--forced_check_baselines]

`--check` selects the checks (default: both; `regression` needs `--against` or
`--against_subject`). `--subject` is the measured subject: the head subject, or
any subject in the database (a release subject, to run `forced_beats_unforced`
on a finished campaign).

Only points where both sides have a median are compared: a backend that
fails or crashes (the AITER family often does) is not a flag, it is counted in
the summary on stderr. Files whose timing method or L2 flush differ from the
head rows are refused (never compared), as `perfdb compare` does. A baseline
measured on a different GPU than head, by the `platform.csv` device id, gets a
warning to widen the noise.

A summary goes to stderr before the list: per check, reference, arch, iface and
backend the number of flags and the largest margin, then the `--top` flags with
the largest margin and their coordinates.

Output columns: `check, arch, variant, iface, backend, reference, <the family's
problem columns>, <its size columns>, reference_us, measured_us, ratio`. The
problem and size columns are the entry's full coordinates (dtype, hdim,
seqlen_q/k, causal, dropout, bias, ...), so a row maps to a tuning-database
cell. `reference` is the tag or subject compared against (`unforced` for the
forced check). `ratio` is reference_us / measured_us: below 1 the head point
is slower than the reference (regression), above 1 the forced backend is faster
than the unforced operator. Rows are ordered by severity within each check.

The core (`compare_points`, `forced_beats_unforced`, the renderers) touches no
database; the command line fetches the rows.
"""

import argparse
import csv
import io
import sys
from dataclasses import dataclass
from pathlib import Path

from . import perfdb
from .export import HEAD_TAG
from .perfdb import PerfdbError

REGRESSION = 'regression'
FORCED_BEATS_UNFORCED = 'forced_beats_unforced'
UNFORCED_REFERENCE = 'unforced'
FIXED_COLUMNS = ('check', 'arch', 'variant', 'iface', 'backend', 'reference')
RATIO_COLUMNS = ('reference_us', 'measured_us', 'ratio')


@dataclass(frozen=True)
class Point:
    """One timing: the median, and how it was taken."""
    median: float
    timing_method: str | None = None
    l2_flush: str | None = None


@dataclass(frozen=True)
class Flag:
    check: str
    arch: str
    variant: str
    iface: str
    backend: str | None
    reference: str
    coordinates: tuple[str, ...]        # problem row + size row
    reference_us: float
    measured_us: float

    @property
    def ratio(self) -> float:
        return self.reference_us / self.measured_us


def entry_key(desc, entry) -> tuple[str, ...]:
    return tuple(desc.problem_row(entry)) + tuple(desc.size_row(entry))


def points_from_rows(desc, rows) -> dict:
    """{(variant, iface, backend, entry key): Point} of database rows (dicts
    with the columns of perfmon.latest_measurement); only OK rows with a
    median."""
    out = {}
    for row in rows:
        if row['status'] != 'OK' or row.get('median_us') is None:
            continue
        method, flush = row.get('timing_method'), row.get('l2_flush')
        key = (row['variant'], row['iface'], row['backend'], entry_key(desc, desc.entry_from_dict(row['entry'])))
        out[key] = Point(float(row['median_us']),
                         None if method is None else str(method),
                         None if flush is None else str(flush))
    return out


def points_from_perfdb(root: Path, desc, tag: str, arch: str, variant: str | None = None) -> dict:
    """The same map for the committed files of (tag, arch)."""
    files = perfdb.load_arch(Path(root), tag, arch)
    if not files:
        return {}
    entries = perfdb.EntryList.load(Path(root), desc.PROBLEM_COLUMNS, desc.SIZE_COLUMNS)
    out = {}
    for f in files.values():
        keys = f.measurement_keys()
        for v, block in f.blocks.items():
            if variant is not None and v != variant:
                continue
            for ordinal, line in block.items():
                m = perfdb.median(line)
                if m is None or m <= 0:
                    continue
                problem, sizes = entries.rows_of(ordinal)
                out[(v, f.iface, f.backend, tuple(problem) + tuple(sizes))] = Point(m,
                                                                                    keys['timing_method'],
                                                                                    keys['l2_flush'])
    return out


def _compatible(a: Point, b: Point) -> bool:
    """Whether two timings were taken the same way (unknown matches anything)."""
    return all(x is None or y is None or x == y
               for x, y in ((a.timing_method, b.timing_method), (a.l2_flush, b.l2_flush)))


def compare_points(arch: str, reference: str, ref_points: dict, measured: dict, noise: float):
    """Flag the points of `measured` slower than `ref_points` beyond `noise`.

    Returns (flags, counts): counts has `compared`, `no_reference` (the
    measured point has no median on the reference side) and `refused` (the
    timing methods differ), for the summary.
    """
    flags = []
    counts = {'compared': 0, 'no_reference': 0, 'refused': 0}
    for (variant, iface, backend, coords), mine in measured.items():
        theirs = ref_points.get((variant, iface, backend, coords))
        if theirs is None:
            counts['no_reference'] += 1
        elif not _compatible(mine, theirs):
            counts['refused'] += 1
        else:
            counts['compared'] += 1
            ratio = theirs.median / mine.median
            if ratio < 1.0 and perfdb.beyond_noise(ratio, noise):
                flags.append(Flag(REGRESSION, arch, variant, iface, backend, reference, coords,
                                  theirs.median, mine.median))
    return flags, counts


def forced_beats_unforced(arch: str, points: dict, noise: float) -> list[Flag]:
    """Within one set of points, the forced backends faster than the unforced
    operator of the same (variant, iface, entry) beyond `noise`."""
    flags = []
    for (variant, iface, backend, coords), forced in points.items():
        if backend is None:
            continue
        unforced = points.get((variant, iface, None, coords))
        if unforced is None or not _compatible(forced, unforced):
            continue
        ratio = unforced.median / forced.median
        if ratio > 1.0 and perfdb.beyond_noise(ratio, noise):
            flags.append(Flag(FORCED_BEATS_UNFORCED, arch, variant, iface, backend, UNFORCED_REFERENCE,
                              coords, unforced.median, forced.median))
    return flags


def sort_flags(flags) -> list[Flag]:
    """By check, then the worst first (regressions: lowest ratio; forced: highest)."""
    def key(f: Flag):
        sign = 1 if f.check == REGRESSION else -1
        return (f.check, f.arch, sign * f.ratio, f.variant, f.iface, f.backend or '', f.coordinates)
    return sorted(flags, key=key)


def flag_coordinates(desc, flag: Flag) -> str:
    """`name=value` pairs of the flagged entry (as the family describes an LLM
    case), the unset ones left out."""
    n = len(desc.PROBLEM_COLUMNS)
    columns = desc.case_columns(desc.entry_from_rows(flag.coordinates[:n], flag.coordinates[n:]))
    return ' '.join(f'{name}={value}' for name, value in columns.items() if value != '-')


def margin(flag: Flag) -> float:
    """How much faster the better side is: `ratio` for a forced backend, 1/ratio for a regression."""
    return flag.ratio if flag.check == FORCED_BEATS_UNFORCED else 1.0 / flag.ratio


def summary_lines(desc, flags, top: int) -> list[str]:
    """Counts per (check, reference, arch, iface, backend) with the largest
    margin, then the `top` flags per check with the largest margin."""
    if not flags:
        return ['nothing flagged']
    groups: dict[tuple, list[Flag]] = {}
    for f in flags:
        groups.setdefault((f.check, f.reference, f.arch, f.iface, f.backend or UNFORCED_REFERENCE), []).append(f)
    lines = []
    for check in sorted({f.check for f in flags}):
        mine = [f for f in flags if f.check == check]
        table = [['reference', 'arch', 'iface', 'backend', 'flagged', 'largest margin']]
        table += [[ref, arch, iface, backend, str(len(g)), perfdb.fmt4(max(margin(f) for f in g))]
                  for (c, ref, arch, iface, backend), g in sorted(groups.items()) if c == check]
        lines += [f'{check}: {len(mine)} flagged'] + ['  ' + t for t in perfdb.text_table(table, left=4)]
        worst = sorted(mine, key=margin, reverse=True)[:top]
        lines.append(f'  top {len(worst)} by margin:')
        for f in worst:
            lines.append(f'    {f.arch} {f.iface} {f.backend or UNFORCED_REFERENCE} vs {f.reference}: '
                         f'{perfdb.fmt4(f.reference_us)} -> {perfdb.fmt4(f.measured_us)} us, '
                         f'margin {perfdb.fmt4(margin(f))}')
            lines.append(f'        {flag_coordinates(desc, f)}')
    return lines


def header_of(desc) -> list[str]:
    return list(FIXED_COLUMNS) + list(desc.PROBLEM_COLUMNS) + list(desc.SIZE_COLUMNS) + list(RATIO_COLUMNS)


def _cells(flag: Flag) -> list[str]:
    return ([flag.check, flag.arch, flag.variant, flag.iface, flag.backend or UNFORCED_REFERENCE,
             flag.reference] + list(flag.coordinates)
            + [perfdb.fmt4(flag.reference_us), perfdb.fmt4(flag.measured_us), perfdb.fmt4(flag.ratio)])


def render_csv(desc, flags) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator='\n')
    writer.writerow(header_of(desc))
    for flag in flags:
        writer.writerow(_cells(flag))
    return out.getvalue()


def render_markdown(desc, flags) -> str:
    header = header_of(desc)
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(' --- ' for _ in header) + '|']
    lines += ['| ' + ' | '.join(_cells(f)) + ' |' for f in flags]
    return '\n'.join(lines) + '\n'


def gpu_warnings(root: Path, tag: str, arch: str, rows) -> list[str]:
    """Warn when the machines of `tag` (platform.csv) have another GPU device
    id than the head rows: the noise of different GPUs is about 6%."""
    committed = {r['gpu_device_id'] for r in perfdb.read_platform(perfdb.data_dir(root, tag, arch)
                                                                  / perfdb.PLATFORM_CSV)
                 if r.get('gpu_device_id')}
    live = set()
    for row in rows:
        device = (row.get('platform') or {}).get('pci_device_id')
        if device is not None:
            live.add(f'0x{device:04x}' if isinstance(device, int) else str(device))
    if committed and live and committed.isdisjoint(live):
        return [f'{tag}/{arch}: GPU device id {sorted(committed)} differs from the live rows\' '
                f'{sorted(live)}; the noise across GPUs is about 6%, consider --noise 0.06']
    return []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='python -m aotriton.perfmon.live',
                                     description='Compare the head subject in PostgreSQL against committed '
                                                 'release tags and other subjects, and check that no forced '
                                                 'backend beats the unforced operator. Lists the flagged '
                                                 'entries with their full problem coordinates.')
    parser.add_argument('--workdir',
                        type=Path,
                        default=None,
                        help='Working directory (config.rc); default: AOTRITON_WORKDIR')
    parser.add_argument('--module',
                        default='flash',
                        help='Family (default: flash)')
    parser.add_argument('--arch',
                        nargs='+',
                        required=True,
                        help='Architectures')
    parser.add_argument('--variant',
                        default=None,
                        help='Only this variant (default: every variant)')
    parser.add_argument('--subject',
                        default=None,
                        help='The measured subject (default: the only head subject in the database)')
    parser.add_argument('--against',
                        nargs='+',
                        default=[],
                        metavar='TAG',
                        help='Committed release tags of the perfdb to compare against')
    parser.add_argument('--against_subject',
                        nargs='+',
                        default=[],
                        help='Other subjects in the database to compare against')
    parser.add_argument('--check',
                        nargs='+',
                        choices=(REGRESSION, FORCED_BEATS_UNFORCED),
                        default=[REGRESSION, FORCED_BEATS_UNFORCED],
                        help='Checks to run (default: both)')
    parser.add_argument('--top',
                        type=int,
                        default=10,
                        help='How many flags of each check the summary lists (default 10)')
    parser.add_argument('--summary_only',
                        action='store_true',
                        help='Print only the summary (to stderr), not the list')
    parser.add_argument('--noise',
                        type=float,
                        default=perfdb.NOISE_FRACTION,
                        help='Fraction of the median below which a change is noise (default 0.03, '
                             'same GPU; use 0.06 when the sides ran on different GPUs)')
    parser.add_argument('--forced_check_baselines',
                        action='store_true',
                        help='Also run the forced-beats-unforced check on the references, not only on head')
    parser.add_argument('--format',
                        choices=('csv', 'markdown'),
                        default='csv')
    parser.add_argument('--output_file',
                        type=Path,
                        default=None,
                        help='Write the list here (default: stdout)')
    parser.add_argument('--perfdb',
                        type=Path,
                        default=None,
                        help='perfdb directory (default: modules/<module>/perfdb)')
    return parser


def run(desc, root: Path, args, fetch, list_subjects) -> tuple[list[Flag], list[str], dict]:
    """The comparison of every arch. `fetch(arch, subjects)` returns the
    latest database rows and `list_subjects(arch)` the subjects that have
    some (so tests need no database). Returns (flags,
    warnings, counts)."""
    warnings: list[str] = []
    flags: list[Flag] = []
    totals = {'compared': 0, 'no_reference': 0, 'refused': 0}
    for arch in args.arch:
        head_subject = args.subject
        if head_subject is None:
            head_subject = _head_subject(list_subjects(arch), arch)
        head_rows = [r for r in fetch(arch, [head_subject]) if args.variant in (None, r['variant'])]
        if not head_rows:
            warnings.append(f'{arch}: no measurements of {head_subject}')
            continue
        measured = points_from_rows(desc, head_rows)
        if FORCED_BEATS_UNFORCED in args.check:
            flags += forced_beats_unforced(arch, measured, args.noise)
        for tag in args.against:
            points = points_from_perfdb(root, desc, tag, arch, args.variant)
            if not points:
                warnings.append(f'{arch}: tag {tag} has no committed data')
                continue
            warnings += gpu_warnings(root, tag, arch, head_rows)
            found, counts = compare_points(arch, tag, points, measured, args.noise)
            if REGRESSION in args.check:
                flags += found
                for k, v in counts.items():
                    totals[k] += v
            if args.forced_check_baselines and FORCED_BEATS_UNFORCED in args.check:
                flags += [_renamed(f, tag) for f in forced_beats_unforced(arch, points, args.noise)]
        for subject in args.against_subject:
            rows = [r for r in fetch(arch, [subject]) if args.variant in (None, r['variant'])]
            if not rows:
                warnings.append(f'{arch}: no measurements of {subject}')
                continue
            points = points_from_rows(desc, rows)
            found, counts = compare_points(arch, subject, points, measured, args.noise)
            if REGRESSION in args.check:
                flags += found
                for k, v in counts.items():
                    totals[k] += v
            if args.forced_check_baselines and FORCED_BEATS_UNFORCED in args.check:
                flags += [_renamed(f, subject) for f in forced_beats_unforced(arch, points, args.noise)]
    return sort_flags(flags), warnings, totals


def _renamed(flag: Flag, reference: str) -> Flag:
    """A baseline's forced flag, labelled with the baseline it came from."""
    return Flag(flag.check, flag.arch, flag.variant, flag.iface, flag.backend,
                f'{UNFORCED_REFERENCE} ({reference})', flag.coordinates, flag.reference_us,
                flag.measured_us)


def _head_subject(subjects, arch: str) -> str:
    heads = set()
    for subject in subjects:
        try:
            if perfdb.subject_tag(subject)[0] == HEAD_TAG:
                heads.add(subject)
        except PerfdbError:
            continue
    if len(heads) != 1:
        raise PerfdbError(f'{arch}: expected one head subject in the database, found {sorted(heads) or "none"}; '
                          'select one with --subject')
    return heads.pop()


def main(argv=None) -> int:
    from .export import _connect
    from . import pq
    from .registry import default_perfdb_dir, make_perf_desc
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    desc = make_perf_desc(args.module)
    root = args.perfdb or default_perfdb_dir(args.module)
    conn = _connect(args)
    try:
        cache: dict = {}

        def fetch(arch, subjects):
            key = (arch, tuple(subjects) if subjects else None)
            if key not in cache:
                cache[key] = list(pq.iter_latest(conn,
                                                 arch=arch,
                                                 variant=args.variant,
                                                 subjects=subjects,
                                                 module=args.module))
            return cache[key]

        def list_subjects(arch):
            return sorted({s['subject'] for s in pq.list_selections(conn,
                                                                    module=args.module,
                                                                    arch=arch,
                                                                    variant=args.variant)})

        try:
            flags, warnings, totals = run(desc, root, args, fetch, list_subjects)
        except PerfdbError as e:
            sys.exit(f'Error: {e}')
    finally:
        conn.close()
    for warning in warnings:
        print(f'warning: {warning}', file=sys.stderr)
    if REGRESSION in args.check:
        print(f'# compared {totals["compared"]} points, {totals["no_reference"]} without data on the reference '
              f'side, {totals["refused"]} refused (timing method differs); noise {args.noise:.1%}',
              file=sys.stderr)
    for line in summary_lines(desc, flags, args.top):
        print(line, file=sys.stderr)
    if args.summary_only:
        return 0
    text = (render_markdown if args.format == 'markdown' else render_csv)(desc, flags)
    if args.output_file:
        args.output_file.write_text(text, encoding='utf-8')
    else:
        sys.stdout.write(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
