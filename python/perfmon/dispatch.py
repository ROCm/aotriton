# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Dispatch perfmon tasks (the perfmon application's `dispatch` entry point).

    python -m aotriton.perfmon.dispatch --pool gfx942 [--subpool <variant>] \\
        --workdir <workdir> --module flash --entry_set prime|coverage|llm \\
        [--subject aotriton-0.13b+rocm10.1.0 ...] [--hdim 64 128] [--causal 0] ...

The engine owns the common options (--pool, --subpool, --category, --dry_run,
--yes, --skip_completed, ...) and the insert; this module is perfmon's
TaskSource:

* One task per (subject, entry), payload {module, subject, entry}; the arch
  is the pool. An LLM case measured forward only adds `ifaces`.
* Tasks are yielded subject-major: claims follow insert order, and switching
  subjects restarts the native runner, so grouping keeps the switches rare.
* `--subject` defaults to every subject installed for the arch under
  `<workdir>/installed/perfmon/<arch>/`; a subject that is not installed is
  an error here, not on a worker.
* One `--<axis>` option per axis of the family's entry sets. `--entry_set`
  only chooses the default values; an option given replaces them.
* `--entry_set llm` dispatches the family's LLM cases instead of a grid;
  `--phase` filters them, and the grid options do not apply.
* Entries the arch cannot run (`validate_hw_feature`) are printed and not
  dispatched; the export records them as excluded.
"""

import argparse
import sys
from pathlib import Path

from aotriton.pgdef import paths
from aotriton.pgdef.pq import dispatch as pgdispatch
from aotriton.pgdef.pq.category import format_category
from aotriton.pgdef.pq.tasks import TaskSpec
from aotriton.pgdef.workdir import load_config

from .exaid_perfmon import INSTALL_SUBTREE
from .registry import available_module_names, make_perf_desc

APP = 'perfmon'
GRID_SETS = ('prime', 'coverage')
ENTRY_SETS = GRID_SETS + ('llm',)
SUBJECT_PREFIX = 'aotriton-'


def installed_subjects(workdir: Path, arch: str) -> list[str]:
    root = workdir / INSTALL_SUBTREE / arch
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.startswith(SUBJECT_PREFIX))


def _axis_type(values):
    first = values[0] if values else ''
    if isinstance(first, bool) or isinstance(first, int):
        return int
    if isinstance(first, float):
        return float
    return str


class PerfmonTaskSource:
    """perfmon's TaskSource (see pgdef.pq.dispatch). `argv` is the command
    line: the axis options depend on the module, so it is read before the
    parser is built."""

    default_category = (APP, None)

    def __init__(self, argv: list[str]):
        self._argv = argv
        self.desc = None
        self.module_name = None
        self._axes: tuple[str, ...] = ()
        self._entries: list[tuple] = []    # (entry, ifaces or None)
        self._entry_fields: tuple[str, ...] = ()

    def _peek_module(self) -> str | None:
        peek = argparse.ArgumentParser(add_help=False)
        peek.add_argument('--module')
        known, _ = peek.parse_known_args(self._argv)
        return known.module

    def add_arguments(self, parser):
        parser.add_argument('--module',
                            required=True,
                            choices=available_module_names(),
                            help='Family to measure')
        parser.add_argument('--workdir',
                            type=Path,
                            default=None,
                            help='Working directory (config.rc, installed/perfmon/); default: AOTRITON_WORKDIR')
        parser.add_argument('--arch',
                            type=str,
                            nargs='+',
                            dest='pool',
                            help='Synonym of --pool: the architecture(s) to dispatch to')
        parser.add_argument('--entry_set',
                            choices=ENTRY_SETS,
                            default='prime',
                            help='Which entries: a grid (prime, coverage) or the LLM cases (llm)')
        parser.add_argument('--subject',
                            type=str,
                            nargs='+',
                            default=None,
                            help='Subjects to measure (default: every one installed for the arch)')
        parser.add_argument('--phase',
                            choices=('fwd', 'fwd_bwd'),
                            nargs='+',
                            default=None,
                            help='llm only: the phases of the LLM cases to dispatch (default: all)')
        name = self._peek_module()
        if name not in available_module_names():
            return      # argparse reports the bad --module
        desc = make_perf_desc(name)
        axes: dict[str, list] = {}
        for set_axes in desc.entry_sets().values():
            for axis, values in set_axes.items():
                axes.setdefault(axis, list(values))
        for axis, values in axes.items():
            arg_type = _axis_type(values)
            parser.add_argument(f'--{axis}',
                                type=arg_type,
                                nargs='+',
                                default=None,
                                metavar=arg_type.__name__.upper(),
                                help=f'{axis} values (default: the entry set\'s; e.g. {values[:4]})')
        self._axes = tuple(axes)

    # -- hooks called by the engine's driver ---------------------------------------

    def configure(self, args):
        if args.category[0] != APP or args.category[1] is not None:
            sys.exit(f'perfmon tasks have category {APP!r} with no second level, '
                     f'not {format_category(args.category)!r}')
        args.workdir = args.workdir or paths.workdir()
        if not args.workdir.is_dir():
            sys.exit(f'Error: working directory does not exist: {args.workdir}')
        load_config(args.workdir)
        self.module_name = args.module
        self.desc = make_perf_desc(args.module)
        self._entry_fields = self.desc.entry_fields()
        given = [axis for axis in self._axes if getattr(args, axis, None) is not None]
        if args.entry_set == 'llm':
            if given:
                sys.exit(f'--entry_set llm takes no grid options: {", ".join("--" + a for a in given)}')
            self._entries = self._llm_entries(args.phase)
        else:
            if args.phase is not None:
                sys.exit('--phase applies to --entry_set llm only')
            axes = dict(self.desc.entry_sets()[args.entry_set])
            for axis in given:
                axes[axis] = getattr(args, axis)
            dropped: list = []
            self._entries = [(e, None) for e in self.desc.entries(axes, dropped=dropped)]
            if dropped:
                print(f'{len(dropped)} grid point(s) do not fit the size rule and are not generated')
        print(f'{args.entry_set}: {len(self._entries)} entr{"y" if len(self._entries) == 1 else "ies"} per subject')

    def _llm_entries(self, phases) -> list[tuple]:
        out = []
        for case in self.desc.llm_cases():
            if case.pending:
                print(f'Skipping pending LLM case {case.model}')
                continue
            if phases is not None and case.phase not in phases:
                continue
            ifaces = self.desc.phase_ifaces(case.phase)
            out.append((case.entry, None if tuple(ifaces) == tuple(self.desc.ifaces()) else list(ifaces)))
        return out

    def skip_key(self, payload):
        """--skip_completed compares (module, subject, entry fields, ifaces)."""
        entry = payload['entry']
        ifaces = payload.get('ifaces')
        return ((payload['module'], payload['subject'])
                + tuple(entry[name] for name in self._entry_fields)
                + (tuple(ifaces) if ifaces else None,))

    def subjects(self, args, pool: str) -> list[str]:
        installed = installed_subjects(args.workdir, pool)
        if args.subject is None:
            if not installed:
                sys.exit(f'no subject installed under {args.workdir / INSTALL_SUBTREE / pool}')
            return installed
        missing = [s for s in args.subject if s not in installed]
        if missing:
            sys.exit(f'subject(s) not installed for {pool} under {args.workdir / INSTALL_SUBTREE / pool}: '
                     f'{", ".join(missing)}')
        return list(args.subject)

    def tasks(self, args, pool):
        accepted = []
        rejected: dict[str, int] = {}
        for entry, ifaces in self._entries:
            ok, reason = self.desc.validate_hw_feature(pool, entry)
            if ok:
                accepted.append((entry, ifaces))
            else:
                rejected[reason] = rejected.get(reason, 0) + 1
        for reason, count in sorted(rejected.items()):
            print(f'{pool}: {count} entr{"y" if count == 1 else "ies"} excluded: {reason}')
        for subject in self.subjects(args, pool):
            for entry, ifaces in accepted:
                payload = {'module': self.module_name,
                           'subject': subject,
                           'entry': self.desc.entry_to_dict(entry)}
                if ifaces is not None:
                    payload['ifaces'] = ifaces
                if args.verbose:
                    print(f'{pool}: {subject} {payload["entry"]}')
                yield TaskSpec(payload=payload, category=args.category)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    return pgdispatch.drive(PerfmonTaskSource(argv),
                            argv,
                            prog='python -m aotriton.perfmon.dispatch')


if __name__ == '__main__':
    sys.exit(main())
