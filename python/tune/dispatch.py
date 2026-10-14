#!/usr/bin/env python
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Dispatch tuning tasks to the PostgreSQL queue (the tune application's
`dispatch` entry point).

    python -m aotriton.tune.dispatch --category tune,kernel --pool gfx950 \\
        --workdir /path/to/workdir --module flash [--subpool MI355] <flash options>

The common options (--pool, --subpool, --category, --dry_run, --yes,
--skip_completed, ...) and the insert belong to the engine (`pgdef.pq.dispatch`);
this module is the tune TaskSource:

1. loads a tuning module (e.g. 'flash') and queries its parameter choices;
2. adds one `--<axis>` option per entry field, filtering those choices;
3. yields a TaskSpec per (entry, pool) that the module's hardware check
   accepts. The payload is {module, entry, max_hsaco?}; the arch is the task's
   pool and the level its category, not payload fields.

`--arch` is accepted as a synonym of `--pool` (tune's word for it). A pool is
required: each must be one tune's pq/schema.sql creates partitions for.
"""

import argparse
import sys
from dataclasses import asdict, fields
from pathlib import Path

from aotriton.pgdef.pq import dispatch as pgdispatch
from aotriton.pgdef.pq.tasks import TaskSpec
from aotriton.pgdef.workdir import load_config

from .pq.tasks import level_of


def str_to_bool(s):
    """Convert '0' or '1' string to boolean for argparse."""
    if s == '0':
        return False
    elif s == '1':
        return True
    else:
        raise argparse.ArgumentTypeError(f"Boolean value must be 0 or 1, got '{s}'")


def get_available_modules():
    """
    Return a dict mapping each registered tuning module name to an
    instantiated TuneDesc object. The module list is a static registry (see
    registry.available_module_names), not a directory glob: family tune blocks
    live under modules/<family>/tune/, outside this package.
    """
    from .registry import available_module_names, load_tune_module
    modules = {}
    for name in available_module_names():
        mod = load_tune_module(name)
        if hasattr(mod, 'TuneDesc'):
            modules[name] = mod.TuneDesc()
    return modules


def generate_filtered_entries(module_instance, args):
    """
    Generate entries filtered by command-line arguments.

    args should have attributes matching entry field names,
    each containing a list of allowed values (or None for no filter).
    """
    all_choices = module_instance.get_entry_choices()

    filtered_choices = {}
    for field in fields(all_choices):
        all_values = getattr(all_choices, field.name)
        filter_values = getattr(args, field.name, None)

        if filter_values is None:
            filtered_choices[field.name] = all_values
        else:
            # Convert bool to int for comparison if needed
            def normalize(v):
                return int(v) if isinstance(v, bool) else v

            normalized_filter = [normalize(v) for v in filter_values]
            filtered_values = [v for v in all_values if normalize(v) in normalized_filter]

            if not filtered_values:
                # No values match the filter - empty result
                return

            filtered_choices[field.name] = filtered_values

    ENTRY_CLASS = type(all_choices)
    yield from module_instance.generate_entries_from_choices(ENTRY_CLASS(**filtered_choices))


def _hashable(value):
    """JSON arrays come back as lists; entries hold tuples."""
    return tuple(value) if isinstance(value, list) else value


def add_entry_arguments(parser, module_instance):
    """One `--<field>` option per entry field, with that field's choices."""
    all_choices = module_instance.get_entry_choices()
    for field in fields(all_choices):
        param_name = field.name
        param_choices = getattr(all_choices, param_name)

        metavar = None
        display_choices = param_choices
        actual_choices = param_choices

        if param_choices:
            first_choice = param_choices[0]
            if isinstance(first_choice, bool):
                arg_type = str_to_bool
                actual_choices = [False, True]
                display_choices = [0, 1]
                metavar = '0/1'
            elif isinstance(first_choice, int):
                arg_type = int
            elif isinstance(first_choice, float):
                arg_type = float
            else:
                arg_type = str
        else:
            arg_type = str

        if metavar is None:
            metavar = arg_type.__name__.upper()

        parser.add_argument(f'--{param_name}',
                            type=arg_type,
                            nargs='*',
                            default=actual_choices,
                            choices=actual_choices,
                            metavar=metavar,
                            help=f'Choices: {display_choices}')


class TuneTaskSource:
    """The tune application's TaskSource (see pgdef.pq.dispatch).

    `argv` is the command line: the entry options depend on which module is
    selected, so the source reads `--module` from it before the parser is built.
    """

    default_category = ('tune', 'kernel')

    def __init__(self, argv: list[str]):
        self._argv = argv
        self._modules = None
        self.module_name = None
        self.module = None
        self._entry_fields: tuple[str, ...] = ()

    def _available(self):
        if self._modules is None:
            self._modules = get_available_modules()
        return self._modules

    def _peek_module(self) -> str | None:
        peek = argparse.ArgumentParser(add_help=False)
        peek.add_argument('--module')
        known, _ = peek.parse_known_args(self._argv)
        return known.module

    def add_arguments(self, parser):
        modules = self._available()
        parser.add_argument('--module', required=True, choices=sorted(modules),
                            help='Tuning module (family) to dispatch')
        parser.add_argument('--workdir', type=Path, required=True,
                            help='Project working directory (config.rc, workers.db)')
        parser.add_argument('--arch', type=str, nargs='+', dest='pool',
                            help='Synonym of --pool: the architecture(s) to dispatch to')
        parser.add_argument('--max_hsaco', type=int, metavar='N',
                            help='Maximum number of hsaco kernels to tune per entry (default: all)')
        name = self._peek_module()
        if name in modules:      # without a valid --module, argparse reports it
            add_entry_arguments(parser, modules[name])

    # -- hooks called by the engine's driver ---------------------------------------

    def configure(self, args):
        """After parsing: validate the workdir, load config.rc into the
        environment (where the engine reads PG parameters), pick the module."""
        if not args.workdir.is_dir():
            sys.exit(f"Error: working directory does not exist: {args.workdir}")
        level_of(args.category)             # tune's categories are ('tune', 'kernel'|'op')
        load_config(args.workdir)
        self.module_name = args.module
        self.module = self._available()[args.module]
        self._entry_fields = tuple(f.name for f in fields(self.module.ENTRY_CLASS))

    def skip_key(self, payload):
        """--skip_completed compares (module, entry fields); max_hsaco does not
        make an entry a different entry."""
        entry = payload['entry']
        return (payload['module'],) + tuple(_hashable(entry[name]) for name in self._entry_fields)

    def tasks(self, args, pool):
        tty_output = sys.stdin.isatty()
        printed_hw_reasons = set()
        for entry in generate_filtered_entries(self.module, args):
            supported, reason = self.module.validate_hw_feature(pool, entry)
            if not supported:
                if tty_output and (pool, reason) not in printed_hw_reasons:
                    printed_hw_reasons.add((pool, reason))
                    print(f"Skipping {pool} configurations: {reason}")
                continue
            payload = {"module": self.module_name, "entry": asdict(entry)}
            if args.max_hsaco is not None:
                payload["max_hsaco"] = {"*": args.max_hsaco}
            yield TaskSpec(payload=payload, category=args.category)
            if args.verbose:
                print(f"Prepared task for {pool}: {payload['entry']}")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    return pgdispatch.drive(TuneTaskSource(argv), argv, prog='python -m aotriton.tune.dispatch')


if __name__ == '__main__':
    sys.exit(main())
