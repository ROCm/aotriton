# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The dispatch driver: what every application's `dispatch` command shares.

An application's entry point is a few lines that build its own TaskSource and
call drive():

    # python/tune/dispatch.py
    if __name__ == '__main__':
        sys.exit(drive(TuneTaskSource(), sys.argv[1:]))

    python -m aotriton.tune.dispatch --category tune,kernel --pool gfx950 \\
        [--subpool MI355] <tune options>

The engine owns the common options (--pool, --subpool, --category, --dry_run,
--yes, --skip_completed, --max_attempts, --verbose), the confirmation summary,
partition creation, batching and the plain INSERT. The application owns what a
task IS: its axes and options, entry validation (`validate_hw_feature`), and the
payload. The engine never looks inside a payload.

Identical tasks are legal; the engine inserts what it is given. Skipping work
that is already done is the opt-in --skip_completed, which compares each new
payload with the payloads of `done` tasks of the same pool and category
(canonical JSON by default, or the application's `skip_key`).
"""

import argparse
import json
import sys
from typing import Callable, Iterable, Protocol

import psycopg

from . import tasks as _tasks
from .category import Category, argparse_type, format_category
from .conn import params_from_env
from .tasks import TaskSelector, TaskSpec


class TaskSource(Protocol):
    """Implemented by each application.

    Optional attributes (looked up with getattr):
        configure(args)      called right after parsing (validate options,
                             load configuration into the environment, ...)
        default_category     Category used when --category is not given
        default_max_attempts retry policy stamped on tasks that do not set one
        default_pools(args)  -> list[str]: pools when --pool is not given
        skip_key(payload)    -> hashable: what --skip_completed compares
        summary(args)        -> dict shown by the confirmation prompt (default:
                             the parsed arguments)
    """

    def add_arguments(self, parser: argparse.ArgumentParser) -> None: ...

    def tasks(self, args: argparse.Namespace, pool: str) -> Iterable[TaskSpec]: ...


def canonical(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(',', ':'))


def add_common_arguments(parser: argparse.ArgumentParser, source) -> None:
    default_category = getattr(source, 'default_category', None)
    parser.add_argument('--category', type=argparse_type, default=default_category,
                        required=default_category is None,
                        help="Task category, e.g. 'tune,kernel' or 'tune,op'"
                             + (f" (default: {format_category(default_category)})" if default_category else ''))
    parser.add_argument('--pool', type=str, nargs='+',
                        help='Pool(s) to dispatch to: the arch (gfx950). Default: the application\'s choice')
    parser.add_argument('--subpool', type=str, default=None,
                        help='Restrict the tasks to one machine group of the pool (e.g. MI355); '
                             'default: any machine of the pool')
    parser.add_argument('--skip_completed', action='store_true',
                        help='Skip tasks whose payload equals that of a done task of the same pool and category')
    parser.add_argument('--max_attempts', type=int, default=None, metavar='N',
                        help='Attempts before a failing task is failed for good. '
                             'Default: the application\'s, else the engine\'s, which is 1 '
                             '(no automatic retry)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print what would be dispatched and exit without inserting')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='Skip the confirmation prompt')


def build_parser(source, prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser, source)
    source.add_arguments(parser)
    return parser


def resolve_pools(source, args) -> list[str]:
    if args.pool:
        return list(args.pool)
    default_pools = getattr(source, 'default_pools', None)
    pools = list(default_pools(args)) if default_pools else []
    if not pools:
        raise SystemExit('no --pool given, and the application has no default pools')
    return pools


def plan(source, args, pool: str, done_keys: set | None, skip_key: Callable | None
         ) -> tuple[list[TaskSpec], int]:
    """The specs to insert for one pool, and how many were skipped as done.

    Pure apart from `source.tasks()`: the unit-testable core of the driver.
    """
    default_attempts = args.max_attempts or getattr(source, 'default_max_attempts', None)
    specs: list[TaskSpec] = []
    skipped = 0
    for spec in source.tasks(args, pool):
        spec.category = args.category if spec.category is None else spec.category
        if spec.subpool is None:
            spec.subpool = args.subpool
        if spec.max_attempts is None:
            spec.max_attempts = default_attempts
        if done_keys is not None and skip_key(spec.payload) in done_keys:
            skipped += 1
            continue
        specs.append(spec)
    return specs, skipped


def drive(source, argv: list[str] | None = None, *, prog: str | None = None, conn=None,
          stdin_is_tty: bool | None = None) -> int:
    """Parse the command line and dispatch. Returns a process exit status.

    `conn` is for tests and tools that already hold a connection; otherwise one
    is opened from the environment (PG* variables) when it is first needed, and
    never for a dry run.
    """
    args = build_parser(source, prog).parse_args(argv)
    configure = getattr(source, 'configure', None)
    if configure is not None:
        configure(args)
    pools = resolve_pools(source, args)
    skip_key = getattr(source, 'skip_key', None) or canonical
    tty = sys.stdin.isatty() if stdin_is_tty is None else stdin_is_tty

    opened = None

    def get_conn():
        nonlocal conn, opened
        if conn is None:
            opened = conn = psycopg.connect(**params_from_env(), autocommit=True)
        return conn

    try:
        plans: list[tuple[str, list[TaskSpec]]] = []
        for pool in pools:
            done_keys = None
            if args.skip_completed:
                print(f'Querying PostgreSQL for done tasks of {format_category(args.category)} in {pool}...')
                done_keys = {skip_key(p) for p in _tasks.payloads(
                    get_conn(), TaskSelector(pool=pool, category=args.category, status='done'))}
                if args.dry_run:
                    print(f'  {len(done_keys)} done payload(s)')
            specs, skipped = plan(source, args, pool, done_keys, skip_key)
            plans.append((pool, specs))
            print(f'{pool}: {len(specs)} task(s) to dispatch'
                  + (f', {skipped} already done' if args.skip_completed else ''))

        total = sum(len(specs) for _, specs in plans)
        print(f'Total: {total} task(s), category {format_category(args.category)}'
              + (f', subpool {args.subpool}' if args.subpool else ''))

        if total and not args.yes and tty and not args.dry_run:
            for key, value in vars(args).items():
                print(f'  {key}: {value}')
            try:
                if input('Proceed with dispatch? [y/N]: ').lower() not in ('y', 'yes'):
                    print('Dispatch cancelled')
                    return 1
            except (KeyboardInterrupt, EOFError):
                print('\nDispatch cancelled')
                return 1

        if args.dry_run:
            print('Dry run mode - tasks not dispatched')
            return 0

        inserted = 0
        for pool, specs in plans:
            if not specs:
                continue
            _tasks.ensure_partition(get_conn(), pool)
            inserted += _tasks.insert(get_conn(), pool, specs, batch_size=1000)
        print(f'Dispatched {inserted} task(s)')
        return 0
    finally:
        if opened is not None:
            opened.close()
