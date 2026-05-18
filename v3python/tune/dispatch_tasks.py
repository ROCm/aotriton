#!/usr/bin/env python
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Dispatch tuning tasks to PostgreSQL queue (Tuner v3.5).

This script:
1. Loads a tuning module (e.g., 'flash')
2. Queries the module's parameter choices
3. Allows filtering via command-line arguments
4. Dispatches tasks to PostgreSQL queue using bulk INSERT
"""

import sys
import os
import argparse
import sqlite3
from pathlib import Path
from importlib import import_module
from dataclasses import fields, asdict
import json

def load_config(workdir: Path):
    """Load config.rc from workdir and set environment variables."""
    config_rc = workdir / 'config.rc'
    if not config_rc.exists():
        sys.exit(f"Error: config.rc not found at {config_rc}")

    # Parse config.rc and set environment variables
    with open(config_rc) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Remove inline comments
            if '#' in line:
                line = line[:line.index('#')].strip()
            if '=' in line:
                # Simple parsing: KEY=VALUE
                key, value = line.split('=', 1)
                # Remove quotes if present
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value

def get_db_connection_params():
    """Get PostgreSQL connection parameters from environment."""
    postgres_user = os.environ.get('POSTGRES_USER')
    postgres_password = os.environ.get('POSTGRES_PASSWORD')
    celery_service_host = os.environ.get('CELERY_SERVICE_HOST')
    postgres_port = os.environ.get('POSTGRES_PORT')

    if not all([postgres_user, postgres_password, celery_service_host, postgres_port]):
        sys.exit("Error: Missing PostgreSQL credentials in config.rc. "
                 "Required: POSTGRES_USER, POSTGRES_PASSWORD, CELERY_SERVICE_HOST, POSTGRES_PORT")

    return {
        'host': celery_service_host,
        'port': int(postgres_port),
        'user': postgres_user,
        'password': postgres_password,
    }

def load_module(module_name: str):
    """Load tuning module and return the module class instance."""
    try:
        # Import from relative path since we're in v3python.tune
        mod = import_module(f'.{module_name}', package='v3python.tune')
        # Module __init__.py should export TuneDesc as the main class
        if not hasattr(mod, 'TuneDesc'):
            sys.exit(f"Error: Module '.{module_name}' has no class 'TuneDesc'")
        module_class = getattr(mod, 'TuneDesc')
        return module_class()
    except ImportError as e:
        sys.exit(f"Error: Failed to import module '{module_name}': {e}")

def get_parameter_choices(module_instance):
    """
    Get parameter choices from module.

    Returns the ENTRY_CLASS instance where each field is a list of choices.
    """
    return module_instance.get_entry_choices()

def generate_filtered_entries(module_instance, args):
    """
    Generate entries filtered by command-line arguments.

    args should have attributes matching entry field names,
    each containing a list of allowed values (or None for no filter).
    """
    # Get all choices
    all_choices = module_instance.get_entry_choices()

    # Filter choices based on command-line arguments
    filtered_choices = {}
    for field in fields(all_choices):
        all_values = getattr(all_choices, field.name)
        filter_values = getattr(args, field.name, None)

        if filter_values is None:
            # No filter specified, use all values
            filtered_choices[field.name] = all_values
        else:
            # Filter the choices
            # Convert bool to int for comparison if needed
            def normalize(v):
                return int(v) if isinstance(v, bool) else v

            normalized_filter = [normalize(v) for v in filter_values]
            filtered_values = [v for v in all_values if normalize(v) in normalized_filter]

            if not filtered_values:
                # No values match the filter - empty result
                return

            filtered_choices[field.name] = filtered_values

    # Create filtered choices instance
    ENTRY_CLASS = type(all_choices)
    filtered_choices_obj = ENTRY_CLASS(**filtered_choices)

    # Generate entries from filtered choices
    yield from module_instance.generate_entries_from_choices(filtered_choices_obj)

def get_registered_archs(workdir: Path) -> list[str]:
    """Get list of registered architectures from workers.db."""
    db_path = workdir / 'workers.db'
    if not db_path.exists():
        sys.exit(f"Error: workers.db not found at {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute("SELECT DISTINCT arch FROM workers ORDER BY arch")
        archs = [row[0] for row in cursor.fetchall()]
        return archs
    finally:
        conn.close()

def get_completed_tasks(module_name: str, module_instance, verbose: bool = False):
    """
    Query PostgreSQL for completed tasks from task_queue.

    Returns a set of task_config tuples (hashable form) that have
    successfully completed (status = 'completed').

    Args:
        module_name: Name of the tuning module (e.g., 'flash')
        module_instance: Module instance with ENTRY_CLASS defining field structure
        verbose: Print debug info

    Raises exception if connection fails - caller should handle errors.
    """
    import psycopg
    from psycopg.rows import dict_row

    # Get PostgreSQL connection parameters
    conn_params = get_db_connection_params()

    # Extract field names once for reuse (avoid repeated metadata access)
    entry_class = module_instance.ENTRY_CLASS
    entry_field_names = tuple(f.name for f in fields(entry_class))

    # Convert task_config to hashable tuple: (arch, field1_val, field2_val, ...)
    def make_hashable(task_config):
        arch = task_config['arch']
        entry_dict = task_config['entry']
        field_values = tuple(entry_dict[fname] for fname in entry_field_names)
        return (arch,) + field_values

    # Connect to PostgreSQL - let exceptions propagate
    conn = psycopg.connect(**conn_params, row_factory=dict_row)

    try:
        with conn.cursor() as cur:
            # Query task_queue for completed tasks for this module
            cur.execute("""
                SELECT task_config
                FROM task_queue
                WHERE status = 'completed'
                  AND module = %s
            """, (module_name,))

            # Extract task_config from each row and convert to hashable tuple
            def extract_config(row):
                task_config = row['task_config']
                # task_config is already a dict from JSONB
                if isinstance(task_config, dict):
                    return make_hashable(task_config)
                return None

            completed_configs = set(filter(None, map(extract_config, cur.fetchall())))

            if verbose:
                print(f"Found {len(completed_configs)} completed tasks for module '{module_name}'")

            return completed_configs

    finally:
        conn.close()

def dispatch_tasks(workdir: Path, module_name: str, module_instance, args):
    """Dispatch tuning tasks to PostgreSQL queue."""
    from .pq.dispatcher import TaskDispatcher
    from dataclasses import asdict

    # Get database connection parameters
    conn_params = get_db_connection_params()

    # Generate filtered entries (don't materialize the list)
    entries_generator = generate_filtered_entries(module_instance, args)

    print(f"Dispatching tasks to architecture(s): {', '.join(args.arch)}")

    # Get completed tasks if --skip_completed is enabled
    completed_configs = set()
    if args.skip_completed:
        print("Querying PostgreSQL for completed tasks...")
        completed_configs = get_completed_tasks(module_name, module_instance, verbose=args.verbose)
        if args.dry_run:
            print(f"{len(completed_configs)=}")
            if completed_configs:
                print(f"Example: {next(iter(completed_configs))}")

    # Extract entry field names once for make_hashable (avoid repeated metadata access)
    entry_class = module_instance.ENTRY_CLASS
    entry_field_names = tuple(f.name for f in fields(entry_class))

    # Convert task_config to hashable tuple for set lookup
    def make_hashable(task_config):
        arch = task_config['arch']
        entry_dict = task_config['entry']
        field_values = tuple(entry_dict[fname] for fname in entry_field_names)
        return (arch,) + field_values

    # Generate task configs
    tty_output = sys.stdin.isatty()
    printed_hw_reasons = set()
    def task_config_gen():
        for entry in entries_generator:
            for arch in args.arch:
                supported, reason = module_instance.validate_hw_feature(arch, entry)
                if not supported:
                    if tty_output:
                        key = (arch, reason)
                        if key not in printed_hw_reasons:
                            printed_hw_reasons.add(key)
                            print(f"Skipping {arch} configurations: {reason}")
                    continue
                task_config = {
                    "arch": arch,
                    "module": module_name,
                    "entry": asdict(entry),
                }
                # Add max_hsaco if specified
                if args.max_hsaco is not None:
                    task_config["max_hsaco"] = {"*": args.max_hsaco}
                yield task_config

    # Collect tasks to dispatch (filter out completed ones)
    tasks_to_dispatch = []
    skipped_count = 0

    for task_config in task_config_gen():
        # Check if this task is already completed
        if args.skip_completed:
            config_key = make_hashable(task_config)
            if config_key in completed_configs:
                skipped_count += 1
                if args.verbose:
                    print(f"Skipping completed task: {task_config['entry']}")
                continue

        # Prepare task for dispatcher
        tasks_to_dispatch.append({
            'arch': task_config['arch'],
            'module': task_config['module'],
            'task_config': task_config,
            'priority': 5  # Default priority
        })

        if args.verbose:
            print(f"Prepared task for {task_config['arch']}: {task_config['entry']}")

    print(f"Prepared {len(tasks_to_dispatch)} tasks for dispatch")
    if args.skip_completed and skipped_count > 0:
        print(f"Skipped {skipped_count} already-completed tasks")

    # Confirmation prompt (skip if -y flag or stdin is not a tty)
    if not args.yes and sys.stdin.isatty():
        for key, value in vars(args).items():
            print(f"  {key}: {value}")
        try:
            response = input(f"Proceed with dispatch? [y/N]: ")
            if response.lower() not in ('y', 'yes'):
                print("Dispatch cancelled")
                return
        except (KeyboardInterrupt, EOFError):
            print("\nDispatch cancelled")
            return

    if args.dry_run:
        print("Dry run mode - tasks not dispatched")
        return

    # Dispatch tasks using bulk INSERT
    dispatcher = TaskDispatcher(conn_params)

    # Ensure partitions exist for all architectures
    for arch in args.arch:
        try:
            dispatcher.ensure_partition(arch)
        except Exception as e:
            print(f"Warning: Failed to ensure partition for {arch}: {e}", file=sys.stderr)

    # Dispatch tasks
    dispatched = dispatcher.dispatch_bulk(tasks_to_dispatch, batch_size=1000)
    print(f"Dispatched {dispatched} tasks to PostgreSQL queue")

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
    Scan v3python/tune/ for packages that export TuneDesc and return a dict
    mapping module name to an instantiated TuneDesc object.
    """
    tune_dir = Path(__file__).parent
    modules = {}
    for init_file in sorted(tune_dir.glob('*/__init__.py')):
        name = init_file.parent.name
        try:
            mod = import_module(f'.{name}', package='v3python.tune')
            if hasattr(mod, 'TuneDesc'):
                modules[name] = mod.TuneDesc()
        except Exception:
            pass
    return modules

def add_common_arguments(parser):
    """Add common arguments (workdir, arch, etc.) to a parser."""
    parser.add_argument('workdir', type=Path,
                        help='Project working directory')
    parser.add_argument('--arch', type=str, nargs='+',
                        help='Target architecture(s). If not specified, uses all registered workers.')
    parser.add_argument('--max_hsaco', type=int, metavar='N',
                        help='Maximum number of hsaco kernels to tune per entry (default: all)')
    parser.add_argument('--skip_completed', action='store_true',
                        help='Query PostgreSQL and skip tasks that have already completed successfully')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print parsed options and exit without dispatching tasks')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='Skip confirmation prompt and proceed with dispatch')

def add_module_subparser(subparsers, module_name, module_instance):
    """
    Add a subparser for a specific tuning module with its parameter choices.

    Args:
        subparsers: The subparsers object from ArgumentParser.add_subparsers()
        module_name: Name of the tuning module (e.g., 'flash')
        module_instance: Already-instantiated TuneDesc object for this module

    Returns:
        The created subparser
    """
    module_parser = subparsers.add_parser(
        module_name,
        help=f'{module_name.capitalize()} tuning module',
        usage=f'%(prog)s <workdir> [options...]',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    add_common_arguments(module_parser)

    all_choices = get_parameter_choices(module_instance)

    # Add dynamic arguments based on module parameters
    for field in fields(all_choices):
        param_name = field.name
        param_choices = getattr(all_choices, param_name)

        # Determine type from first choice
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

        module_parser.add_argument(f'--{param_name}',
                            type=arg_type,
                            nargs='*',
                            default=actual_choices,
                            choices=actual_choices,
                            metavar=metavar,
                            help=f'Choices: {display_choices}')

    return module_parser

def main():
    parser = argparse.ArgumentParser(
        description='Dispatch tuning tasks to PostgreSQL queue (Tuner v3.5)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Dispatch all flash tasks for gfx942
  %(prog)s flash /path/to/workdir --arch gfx942

  # Dispatch only float16 tasks with specific sequence lengths
  %(prog)s flash /path/to/workdir --arch gfx942 --dtype float16 --seqlen_q 128 256 --seqlen_k 128 256

  # Dispatch to multiple architectures
  %(prog)s flash /path/to/workdir --arch gfx942 gfx90a

  # Limit number of hsaco kernels to tune per entry
  %(prog)s flash /path/to/workdir --max_hsaco 5 --dtype float16

  # Skip tasks that have already completed (queries PostgreSQL)
  %(prog)s flash /path/to/workdir --skip_completed --arch gfx942
''')

    # Create subparsers for each module BEFORE parsing
    # Module comes first as a positional argument
    subparsers = parser.add_subparsers(dest='module', required=True,
                                       help='Tuning module')

    available_modules = get_available_modules()
    for module_name, module_instance in available_modules.items():
        add_module_subparser(subparsers, module_name, module_instance)

    # Parse all arguments
    args = parser.parse_args()

    # Validate workdir
    if not args.workdir.is_dir():
        parser.error(f"Working directory does not exist: {args.workdir}")

    # Load config (required for task dispatch)
    load_config(args.workdir)

    # If arch not specified, use all registered architectures
    if args.arch is None:
        args.arch = get_registered_archs(args.workdir)
        if not args.arch:
            parser.error(f"No workers registered in {args.workdir}/workers.db")
        print(f"No --arch specified, using all registered: {', '.join(args.arch)}")

    # Dispatch tasks
    dispatch_tasks(args.workdir, args.module, available_modules[args.module], args)

if __name__ == '__main__':
    main()
