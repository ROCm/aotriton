#!/usr/bin/env python
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Dispatch tuning tasks to Celery distributed framework.

This script:
1. Loads a tuning module (e.g., 'flash')
2. Queries the module's parameter choices
3. Allows filtering via command-line arguments
4. Dispatches tasks to Celery workers
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
            if '=' in line:
                # Simple parsing: KEY=VALUE
                key, value = line.split('=', 1)
                # Remove quotes if present
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value

    # Set AOTRITON_CELERY_WORKDIR for tasks.py
    os.environ['AOTRITON_CELERY_WORKDIR'] = str(workdir.resolve())

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
    Query PostgreSQL for completed tasks.

    Returns a set of task_config tuples (hashable form) that have
    successfully completed (result.brief is an object, not a string).

    Uses JSONB operators to query the hacked Celery result backend where
    result column is JSONB instead of bytea.

    Args:
        module_name: Name of the tuning module (e.g., 'flash')
        module_instance: Module instance with ENTRY_CLASS defining field structure
        verbose: Print debug info

    Requires load_config() to be called first to set environment variables.
    Raises exception if connection fails - caller should handle errors.
    """
    import psycopg
    from psycopg.rows import dict_row

    # Get PostgreSQL connection from environment (set by load_config from config.rc)
    # Database name defaults to username if not specified
    postgres_user = os.environ.get('POSTGRES_USER')
    postgres_password = os.environ.get('POSTGRES_PASSWORD')
    celery_service_host = os.environ.get('CELERY_SERVICE_HOST')
    postgres_port = os.environ.get('POSTGRES_PORT')

    if not all([postgres_user, postgres_password, celery_service_host, postgres_port]):
        raise RuntimeError("Missing PostgreSQL credentials in config.rc. Required: POSTGRES_USER, POSTGRES_PASSWORD, CELERY_SERVICE_HOST, POSTGRES_PORT")

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
    conn = psycopg.connect(
        host=celery_service_host,
        port=int(postgres_port),
        user=postgres_user,
        password=postgres_password,
        # Database defaults to username (PostgreSQL convention)
        row_factory=dict_row
    )

    try:
        with conn.cursor() as cur:
            # Query celery_taskmeta for results with brief field
            # The result column is JSONB
            # Success: brief is an object/dict
            # Failure: brief is a string "Exception raised"
            # Filter by module name in SQL for efficiency
            cur.execute("""
                SELECT result
                FROM celery_taskmeta
                WHERE result IS NOT NULL
                  AND result ? 'brief'
                  AND jsonb_typeof(result->'brief') = 'object'
                  AND result->'task_config'->>'module' = %s
            """, (module_name,))

            # Extract task_config from each row and convert to hashable tuple
            def extract_config(row):
                result_data = row['result']
                result_obj = json.loads(result_data) if isinstance(result_data, str) else result_data

                if isinstance(result_obj, dict) and 'task_config' in result_obj:
                    task_config = result_obj['task_config']
                    return make_hashable(task_config)
                return None

            completed_configs = set(filter(None, map(extract_config, cur.fetchall())))

            if verbose:
                print(f"Found {len(completed_configs)} completed tasks for module '{module_name}'")

            return completed_configs

    finally:
        conn.close()

def dispatch_tasks(workdir: Path, module_name: str, args):
    """Dispatch tuning tasks to Celery."""
    from v3python.celery.tasks import tune_kernel
    from celery.result import allow_join_result
    from dataclasses import asdict

    # Load module
    module_instance = load_module(module_name)

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
            return

    if args.dry_run:
        return

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
    def task_config_gen():
        for entry in entries_generator:
            for arch in args.arch:
                task_config = {
                    "arch": arch,
                    "module": module_name,
                    "entry": asdict(entry),
                }
                # Add max_hsaco if specified
                if args.max_hsaco is not None:
                    task_config["max_hsaco"] = {"*": args.max_hsaco}
                yield task_config

    # Dispatch tasks
    results = []
    task_count = 0
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

        arch = task_config['arch']
        if not args.dry_run:
            res = tune_kernel.apply_async(args=(task_config,), queue=arch)
        results.append(res)
        task_count += 1
        if args.verbose:
            print(f"Dispatched task for {arch}: {task_config['entry']}")

    print(f"Dispatched {task_count} tasks")
    if args.skip_completed and skipped_count > 0:
        print(f"Skipped {skipped_count} already-completed tasks")

    # Wait for results if requested
    if args.wait:
        print("Waiting for tasks to complete...")
        with allow_join_result():
            completed = 0
            for res in results:
                try:
                    result = res.get()
                    completed += 1
                    if args.verbose:
                        print(f"Task {completed}/{len(results)} completed. result:\n{result=}")
                except Exception as e:
                    print(f"Task failed: {e}", file=sys.stderr)
        print(f"Completed {completed}/{len(results)} tasks")

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
    Get list of available tuning modules.

    Scans v3python/tune directory for modules that have a capitalized class.
    """
    # For now, hardcode known modules
    # TODO: Auto-discover by scanning v3python/tune/
    return ['flash']

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
    parser.add_argument('--wait', action='store_true',
                        help='Wait for all tasks to complete')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print parsed options and exit without dispatching tasks')

def add_module_subparser(subparsers, module_name, load_params=True):
    """
    Add a subparser for a specific tuning module with its parameter choices.

    Args:
        subparsers: The subparsers object from ArgumentParser.add_subparsers()
        module_name: Name of the tuning module (e.g., 'flash')
        load_params: If True, load module and add parameter arguments

    Returns:
        The created subparser
    """
    # Create subparser for this module
    module_parser = subparsers.add_parser(
        module_name,
        help=f'{module_name.capitalize()} tuning module',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Add common arguments to this subparser
    add_common_arguments(module_parser)

    if not load_params:
        return module_parser

    # Load module to get parameter choices
    module_instance = load_module(module_name)
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
    # Defensive check: This script should be launched via .celery/dispatch-tasks.sh
    # which sets up venv at scratch/venv.devnode
    venv_path = sys.prefix
    if not venv_path.endswith('scratch/venv.devnode'):
        sys.stderr.write(f"""ERROR: Do not run this script directly.

This script requires a patched celery installation and proper environment setup.
Please use the wrapper script instead:

  .celery/dispatch-tasks.sh <workdir> [options] <module> [module-options]

Example:
  .celery/dispatch-tasks.sh /path/to/workdir flash --dtype float16

Current Python: {sys.executable}
Expected venv suffix: scratch/venv.devnode
Actual prefix: {venv_path}
""")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description='Dispatch tuning tasks to Celery distributed framework',
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
    for module_name in available_modules:
        add_module_subparser(subparsers, module_name, load_params=True)

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

    # Dry run: print options and exit
    if args.dry_run:
        print("Dry run mode - parsed options:")
        for key, value in vars(args).items():
            print(f"  {key}: {value}")

    # Dispatch tasks
    dispatch_tasks(args.workdir, args.module, args)

if __name__ == '__main__':
    main()
