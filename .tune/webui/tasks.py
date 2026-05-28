# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Task execution module for web dashboard.
Provides functions to execute CLI commands and query the database.
"""

import subprocess
import sqlite3
import sys
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
import json
import re
import logging
from flask import current_app

# Global constant for aotriton root directory
AOTRITON_ROOT = Path(__file__).parent.parent.parent.resolve()

# Import tuning architectures configuration
from .config import TUNING_ARCHITECTURES

# Add v3python to import get_db_connection_params
sys.path.insert(0, AOTRITON_ROOT.as_posix())
from v3python.tune.utils import get_db_connection_params
from v3python.tune.flash.module import FlashEntry
from .pytest_entry_parser import parse_pytest_node_id, entry_to_sql_clauses
from v3python.tune.pq.visperf import query_best_results, get_available_archs
from v3python.tune.pq.export_visperf import export_visperf as _do_export_visperf

def run_command(cmd, cwd, workdir, description=None, dry_run: bool = False):
    """
    Execute shell command with per-action tracker

    Args:
        cmd: Command as list of Path/str objects (e.g., [script_path, arg1, arg2])
        cwd: Current working directory for command execution (Path object)
        workdir: Workdir path where logs should be stored (str)
        description: Human-readable description
        dry_run: When True, log the command but do not execute it

    Returns:
        dict with action_id, status, message
    """
    # Convert cmd list to strings
    cmd_parts = [str(p) for p in cmd]
    cmd_str = ' '.join(cmd_parts)

    if dry_run:
        logger.info('[DRY RUN] %s', cmd_str)
        return {
            'action_id': None,
            'status': 'dry_run',
            'message': f'[DRY RUN] Would run: {description or cmd_str}',
        }

    # Get log directory from workdir (use /scratch which is excluded from sync)
    log_dir = Path(workdir) / 'scratch' / 'webui-commands'

    # Create tracker
    tracker = current_app.tracker_registry.create(
        command=cmd_parts,  # Pass as list for subprocess
        description=description or cmd_str,
        cwd=Path(cwd).as_posix(),
        log_dir=log_dir.as_posix()
    )

    # Start execution in background
    tracker.start()

    return {
        'action_id': tracker.action_id,
        'status': 'running',
        'message': f'Command started: {description or cmd_str}'
    }


def get_workers(workdir):
    """Query workers from database"""
    workdir_path = Path(workdir)
    db_path = workdir_path / 'workers.db'
    init_workers_db(workdir)  # Ensure database exists

    conn = sqlite3.connect(db_path.as_posix())
    cursor = conn.cursor()
    cursor.execute("SELECT hostname, arch, COALESCE(workdir_override, '') FROM workers ORDER BY hostname")
    workers = cursor.fetchall()
    conn.close()
    return workers


def get_worker_by_hostname(workdir, hostname):
    """Get single worker info by hostname"""
    workdir_path = Path(workdir)
    db_path = workdir_path / 'workers.db'
    if not db_path.exists():
        return None

    conn = sqlite3.connect(db_path.as_posix())
    cursor = conn.cursor()
    cursor.execute(
        "SELECT hostname, arch, COALESCE(workdir_override, '') FROM workers WHERE hostname = ? LIMIT 1",
        (hostname,)
    )
    worker = cursor.fetchone()
    conn.close()
    return worker


def get_architectures(workdir):
    """Get list of distinct architectures"""
    workdir_path = Path(workdir)
    db_path = workdir_path / 'workers.db'
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path.as_posix())
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT arch FROM workers ORDER BY arch")
    archs = [row[0] for row in cursor.fetchall()]
    conn.close()
    return archs


def get_hostnames(workdir):
    """Get list of distinct hostnames"""
    workdir_path = Path(workdir)
    db_path = workdir_path / 'workers.db'
    if not db_path.exists():
        return []

    conn = sqlite3.connect(db_path.as_posix())
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT hostname FROM workers ORDER BY hostname")
    hostnames = [row[0] for row in cursor.fetchall()]
    conn.close()
    return hostnames


def get_status_summary(workdir):
    """Get dashboard summary status"""
    workers = get_workers(workdir)
    archs = get_architectures(workdir)

    # Get PostgreSQL status
    server_status = get_server_status(workdir)
    postgres_status = 'running' if server_status['running'] else 'stopped'

    return {
        'worker_count': len(workers),
        'architecture_count': len(archs),
        'architectures': archs,
        'postgres_status': postgres_status,
    }


def _merge_progress_rows(progress_rows, speed_rows, stale_rows):
    speed_map = {row['arch']: row['recent_completions'] / 5.0 for row in speed_rows}
    stale_map = {row['arch']: row['stale_count'] for row in stale_rows}
    result = []
    for row in progress_rows:
        data = dict(row)
        data['speed_per_minute'] = speed_map.get(data['arch'], 0.0)
        data['stale'] = stale_map.get(data['arch'], 0)
        cancelled = data.get('cancelled', 0) or 0
        effective_total = data['total'] - cancelled
        data['effective_total'] = effective_total
        data['pct_complete'] = round(
            100.0 * data['completed'] / effective_total, 1
        ) if effective_total > 0 else 0.0
        result.append(data)
    return result


def get_tuning_progress(workdir):
    """Get kernel and op tuning progress using the two queue-progress views."""
    try:
        conn_params = get_db_connection_params(Path(workdir))
        with psycopg.connect(**conn_params, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM kernel_queue_progress ORDER BY arch")
                kernel_rows = cur.fetchall()

                cur.execute("SELECT * FROM op_queue_progress ORDER BY arch")
                op_rows = cur.fetchall()

                cur.execute("""
                    SELECT arch, COUNT(*) as recent_completions
                    FROM task_queue
                    WHERE status = 'completed'
                      AND completed_at > NOW() - INTERVAL '5 minutes'
                      AND module NOT LIKE '%_op'
                    GROUP BY arch
                """)
                kernel_speed_rows = cur.fetchall()

                cur.execute("""
                    SELECT arch, COUNT(*) as recent_completions
                    FROM task_queue
                    WHERE status = 'completed'
                      AND completed_at > NOW() - INTERVAL '5 minutes'
                      AND module LIKE '%_op'
                    GROUP BY arch
                """)
                op_speed_rows = cur.fetchall()

                cur.execute("""
                    SELECT arch, COUNT(*) as stale_count
                    FROM task_queue
                    WHERE status = 'running'
                      AND EXTRACT(EPOCH FROM (NOW() - started_at)) > 7200
                      AND module NOT LIKE '%_op'
                    GROUP BY arch
                """)
                kernel_stale_rows = cur.fetchall()

                cur.execute("""
                    SELECT arch, COUNT(*) as stale_count
                    FROM task_queue
                    WHERE status = 'running'
                      AND EXTRACT(EPOCH FROM (NOW() - started_at)) > 7200
                      AND module LIKE '%_op'
                    GROUP BY arch
                """)
                op_stale_rows = cur.fetchall()

                return {
                    'kernel': _merge_progress_rows(kernel_rows, kernel_speed_rows, kernel_stale_rows),
                    'op': _merge_progress_rows(op_rows, op_speed_rows, op_stale_rows),
                }
    except Exception as e:
        logging.error(f"Failed to get tuning progress: {e}")
        return {'kernel': [], 'op': []}


_TUNE_V3BIS_MARKER = 'TUNE_V3BIS testrun Item: '


def _resolve_pytest_entry(workdir, line: str) -> dict:
    """
    Resolve a pytest node ID to a task_queue id.

    Delegates parsing to pytest_entry_parser.parse_pytest_node_id(), then
    queries task_queue without an arch filter (pytest IDs do not encode arch).
    Returns {'matches': [{'task_id': int, 'arch': str, 'module': str}, ...]} or {'error': str}.
    """
    try:
        entry = parse_pytest_node_id(line)
    except ValueError as e:
        return {'error': str(e)}

    clauses, params = entry_to_sql_clauses(entry)
    sql = (
        'SELECT id, arch, module FROM task_queue WHERE '
        + ' AND '.join(clauses)
        + ' ORDER BY arch, id'
    )
    parsed_desc = ', '.join(f'{k}={v!r}' for k, v in entry.items())

    try:
        conn_params = get_db_connection_params(Path(workdir))
        with psycopg.connect(**conn_params, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        if not rows:
            return {'error': f'No task_queue row found for {parsed_desc}'}
        return {'matches': [{'task_id': r['id'], 'arch': r['arch'], 'module': r['module']} for r in rows]}
    except Exception as e:
        logging.error('_resolve_pytest_entry failed: %s', e)
        return {'error': str(e)}


def resolve_tune_entry(workdir, line: str) -> dict:
    """
    Parse a TUNE_V3BIS testrun line or pytest node ID and return the matching
    task_queue id.

    Accepts:
      - A TUNE_V3BIS testrun Item line (full or payload only)
      - A pytest node ID: path/test_file.py::test_name[params]

    Returns {'task_id': <int>} or {'error': <str>}.
    """
    # Detect pytest node ID format before trying TUNE_V3BIS parsing
    if '::' in line and '[' in line:
        return _resolve_pytest_entry(workdir, line)

    try:
        idx = line.find(_TUNE_V3BIS_MARKER)
        payload = line[idx + len(_TUNE_V3BIS_MARKER):].strip() if idx != -1 else line.strip()

        arch_part, entry_part = payload.split(' ', 1)
        arch = arch_part.split('=', 1)[1]
        entry = FlashEntry.parse_text(entry_part)
    except Exception as e:
        return {'error': f'Failed to parse entry: {e}'}

    from dataclasses import asdict
    d = asdict(entry)

    clauses = ["task_config->>'arch' = %s", "module NOT LIKE %s"]
    params: list = [arch, '%_op']
    for field, value in d.items():
        col = f"task_config->'entry'->>'{field}'"
        if isinstance(value, bool):
            clauses.append(f"({col})::boolean = %s")
        elif isinstance(value, int):
            clauses.append(f"({col})::integer = %s")
        elif isinstance(value, float):
            clauses.append(f"({col})::float = %s")
        else:
            clauses.append(f"{col} = %s")
        params.append(value)

    sql = 'SELECT id FROM task_queue WHERE ' + ' AND '.join(clauses) + ' LIMIT 1'

    try:
        conn_params = get_db_connection_params(Path(workdir))
        with psycopg.connect(**conn_params, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
        if row is None:
            return {'error': f'No task_queue row found for arch={arch} {entry_part}'}
        return {'task_id': row['id']}
    except Exception as e:
        logging.error('resolve_tune_entry failed: %s', e)
        return {'error': str(e)}


def get_debug_task_data(workdir, task_id: int) -> dict:
    """Return all rows related to task_id from every relevant table."""
    try:
        conn_params = get_db_connection_params(Path(workdir))
        with psycopg.connect(**conn_params, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM task_queue WHERE id = %s", (task_id,))
                task = cur.fetchone()

                cur.execute(
                    "SELECT id, task_id, kernel_name, hsaco_index, result,"
                    " result_data, error, gpu_id, created_at FROM tuning_results"
                    " WHERE task_id = %s ORDER BY kernel_name, hsaco_index",
                    (task_id,),
                )
                tuning_results = cur.fetchall()

                cur.execute(
                    "SELECT * FROM best_tuning_results WHERE task_id = %s"
                    " ORDER BY kernel_name",
                    (task_id,),
                )
                best_results = cur.fetchall()

                cur.execute(
                    "SELECT kernel_name, test_case, tensor_name,"
                    " target_fudge_factor, absolute_error"
                    " FROM most_accurate_tuning_results WHERE task_id = %s"
                    " ORDER BY kernel_name, test_case, tensor_name",
                    (task_id,),
                )
                accurate_results = cur.fetchall()

                cur.execute(
                    "SELECT id, op_name, backend_index, result, result_data,"
                    " error, gpu_id, created_at FROM optune_results"
                    " WHERE task_id = %s ORDER BY op_name, backend_index",
                    (task_id,),
                )
                optune_results = cur.fetchall()

                cur.execute(
                    "SELECT op_name, backend_index, median_time, arch, impl_desc, computed_at"
                    " FROM best_optune_results WHERE task_id = %s"
                    " ORDER BY op_name",
                    (task_id,),
                )
                best_optune_results = cur.fetchall()

        return {
            'task': task,
            'tuning_results': tuning_results,
            'best_results': best_results,
            'accurate_results': accurate_results,
            'optune_results': optune_results,
            'best_optune_results': best_optune_results,
        }
    except Exception as e:
        logging.error('Failed to get debug data for task %s: %s', task_id, e)
        return {'error': str(e)}


# Command execution helpers

class CommandBuilder:
    """Base class for building commands"""

    def _run(self, script_relative_path, args, workdir, description, dry_run: bool = False):
        """Execute command with proper paths"""
        cmd = [script_relative_path] + list(args)
        return run_command(cmd, cwd=AOTRITON_ROOT, workdir=workdir, description=description, dry_run=dry_run)


def _build_worker_args(workdir, hostname, options=None):
    """Build argument list for worker commands with optional multi_gpu"""
    extargs = []
    if options and 'multi_gpu' in options:
        multi_gpu = options['multi_gpu']
        extargs.append('--multi_gpu')
        extargs.extend([str(gpu) for gpu in multi_gpu])

    if extargs:
        return [workdir, hostname, '--'] + extargs
    else:
        return [workdir, hostname]


class SingleWorkerCommand(CommandBuilder):
    """Base class for single worker operations"""
    RELATIVE = None  # Subclass must define
    ACTION_NAME = None  # Subclass must define

    def exec(self, workdir, hostname, options=None, extra_args=None, dry_run: bool = False):
        """Execute command with script at RELATIVE path"""
        args = _build_worker_args(workdir, hostname, options)
        if extra_args:
            args = args + list(extra_args)
        return self._run(self.RELATIVE, args, workdir, f'{self.ACTION_NAME} worker {hostname}', dry_run=dry_run)


class StartWorkerCommand(SingleWorkerCommand):
    RELATIVE = '.tune/single/start_worker.sh'
    ACTION_NAME = 'Start'


class StopWorkerCommand(SingleWorkerCommand):
    RELATIVE = '.tune/single/stop_worker.sh'
    ACTION_NAME = 'Stop'


class RestartWorkerCommand(SingleWorkerCommand):
    RELATIVE = '.tune/single/restart_worker.sh'
    ACTION_NAME = 'Restart'


class StopStartWorkerCommand(SingleWorkerCommand):
    RELATIVE = '.tune/single/stopstart_worker.sh'
    ACTION_NAME = 'Stop & start'


class DeployWorkerCommand(SingleWorkerCommand):
    RELATIVE = '.tune/single/sync_workdir.sh'
    ACTION_NAME = 'Deploy to'


class BulkWorkerCommand(CommandBuilder):
    """Base class for bulk worker operations"""
    RELATIVE = '.tune/bin/wkctl'
    ACTION = None  # Subclass must define

    def exec(self, workdir, dry_run: bool = False):
        """Execute wkctl with action"""
        return self._run(self.RELATIVE, [workdir, self.ACTION], workdir, f'{self.ACTION.capitalize()} all workers', dry_run=dry_run)


class StartAllWorkersCommand(BulkWorkerCommand):
    ACTION = 'start'


class StopAllWorkersCommand(BulkWorkerCommand):
    ACTION = 'stop'


class RestartAllWorkersCommand(BulkWorkerCommand):
    ACTION = 'restart'


class ServerCommand(CommandBuilder):
    """Base class for server operations"""
    RELATIVE = '.tune/bin/srvctl'
    ACTION = None  # Subclass must define

    def exec(self, workdir, dry_run: bool = False):
        """Execute srvctl with action"""
        return self._run(self.RELATIVE, [workdir, self.ACTION], workdir, f'{self.ACTION.capitalize()} servers', dry_run=dry_run)


class StartServersCommand(ServerCommand):
    ACTION = 'start'


class StopServersCommand(ServerCommand):
    ACTION = 'stop'


class RestartServersCommand(ServerCommand):
    ACTION = 'restart'


class InitDatabaseCommand(CommandBuilder):
    """Initialize database schema"""
    RELATIVE = '.tune/bin/initdb'

    def exec(self, workdir, dry_run: bool = False):
        """Execute initdb script"""
        return self._run(self.RELATIVE, [workdir], workdir, 'Initialize database schema', dry_run=dry_run)


class RecreateSchemaCommand(CommandBuilder):
    """Recreate database schema (drop all tables first)"""
    RELATIVE = '.tune/bin/initdb'

    def exec(self, workdir, dry_run: bool = False):
        """Execute initdb script with --recreate flag"""
        return self._run(self.RELATIVE, [workdir, '--recreate'], workdir, 'Recreate database schema', dry_run=dry_run)


class ComputeBestResultsCommand(CommandBuilder):
    """Compute best_tuning_results table from raw tuning results"""
    RELATIVE = '.tune/bin/compute_best_results'

    def exec(self, workdir, tuning_mode: str = 'kernel', dry_run: bool = False):
        args = [workdir, '--tuning_mode', tuning_mode]
        return self._run(self.RELATIVE, args, workdir, 'Compute best tuning results', dry_run=dry_run)


class ExportBestResultsCommand(CommandBuilder):
    """Export best results to centralized SQLite database"""
    RELATIVE = '.tune/bin/export_best_results'

    def exec(self, workdir, tuning_mode: str = 'kernel', dry_run: bool = False):
        args = [workdir, '--tuning_mode', tuning_mode]
        return self._run(self.RELATIVE, args, workdir, 'Export best results to centraldb', dry_run=dry_run)


class RecreateMaterializedViewCommand(CommandBuilder):
    """Recreate accuracy table via DROP + CREATE (faster than REFRESH CONCURRENTLY)"""
    RELATIVE = '.tune/bin/recreate_materialized_view'

    def exec(self, workdir, tuning_mode: str = 'kernel', dry_run: bool = False):
        args = [workdir, '--tuning_mode', tuning_mode]
        return self._run(self.RELATIVE, args, workdir, 'Recreate materialized view', dry_run=dry_run)


class UpdateMaterializedViewCommand(CommandBuilder):
    """Incremental upsert of accuracy table for cached task_ids"""
    RELATIVE = '.tune/bin/update_materialized_view'

    def exec(self, workdir, tuning_mode: str = 'kernel', dry_run: bool = False):
        args = [workdir, '--tuning_mode', tuning_mode]
        return self._run(self.RELATIVE, args, workdir, 'Update materialized view (incremental)', dry_run=dry_run)


class SancheckCommand(CommandBuilder):
    """Run LUT sanity check against the exported centralized database"""
    RELATIVE = '.tune/bin/sancheck'

    def exec(self, workdir, tuning_mode: str = 'kernel', dry_run: bool = False):
        args = [workdir, '--tuning_mode', tuning_mode]
        return self._run(self.RELATIVE, args, workdir, 'LUT sanity check', dry_run=dry_run)


class DecomposeDbCommand(CommandBuilder):
    """Decompose centraldb.sqlite3 into per-arch/kernel shards under <workdir>/installed/database/"""
    RELATIVE = '.tune/bin/decomposedb'

    def exec(self, workdir, tuning_mode: str = 'kernel', dry_run: bool = False):
        args = [workdir, '--tuning_mode', tuning_mode]
        return self._run(self.RELATIVE, args, workdir, 'Decompose database', dry_run=dry_run)


class BakeLutCommand(CommandBuilder):
    """Bake LUT: convert raw PG tuning results into the aotriton SQLite DB"""
    RELATIVE = '.tune/bin/bake_lut'

    def exec(self, workdir, extra_args: list | None = None, tuning_mode: str = 'kernel', dry_run: bool = False):
        args = [workdir, '--tuning_mode', tuning_mode] + (extra_args or [])
        label = 'Bake LUT' + (f' ({" ".join(extra_args)})' if extra_args else '')
        return self._run(self.RELATIVE, args, workdir, label, dry_run=dry_run)


class BuildCommand(CommandBuilder):
    """Base class for build operations"""
    RELATIVE = None  # Subclass must define
    DESCRIPTION = None  # Subclass must define

    def exec(self, workdir, dry_run: bool = False):
        """Execute build script"""
        return self._run(self.RELATIVE, [workdir], workdir, self.DESCRIPTION, dry_run=dry_run)


class BuildLibrariesCommand(CommandBuilder):
    """Build tuning version of AOTriton libraries via remotebld (handles local/remote transparently)"""
    RELATIVE = '.tune/bin/remotebld'
    DESCRIPTION = 'Build tuning version of AOTriton libraries'

    def exec(self, workdir, single_arch: str | None = None, dry_run: bool = False):
        args = [workdir]
        if single_arch:
            args += ['--single_arch', single_arch]
        label = f'Build tuning AOTriton libraries ({single_arch})' if single_arch else self.DESCRIPTION
        return self._run(self.RELATIVE, args, workdir, label, dry_run=dry_run)


class BuildTestLibrariesCommand(CommandBuilder):
    """Build testing version of AOTriton libraries inside container via remotebld --test"""
    RELATIVE = '.tune/bin/remotebld'

    def exec(self, workdir, single_arch: str | None = None, use_installed_db: bool = True, dry_run: bool = False):
        args = [workdir, '--test']
        if single_arch:
            args += ['--single_arch', single_arch]
        if not use_installed_db:
            args += ['--source_db']
        label = f'Build testing AOTriton libraries ({single_arch})' if single_arch else 'Build testing AOTriton libraries'
        return self._run(self.RELATIVE, args, workdir, label, dry_run=dry_run)


class BuildImagesCommand(BuildCommand):
    RELATIVE = '.tune/bin/imgbld'
    DESCRIPTION = 'Build Docker images'


class BuildImageOnWorkerCommand(CommandBuilder):
    """Build Docker image on a single worker"""
    RELATIVE = '.tune/single/build_image.sh'

    def exec(self, workdir, hostname, dry_run: bool = False):
        """Execute build_image.sh for a specific worker with --follow for web UI"""
        return self._run(self.RELATIVE, [workdir, hostname, '--follow'], workdir, f'Build image on {hostname}', dry_run=dry_run)


class DeployCommand(CommandBuilder):
    """Base class for deployment operations"""
    RELATIVE = None  # Subclass must define
    DESCRIPTION = None  # Subclass must define

    def exec(self, workdir, dry_run: bool = False):
        """Execute deployment script"""
        return self._run(self.RELATIVE, [workdir], workdir, self.DESCRIPTION, dry_run=dry_run)


class DeployAllCommand(DeployCommand):
    RELATIVE = '.tune/bin/deploy'
    DESCRIPTION = 'Deploy to all workers'

    def exec(self, workdir, tuning_mode: str = 'kernel', dry_run: bool = False):
        return self._run(self.RELATIVE, [workdir, '--tuning_mode', tuning_mode], workdir, self.DESCRIPTION, dry_run=dry_run)


class PrepareWorkdirCommand(DeployCommand):
    RELATIVE = '.tune/bin/prepwkdir'
    DESCRIPTION = 'Prepare workdir'

    def exec(self, workdir, dry_run: bool = False):
        # Ensure log directory exists (use /scratch which is excluded from sync)
        log_dir = Path(workdir) / 'scratch' / 'webui-commands'
        log_dir.mkdir(parents=True, exist_ok=True)

        return super().exec(workdir, dry_run=dry_run)


# Global command instances
_start_worker = StartWorkerCommand()
_stop_worker = StopWorkerCommand()
_restart_worker = RestartWorkerCommand()
_stopstart_worker = StopStartWorkerCommand()
_deploy_worker = DeployWorkerCommand()

_start_all_workers = StartAllWorkersCommand()
_stop_all_workers = StopAllWorkersCommand()
_restart_all_workers = RestartAllWorkersCommand()

_start_servers = StartServersCommand()
_stop_servers = StopServersCommand()
_restart_servers = RestartServersCommand()
_init_database = InitDatabaseCommand()
_recreate_schema = RecreateSchemaCommand()
_compute_best_results = ComputeBestResultsCommand()
_export_best_results = ExportBestResultsCommand()
_recreate_materialized_view = RecreateMaterializedViewCommand()
_update_materialized_view = UpdateMaterializedViewCommand()
_sancheck = SancheckCommand()
_decomposedb = DecomposeDbCommand()
_bake_lut = BakeLutCommand()

_build_libraries = BuildLibrariesCommand()
_build_test_libraries = BuildTestLibrariesCommand()
_build_images = BuildImagesCommand()
_build_image_on_worker = BuildImageOnWorkerCommand()

_deploy_all = DeployAllCommand()
_prepare_workdir = PrepareWorkdirCommand()


# Worker control functions

def start_worker_single(workdir, hostname, options=None, tuning_mode: str = 'kernel', dry_run: bool = False):
    """Start single worker"""
    extra = ['--tuning_mode', tuning_mode]
    return _start_worker.exec(workdir, hostname, options, extra_args=extra, dry_run=dry_run)


def stop_worker_single(workdir, hostname, dry_run: bool = False):
    """Stop single worker"""
    return _stop_worker.exec(workdir, hostname, dry_run=dry_run)


def restart_worker_single(workdir, hostname, options=None, tuning_mode: str = 'kernel', dry_run: bool = False):
    """Restart single worker"""
    extra = ['--tuning_mode', tuning_mode]
    return _restart_worker.exec(workdir, hostname, options, extra_args=extra, dry_run=dry_run)


def stop_start_worker_single(workdir, hostname, options=None, tuning_mode: str = 'kernel', dry_run: bool = False):
    """Stop then start single worker"""
    extra = ['--tuning_mode', tuning_mode]
    return _stopstart_worker.exec(workdir, hostname, options, extra_args=extra, dry_run=dry_run)


def _bulk_worker_action(workdir, action, options=None, tuning_mode: str = 'kernel', dry_run: bool = False):
    """Run a worker action on all Tuner-role hosts."""
    hostnames = get_tuner_hostnames(workdir)
    if not hostnames:
        return {'status': 'ok', 'message': 'No Tuner-role workers configured', 'output': ''}
    results = []
    for hostname in hostnames:
        if action == 'start':
            r = start_worker_single(workdir, hostname, options, tuning_mode=tuning_mode, dry_run=dry_run)
        elif action == 'stop':
            r = stop_worker_single(workdir, hostname, dry_run=dry_run)
        elif action == 'restart':
            r = restart_worker_single(workdir, hostname, options, tuning_mode=tuning_mode, dry_run=dry_run)
        elif action == 'stop-start':
            r = stop_start_worker_single(workdir, hostname, options, tuning_mode=tuning_mode, dry_run=dry_run)
        else:
            r = {'status': 'error', 'message': f'Unknown action: {action}'}
        results.append(f"{hostname}: {r.get('status', 'unknown')}")
    return {'status': 'ok', 'message': '\n'.join(results), 'output': '\n'.join(results)}


def start_all_workers(workdir, tuning_mode: str = 'kernel', dry_run: bool = False):
    """Start all Tuner-role workers"""
    return _bulk_worker_action(workdir, 'start', tuning_mode=tuning_mode, dry_run=dry_run)


def stop_all_workers(workdir, dry_run: bool = False):
    """Stop all Tuner-role workers"""
    return _bulk_worker_action(workdir, 'stop', dry_run=dry_run)


def restart_all_workers(workdir, tuning_mode: str = 'kernel', dry_run: bool = False):
    """Restart all Tuner-role workers"""
    return _bulk_worker_action(workdir, 'restart', tuning_mode=tuning_mode, dry_run=dry_run)


def stop_start_all_workers(workdir, tuning_mode: str = 'kernel', dry_run: bool = False):
    """Stop then start all Tuner-role workers"""
    return _bulk_worker_action(workdir, 'stop-start', tuning_mode=tuning_mode, dry_run=dry_run)


# Server control functions

def start_servers(workdir, dry_run: bool = False):
    """Start servers"""
    return _start_servers.exec(workdir, dry_run=dry_run)


def stop_servers(workdir, dry_run: bool = False):
    """Stop servers"""
    return _stop_servers.exec(workdir, dry_run=dry_run)


def restart_servers(workdir, dry_run: bool = False):
    """Restart servers"""
    return _restart_servers.exec(workdir, dry_run=dry_run)


def init_database(workdir, dry_run: bool = False):
    """Initialize database schema"""
    return _init_database.exec(workdir, dry_run=dry_run)


def recreate_schema(workdir, dry_run: bool = False):
    """Recreate database schema (drop all tables first)"""
    return _recreate_schema.exec(workdir, dry_run=dry_run)


def compute_best_results(workdir, tuning_mode: str = 'kernel', dry_run: bool = False):
    """Compute best_tuning_results table from raw tuning results"""
    return _compute_best_results.exec(workdir, tuning_mode=tuning_mode, dry_run=dry_run)


def export_best_results(workdir, tuning_mode: str = 'kernel', dry_run: bool = False):
    """Export best results to centralized SQLite database"""
    return _export_best_results.exec(workdir, tuning_mode=tuning_mode, dry_run=dry_run)


def recreate_materialized_view(workdir, tuning_mode: str = 'kernel', dry_run: bool = False):
    """Recreate accuracy table via DROP + CREATE"""
    return _recreate_materialized_view.exec(workdir, tuning_mode=tuning_mode, dry_run=dry_run)


def sancheck(workdir, tuning_mode: str = 'kernel', dry_run: bool = False):
    """Run LUT sanity check against the exported centralized database"""
    return _sancheck.exec(workdir, tuning_mode=tuning_mode, dry_run=dry_run)


def bake_lut(workdir, extra_args: list | None = None, tuning_mode: str = 'kernel', dry_run: bool = False):
    """Bake LUT: convert raw PG tuning results into the aotriton SQLite DB.

    extra_args examples: ['--incremental'], ['--fix', 'gpu01:0'], ['--fix', '0']
    """
    return _bake_lut.exec(workdir, extra_args, tuning_mode=tuning_mode, dry_run=dry_run)


def update_materialized_view(workdir, tuning_mode: str = 'kernel', dry_run: bool = False):
    return _update_materialized_view.exec(workdir, tuning_mode=tuning_mode, dry_run=dry_run)


def decomposedb(workdir, tuning_mode: str = 'kernel', dry_run: bool = False):
    """Decompose centraldb.sqlite3 into per-arch/kernel shards"""
    return _decomposedb.exec(workdir, tuning_mode=tuning_mode, dry_run=dry_run)


def get_git_status(workdir):
    """Get git status for workdir/aotriton.src and current directory"""
    aotriton_src = Path(workdir) / 'aotriton.src'
    cwd = Path.cwd()

    script = f'''
    # aotriton.src HEAD commit
    echo $(cd {aotriton_src} && git rev-parse --short=12 HEAD 2>/dev/null || echo unknown)
    # current directory HEAD commit
    echo $(cd {cwd} && git rev-parse --short=12 HEAD 2>/dev/null || echo not-a-git-repo)
    # check if working tree is dirty (1=dirty, 0=clean)
    cd {cwd} && git diff --quiet HEAD && git diff --cached --quiet 2>/dev/null; echo $?
    # check if any remote URL ends with aotriton or aotriton.git (0=yes, 1=no)
    cd {cwd} && git config --get-regexp 'remote\\..*\\.url' 2>/dev/null | cut -d' ' -f2- | grep -qE 'aotriton(\\.git)?$'; echo $?
    '''

    result = subprocess.run(script, shell=True, capture_output=True, text=True, executable='/bin/bash')
    lines = result.stdout.strip().split('\n')

    return {
        'aotriton_src_head': lines[0],
        'cwd_head': lines[1],
        'cwd_dirty': lines[2] == '1',
        'cwd_is_aotriton': lines[3] == '0'
    }


def get_config_vars(workdir):
    """Parse all variables from config.rc"""
    config_rc = Path(workdir) / 'config.rc'

    if not config_rc.exists():
        return {}

    # Parse keys from config.rc using pattern ^\w+=
    keys = []
    for line in config_rc.read_text().splitlines():
        line = line.strip()
        # Match lines like VAR= or VAR_NAME=
        match = re.match(r'^(\w+)=', line)
        if match:
            keys.append(match.group(1))

    if not keys:
        return {}

    # Source config.rc and echo all found keys
    echo_cmds = ' && '.join([f'echo "${key}"' for key in keys])
    result = subprocess.run(
        f'. {config_rc} && {echo_cmds}',
        shell=True,
        capture_output=True,
        text=True,
        executable='/bin/bash'
    )

    if result.returncode != 0:
        return {}

    # Pair keys with values
    values = result.stdout.strip().split('\n')
    config_vars = {}
    for key, value in zip(keys, values):
        config_vars[key] = value

    return dict(sorted(config_vars.items()))


def get_server_status(workdir):
    """Get server status (cheap, suitable for polling)"""
    pidf = Path(workdir) / 'run' / 'container.pids'

    if not pidf.exists():
        return {
            'status': 'stopped',
            'running': False,
            'container_name': None
        }

    container_id = pidf.read_text().strip()

    if not container_id:
        return {
            'status': 'stopped',
            'running': False,
            'container_name': None
        }

    # Fast check using docker ps --filter (cheap for polling)
    result = subprocess.run(
        ['docker', 'ps', '--no-trunc', '--quiet', '--filter', f'id={container_id}'],
        capture_output=True,
        text=True
    )

    running = bool(result.stdout.strip())

    # Get container name (only if running)
    container_name = None
    if running:
        name_result = subprocess.run(
            ['docker', 'ps', '--no-trunc', '--filter', f'id={container_id}', '--format', '{{.Names}}'],
            capture_output=True,
            text=True
        )
        container_name = name_result.stdout.strip()

    return {
        'status': 'running' if running else 'stopped',
        'running': running,
        'container_name': container_name
    }


# Build functions

def sync_build_node(workdir, dry_run: bool = False):
    """Sync local workdir to the remote build node"""
    cfg = get_build_node_config(workdir)
    hostname = cfg.get('hostname', '')
    if not hostname:
        return {'status': 'error', 'message': 'Remote build node hostname is not configured'}
    cmd = ['.tune/single/sync_workdir.sh', workdir, hostname, '--buildnode']
    return run_command(cmd, cwd=AOTRITON_ROOT, workdir=workdir,
                       description=f'Sync workdir to remote build node {hostname}', dry_run=dry_run)


def build_libraries(workdir, single_arch: str | None = None, dry_run: bool = False):
    """Build tuning version of AOTriton libraries (all arches or one)."""
    return _build_libraries.exec(workdir, single_arch, dry_run=dry_run)


def build_test_libraries(workdir, single_arch: str | None = None, use_installed_db: bool = True, dry_run: bool = False):
    """Build testing version of AOTriton libraries inside container (all arches or one)."""
    return _build_test_libraries.exec(workdir, single_arch, use_installed_db=use_installed_db, dry_run=dry_run)


def fetch_tuning_build(workdir, dry_run: bool = False):
    """Fetch tuning build artifacts from remote build node"""
    cfg = get_build_node_config(workdir)
    hostname = cfg.get('hostname', '')
    if not hostname:
        return {'status': 'error', 'message': 'Remote build node hostname is not configured'}
    cmd = ['.tune/bin/fetchbuild', workdir, '--tuning']
    return run_command(cmd, cwd=AOTRITON_ROOT, workdir=workdir,
                       description=f'Fetch tuning build from {hostname}', dry_run=dry_run)


def fetch_test_build(workdir, dry_run: bool = False):
    """Fetch test build artifacts from remote build node"""
    cfg = get_build_node_config(workdir)
    hostname = cfg.get('hostname', '')
    if not hostname:
        return {'status': 'error', 'message': 'Remote build node hostname is not configured'}
    cmd = ['.tune/bin/fetchbuild', workdir, '--test']
    return run_command(cmd, cwd=AOTRITON_ROOT, workdir=workdir,
                       description=f'Fetch test build from {hostname}', dry_run=dry_run)


def build_images(workdir, dry_run: bool = False):
    """Build Docker images"""
    return _build_images.exec(workdir, dry_run=dry_run)


def build_image_on_worker(workdir, hostname, dry_run: bool = False):
    """Build Docker image on specific worker"""
    return _build_image_on_worker.exec(workdir, hostname, dry_run=dry_run)


# Deploy functions

def deploy_workdir(workdir, tuning_mode: str = 'kernel', dry_run: bool = False):
    """Deploy to all workers"""
    return _deploy_all.exec(workdir, tuning_mode=tuning_mode, dry_run=dry_run)


def deploy_workdir_single(workdir, hostname, extra_args=None, tuning_mode: str = 'kernel', dry_run: bool = False):
    """Deploy to single worker"""
    testnode_args = ['--testnode'] if tuning_mode == 'op' else []
    combined = testnode_args + list(extra_args or [])
    return _deploy_worker.exec(workdir, hostname, extra_args=combined or None, dry_run=dry_run)


def prepare_workdir(workdir, dry_run: bool = False):
    """Prepare workdir"""
    return _prepare_workdir.exec(workdir, dry_run=dry_run)


# Worker management functions

def init_workers_db(workdir):
    """Initialize workers database schema if it doesn't exist"""
    workdir_path = Path(workdir)
    db_path = workdir_path / 'workers.db'
    with sqlite3.connect(db_path.as_posix()) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS workers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hostname TEXT NOT NULL UNIQUE,
                arch TEXT NOT NULL,
                workdir_override TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TRIGGER IF NOT EXISTS update_worker_timestamp
            AFTER UPDATE ON workers FOR EACH ROW
            BEGIN
                UPDATE workers SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
            END;

            CREATE TRIGGER IF NOT EXISTS update_config_timestamp
            AFTER UPDATE ON config FOR EACH ROW
            BEGIN
                UPDATE config SET updated_at = CURRENT_TIMESTAMP WHERE key = OLD.key;
            END;

            CREATE TABLE IF NOT EXISTS slurm_batch (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gres TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TRIGGER IF NOT EXISTS update_slurm_batch_timestamp
            AFTER UPDATE ON slurm_batch FOR EACH ROW
            BEGIN
                UPDATE slurm_batch SET updated_at = CURRENT_TIMESTAMP WHERE id = OLD.id;
            END;

            CREATE TABLE IF NOT EXISTS slurm_bad_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hostname TEXT NOT NULL UNIQUE,
                reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS roles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            INSERT OR IGNORE INTO roles (name) VALUES ('Tuner'), ('Builder'), ('Tester');

            CREATE TABLE IF NOT EXISTS worker_roles (
                hostname TEXT NOT NULL,
                role_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (hostname, role_name)
            );
        """)


def get_supported_architectures():
    """Get list of supported GPU architectures for tuning"""
    return TUNING_ARCHITECTURES


def get_workers_by_architecture(workdir):
    """Get workers grouped by architecture"""
    workers = get_workers(workdir)

    # Load GPU metadata, selection, and status from config table
    workdir_path = Path(workdir)
    db_path = workdir_path / 'workers.db'
    gpu_info_map = {}

    init_workers_db(workdir)  # Ensure database exists
    with sqlite3.connect(db_path.as_posix()) as conn:
        cursor = conn.cursor()
        # Fetch all tuner-role hostnames
        cursor.execute("SELECT hostname FROM worker_roles WHERE role_name = 'Tuner'")
        tuner_set = {row[0] for row in cursor.fetchall()}
        # Fetch all GPU metadata, selections, and status
        cursor.execute("SELECT key, value FROM config WHERE key LIKE '%::gpu::%' OR key LIKE '%::gpu_selection' OR key LIKE '%::status'")
        for key, value in cursor.fetchall():
            if '::gpu_selection' in key:
                # Handle gpu_selection: hostname::gpu_selection
                hostname = key.replace('::gpu_selection', '')
                if hostname not in gpu_info_map:
                    gpu_info_map[hostname] = {}
                # Parse "0,1,2" or "-1" to list
                if value == "-1":
                    gpu_info_map[hostname]['selection'] = [-1]
                else:
                    gpu_info_map[hostname]['selection'] = [int(gid) for gid in value.split(',')]
            elif '::status' in key:
                # Handle cached status: hostname::status
                hostname = key.replace('::status', '')
                if hostname not in gpu_info_map:
                    gpu_info_map[hostname] = {}
                try:
                    gpu_info_map[hostname]['status'] = json.loads(value)
                except:
                    gpu_info_map[hostname]['status'] = {'status': 'unknown', 'display': 'Unknown'}
            else:
                # Handle gpu metadata: hostname::gpu::field
                parts = key.split('::')
                if len(parts) == 3:
                    hostname, _, field = parts
                    if hostname not in gpu_info_map:
                        gpu_info_map[hostname] = {}
                    gpu_info_map[hostname][field] = value

    # Group workers by architecture
    workers_by_arch = {arch: [] for arch in TUNING_ARCHITECTURES}

    for hostname, arch, workdir_override in workers:
        if arch in workers_by_arch:
            gpu_info = gpu_info_map.get(hostname, {})
            status = gpu_info.get('status', {'status': 'unknown', 'display': 'Unknown'})
            workers_by_arch[arch].append({
                'hostname': hostname,
                'arch': arch,
                'workdir': workdir_override or '',
                'gpu_arch': gpu_info.get('arch', ''),
                'gpu_pciid': gpu_info.get('pciid', ''),
                'gpu_number': gpu_info.get('number', ''),
                'gpu_selection': gpu_info.get('selection'),  # None if not set, [-1] for all, or [0,1,2,...]
                'status_display': status.get('display', 'Unknown'),
                'is_tuner': hostname in tuner_set,
            })

    return workers_by_arch


def add_worker(workdir, hostname, arch, workdir_override=None):
    """Add a worker to the database"""
    # Validate architecture
    if arch not in TUNING_ARCHITECTURES:
        return {
            'success': False,
            'error': f"Unsupported architecture '{arch}'. Supported: {', '.join(get_supported_architectures())}"
        }

    init_workers_db(workdir)  # Ensure database exists
    workdir_path = Path(workdir)
    db_path = workdir_path / 'workers.db'
    try:
        with sqlite3.connect(db_path.as_posix()) as conn:
            conn.execute(
                "INSERT INTO workers (hostname, arch, workdir_override) VALUES (?, ?, ?)",
                (hostname, arch, workdir_override if workdir_override else None)
            )
        return {'success': True, 'message': f"Worker '{hostname}' added successfully"}
    except sqlite3.IntegrityError:
        return {'success': False, 'error': f"Worker '{hostname}' already exists"}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def remove_worker(workdir, hostname):
    """Remove a worker from the database"""
    init_workers_db(workdir)  # Ensure database exists
    workdir_path = Path(workdir)
    db_path = workdir_path / 'workers.db'
    try:
        with sqlite3.connect(db_path.as_posix()) as conn:
            cursor = conn.execute("DELETE FROM workers WHERE hostname = ?", (hostname,))
            if cursor.rowcount == 0:
                return {'success': False, 'error': f"Worker '{hostname}' not found"}
        return {'success': True, 'message': f"Worker '{hostname}' removed successfully"}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def update_worker_workdir(workdir, hostname, workdir_override):
    """Update a worker's custom workdir"""
    init_workers_db(workdir)  # Ensure database exists
    workdir_path = Path(workdir)
    db_path = workdir_path / 'workers.db'
    try:
        with sqlite3.connect(db_path.as_posix()) as conn:
            cursor = conn.execute(
                "UPDATE workers SET workdir_override = ? WHERE hostname = ?",
                (workdir_override if workdir_override else None, hostname)
            )
            if cursor.rowcount == 0:
                return {'success': False, 'error': f"Worker '{hostname}' not found"}
        return {'success': True, 'message': f"Worker '{hostname}' workdir updated successfully"}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def get_worker_role(workdir, hostname, role):
    """Return True if hostname has the given role."""
    init_workers_db(workdir)
    db_path = Path(workdir) / 'workers.db'
    with sqlite3.connect(db_path.as_posix()) as conn:
        cursor = conn.execute(
            "SELECT 1 FROM worker_roles WHERE hostname = ? AND role_name = ?",
            (hostname, role)
        )
        return cursor.fetchone() is not None


def set_worker_role(workdir, hostname, role, enabled):
    """Add or remove a role association for a hostname."""
    init_workers_db(workdir)
    db_path = Path(workdir) / 'workers.db'
    try:
        with sqlite3.connect(db_path.as_posix()) as conn:
            if enabled:
                conn.execute(
                    "INSERT OR IGNORE INTO worker_roles (hostname, role_name) VALUES (?, ?)",
                    (hostname, role)
                )
            else:
                conn.execute(
                    "DELETE FROM worker_roles WHERE hostname = ? AND role_name = ?",
                    (hostname, role)
                )
        return {'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def get_tuner_hostnames(workdir):
    """Return list of hostnames that have the Tuner role."""
    init_workers_db(workdir)
    db_path = Path(workdir) / 'workers.db'
    with sqlite3.connect(db_path.as_posix()) as conn:
        cursor = conn.execute(
            "SELECT hostname FROM worker_roles WHERE role_name = 'Tuner' ORDER BY hostname"
        )
        return [row[0] for row in cursor.fetchall()]


def get_tester_hostnames(workdir):
    """Return list of hostnames that have the Tester role."""
    init_workers_db(workdir)
    db_path = Path(workdir) / 'workers.db'
    with sqlite3.connect(db_path.as_posix()) as conn:
        cursor = conn.execute(
            "SELECT hostname FROM worker_roles WHERE role_name = 'Tester' ORDER BY hostname"
        )
        return [row[0] for row in cursor.fetchall()]


def get_workers_for_testing(workdir):
    """Return all workers with is_tester flag and their workdir, for the Testing tab."""
    workers = get_workers(workdir)
    init_workers_db(workdir)
    db_path = Path(workdir) / 'workers.db'
    with sqlite3.connect(db_path.as_posix()) as conn:
        cursor = conn.execute(
            "SELECT hostname FROM worker_roles WHERE role_name = 'Tester'"
        )
        tester_set = {row[0] for row in cursor.fetchall()}
    result = []
    for hostname, arch, workdir_override in workers:
        result.append({
            'hostname': hostname,
            'arch': arch,
            'workdir': workdir_override or '',
            'is_tester': hostname in tester_set,
        })
    return result


def get_tester_signature(workdir, hostname):
    """Read the __signature__ file from installed/test/<arch>/ on a remote tester host."""
    worker = get_worker_by_hostname(workdir, hostname)
    if not worker:
        return {'status': 'error', 'message': f"Worker '{hostname}' not found"}
    _, arch, workdir_override = worker
    default_wd = get_default_workdir(workdir) or workdir
    remote_wd = workdir_override or default_wd
    sig_glob = f'{remote_wd}/installed/test/{arch}/lib/aotriton.images/*/__signature__'
    script = f'f=$(ls {sig_glob} 2>/dev/null | head -1); [ -n "$f" ] && cat "$f" || echo "__NOT_FOUND__"'
    result = subprocess.run(
        ['ssh', hostname, script],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return {'status': 'error', 'message': result.stderr.strip() or '(ssh error)'}
    content = result.stdout
    if content.strip() == '__NOT_FOUND__':
        return {'status': 'error', 'message': '(no __signature__ file found)'}
    return {'status': 'ok', 'content': content}



def run_test_on_host(workdir, hostname, pass_num, test_level, backend, variant=None, adiff: bool = False, dry_run: bool = False):
    """Queue run-test on a remote tester host via .tune/single/run-test.sh (tsp-backed).

    When adiff is True the remote script is switched to .ci/run-ci-test.sh via
    the --adiff flag, which reads adiff.txt to select the tests to run.
    """
    worker = get_worker_by_hostname(workdir, hostname)
    if not worker:
        return {'status': 'error', 'message': f"Worker '{hostname}' not found"}
    _, arch, workdir_override = worker
    cmd = [
        '.tune/single/run-test.sh',
        '--workdir', workdir,
        '--hostname', hostname,
        '--arch', arch,
        '--pass', str(pass_num),
        '--test_level', str(test_level),
        '--backend', backend,
    ]
    desc = f'run-test on {hostname} ({arch}) pass={pass_num} level={test_level} backend={backend}'
    if workdir_override:
        cmd += ['--workdir_override', workdir_override]
    if variant in ('partial', 'partial_adiffs'):
        cmd += ['--variant', variant]
        desc += f' variant={variant}'
    if adiff:
        cmd += ['--adiff']
        desc += ' adiff=1'
    return run_command(cmd, cwd=AOTRITON_ROOT, workdir=workdir, description=desc, dry_run=dry_run)


def get_failed_tests(workdir, hostname, pass_num, backend, variant=None):
    """Grep '^FAILED' lines (excluding OutOfMemoryError) from the test output file on a tester host."""
    worker = get_worker_by_hostname(workdir, hostname)
    if not worker:
        return {'status': 'error', 'message': f"Worker '{hostname}' not found"}
    _, _, workdir_override = worker
    default_wd = get_default_workdir(workdir) or workdir
    remote_wd = workdir_override or default_wd
    prefix_map = {'split': 'ut_pass', 'fused': 'fused_pass', 'aiter': 'aiter_pass', 'v3': 'oput_pass'}
    prefix = prefix_map.get(backend, 'ut_pass')
    output_dir = f'{remote_wd}/run/tests'
    if variant == 'partial':
        output_dir += '/partial'
    fname = f'{prefix}{pass_num}.out'
    remote_path = f'{output_dir}/{fname}'
    script = (
        f'if [ -f {remote_path} ]; then '
        f'grep \'^FAILED\' {remote_path} | grep -v OutOfMemoryError; '
        f'else echo "__NOT_FOUND__"; fi'
    )
    r = subprocess.run(
        ['ssh', hostname, script],
        capture_output=True, text=True
    )
    raw = r.stdout
    if raw.strip() == '__NOT_FOUND__':
        return {'status': 'not_found', 'filename': fname}
    lines = [line for line in raw.splitlines() if line.strip()]
    return {'status': 'ok', 'filename': fname, 'lines': lines}


def get_tail_output(workdir, hostname, pass_num, backend, variant=None):
    """Fetch tail -n 5 of test output files for the given pass and backend."""
    worker = get_worker_by_hostname(workdir, hostname)
    if not worker:
        return {'status': 'error', 'message': f"Worker '{hostname}' not found"}
    _, _, workdir_override = worker
    default_wd = get_default_workdir(workdir) or workdir
    remote_wd = workdir_override or default_wd
    prefix_map = {'split': 'ut_pass', 'fused': 'fused_pass', 'aiter': 'aiter_pass', 'v3': 'oput_pass'}
    prefix = prefix_map.get(backend, 'ut_pass')
    output_dir = f'{remote_wd}/run/tests'
    if variant == 'partial':
        output_dir += '/partial'
    results = {}
    for suffix in ('', '.varlen'):
        fname = f'{prefix}{pass_num}{suffix}.out'
        r = subprocess.run(
            ['ssh', hostname, f'tail -n 5 {output_dir}/{fname} 2>/dev/null || echo "(not found)"'],
            capture_output=True, text=True
        )
        results[fname] = r.stdout
    return {'status': 'ok', 'files': results}


def get_adiffs_file(workdir, hostname, arch):
    """SSH-read <remote_workdir>/run/tests/partial/adiffs.txt and save it locally.

    Returns dict with keys:
      status: 'ok' | 'not_found' | 'error'
      content: file text (when status == 'ok')
      local_path: where the file was saved (when status == 'ok')
      message: error description (when status != 'ok')
    """
    worker = get_worker_by_hostname(workdir, hostname)
    if not worker:
        return {'status': 'error', 'message': f"Worker '{hostname}' not found"}
    _, _, workdir_override = worker
    default_wd = get_default_workdir(workdir) or workdir
    remote_wd = workdir_override or default_wd
    remote_path = f'{remote_wd}/run/tests/partial/adiffs.txt'
    script = (
        f'if [ -f {remote_path} ]; then sort -u {remote_path}; '
        f'else echo "__NOT_FOUND__"; fi'
    )
    r = subprocess.run(
        ['ssh', hostname, script],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        return {'status': 'error', 'message': r.stderr.strip() or '(ssh error)'}
    content = r.stdout
    if content.strip() == '__NOT_FOUND__':
        return {'status': 'not_found', 'message': f'{remote_path} not found on {hostname}'}
    # Save locally
    local_dir = Path(workdir) / 'installed' / 'adiffs'
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / f'{arch}.txt'
    local_path.write_text(content)
    return {'status': 'ok', 'content': content, 'local_path': local_path.as_posix()}


def get_default_workdir(workdir):
    """Get the default working directory from config table"""
    init_workers_db(workdir)  # Ensure database exists
    workdir_path = Path(workdir)
    db_path = workdir_path / 'workers.db'

    try:
        with sqlite3.connect(db_path.as_posix()) as conn:
            cursor = conn.execute("SELECT value FROM config WHERE key = 'default_workdir'")
            row = cursor.fetchone()
            return row[0] if row else None
    except Exception:
        return None


def set_default_workdir(workdir, path):
    """Set the default working directory in config table"""
    init_workers_db(workdir)  # Ensure database exists
    workdir_path = Path(workdir)
    db_path = workdir_path / 'workers.db'
    try:
        with sqlite3.connect(db_path.as_posix()) as conn:
            conn.execute("""
                INSERT INTO config (key, value) VALUES ('default_workdir', ?)
                ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = CURRENT_TIMESTAMP
            """, (path, path))
        return {'success': True, 'message': f"Default working directory set to: {path}"}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def get_build_node_config(workdir):
    """Get build node configuration from workers.db config table"""
    init_workers_db(workdir)
    workdir_path = Path(workdir)
    db_path = workdir_path / 'workers.db'
    try:
        with sqlite3.connect(db_path.as_posix()) as conn:
            cursor = conn.execute(
                "SELECT key, value FROM config WHERE key LIKE 'buildnode::%'"
            )
            rows = {row[0]: row[1] for row in cursor.fetchall()}
        return {
            'hostname': rows.get('buildnode::hostname', ''),
            'workdir_override': rows.get('buildnode::workdir_override', ''),
            'enabled': rows.get('buildnode::enable', '0') == '1',
        }
    except Exception:
        return {'hostname': '', 'workdir_override': '', 'enabled': False}


def get_test_build_use_installed_db(workdir) -> bool:
    """Get whether test builds use installed/database/ (True by default)."""
    init_workers_db(workdir)
    workdir_path = Path(workdir)
    db_path = workdir_path / 'workers.db'
    try:
        with sqlite3.connect(db_path.as_posix()) as conn:
            cursor = conn.execute(
                "SELECT value FROM config WHERE key = 'webui::test_build_use_installed_db'"
            )
            row = cursor.fetchone()
            return row[0] != '0' if row else True
    except Exception:
        return True


def set_test_build_use_installed_db(workdir, enabled: bool):
    """Set whether test builds use installed/database/ in workers.db."""
    value = '1' if enabled else '0'
    init_workers_db(workdir)
    workdir_path = Path(workdir)
    db_path = workdir_path / 'workers.db'
    try:
        with sqlite3.connect(db_path.as_posix()) as conn:
            conn.execute("""
                INSERT INTO config (key, value) VALUES ('webui::test_build_use_installed_db', ?)
                ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = CURRENT_TIMESTAMP
            """, (value, value))
        return {'success': True, 'message': f"Test build use installed db set to: {enabled}"}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def get_tuning_mode(workdir) -> str:
    """Get the WebUI tuning mode from workers.db; defaults to 'kernel'."""
    init_workers_db(workdir)
    workdir_path = Path(workdir)
    db_path = workdir_path / 'workers.db'
    try:
        with sqlite3.connect(db_path.as_posix()) as conn:
            cursor = conn.execute("SELECT value FROM config WHERE key = 'webui::tuning_mode'")
            row = cursor.fetchone()
            return row[0] if row else 'kernel'
    except Exception:
        return 'kernel'


def set_tuning_mode(workdir, mode: str):
    """Set the WebUI tuning mode in workers.db. mode must be 'kernel' or 'op'."""
    if mode not in ('kernel', 'op'):
        return {'success': False, 'error': f"Invalid tuning mode: {mode!r}"}
    init_workers_db(workdir)
    workdir_path = Path(workdir)
    db_path = workdir_path / 'workers.db'
    try:
        with sqlite3.connect(db_path.as_posix()) as conn:
            conn.execute("""
                INSERT INTO config (key, value) VALUES ('webui::tuning_mode', ?)
                ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = CURRENT_TIMESTAMP
            """, (mode, mode))
        return {'success': True, 'message': f"Tuning mode set to: {mode}"}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def set_build_node_config(workdir, hostname, workdir_override, enabled):
    """Save build node configuration to workers.db config table"""
    init_workers_db(workdir)
    workdir_path = Path(workdir)
    db_path = workdir_path / 'workers.db'
    upsert = """
        INSERT INTO config (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = CURRENT_TIMESTAMP
    """
    try:
        with sqlite3.connect(db_path.as_posix()) as conn:
            conn.execute(upsert, ('buildnode::hostname', hostname, hostname))
            conn.execute(upsert, ('buildnode::workdir_override', workdir_override, workdir_override))
            enable_val = '1' if enabled else '0'
            conn.execute(upsert, ('buildnode::enable', enable_val, enable_val))
        return {'success': True, 'message': 'Build node configuration saved'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def detect_gpu_for_worker(workdir, hostname, dry_run: bool = False):
    """Detect GPU metadata for a specific worker"""
    cmd = ['.tune/single/detect_gpu.sh', workdir, hostname]
    return run_command(cmd, cwd=AOTRITON_ROOT, workdir=workdir, description=f'Detect GPU info for {hostname}', dry_run=dry_run)


def save_worker_gpu_selection(workdir, hostname, gpu_ids):
    """Save GPU selection for a worker to config table"""
    init_workers_db(workdir)
    workdir_path = Path(workdir)
    db_path = workdir_path / 'workers.db'

    # Convert list to comma-separated string
    # [-1] means all GPUs, otherwise list of GPU IDs like [0,1,2,3]
    if gpu_ids == [-1]:
        value = "-1"
    else:
        value = ",".join(str(gid) for gid in gpu_ids)

    try:
        with sqlite3.connect(db_path.as_posix()) as conn:
            conn.execute("""
                INSERT INTO config (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = CURRENT_TIMESTAMP
            """, (f'{hostname}::gpu_selection', value, value))
        return {'success': True, 'message': f'GPU selection saved for {hostname}'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def get_worker_gpu_selection(workdir, hostname):
    """Get saved GPU selection for a worker"""
    init_workers_db(workdir)
    workdir_path = Path(workdir)
    db_path = workdir_path / 'workers.db'

    try:
        with sqlite3.connect(db_path.as_posix()) as conn:
            cursor = conn.execute(
                "SELECT value FROM config WHERE key = ?",
                (f'{hostname}::gpu_selection',)
            )
            row = cursor.fetchone()
            if row:
                value = row[0]
                if value == "-1":
                    return [-1]
                else:
                    return [int(gid) for gid in value.split(',')]
            return None
    except Exception:
        return None


def get_worker_status_single(workdir, hostname):
    """Get worker status (container ID and GPU process count)"""
    script_path = AOTRITON_ROOT / '.tune/single/get_worker_status.sh'
    result = subprocess.run(
        [script_path.as_posix(), workdir, hostname],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return {'status': 'error', 'display': 'Error'}

    # Output is already in human-readable format
    # "Stopped" or "pod: <id>; ngproc: <counts>"
    display = result.stdout.strip()

    if display == 'Stopped':
        return {'status': 'stopped', 'display': 'Stopped'}
    elif display.startswith('pod:'):
        return {'status': 'running', 'display': display}
    else:
        return {'status': 'unknown', 'display': display if display else 'Unknown'}


def save_worker_status(workdir, hostname, status_data):
    """Save worker status to config table for caching"""
    init_workers_db(workdir)
    workdir_path = Path(workdir)
    db_path = workdir_path / 'workers.db'

    # Store as JSON string
    value = json.dumps(status_data)

    try:
        with sqlite3.connect(db_path.as_posix()) as conn:
            conn.execute("""
                INSERT INTO config (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = CURRENT_TIMESTAMP
            """, (f'{hostname}::status', value, value))
    except Exception:
        pass  # Ignore errors in caching


def get_cached_worker_status(workdir, hostname):
    """Get cached worker status from config table"""
    init_workers_db(workdir)
    workdir_path = Path(workdir)
    db_path = workdir_path / 'workers.db'

    try:
        with sqlite3.connect(db_path.as_posix()) as conn:
            cursor = conn.execute(
                "SELECT value FROM config WHERE key = ?",
                (f'{hostname}::status',)
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            return None
    except Exception:
        return None


def probe_worker_status(workdir, hostname):
    """Probe worker status and cache result"""
    status_data = get_worker_status_single(workdir, hostname)
    save_worker_status(workdir, hostname, status_data)
    return status_data


def probe_all_workers_status(workdir):
    """Probe status for all registered workers and save to database"""
    workers = get_workers(workdir)

    results = []
    for hostname, arch, workdir_override in workers:
        try:
            status_data = probe_worker_status(workdir, hostname)
            results.append({
                'hostname': hostname,
                'success': True,
                'status': status_data
            })
        except Exception as e:
            logger.error(f"Failed to probe status for {hostname}: {e}")
            results.append({
                'hostname': hostname,
                'success': False,
                'error': str(e)
            })

    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)

    return {
        'success': success_count == total_count,
        'message': f'Probed {success_count}/{total_count} workers successfully',
        'results': results
    }


# Schedule management functions

def get_worker_schedule(workdir, hostname):
    """Get schedule configuration for a worker"""
    init_workers_db(workdir)
    workdir_path = Path(workdir)
    db_path = workdir_path / 'workers.db'

    try:
        with sqlite3.connect(db_path.as_posix()) as conn:
            cursor = conn.execute("""
                SELECT key, value FROM config
                WHERE key LIKE ? || '::schedule::%'
            """, (hostname,))

            schedule = {}
            for row in cursor:
                key = row[0]
                # key format: hostname::schedule::field
                field = key.split('::')[-1]
                schedule[field] = row[1]

            return schedule if schedule else None
    except Exception as e:
        logger.error(f"Error getting schedule for {hostname}: {e}")
        return None


def save_worker_schedule(workdir, hostname, schedule_data):
    """Save schedule configuration for a worker"""
    init_workers_db(workdir)
    workdir_path = Path(workdir)
    db_path = workdir_path / 'workers.db'

    try:
        with sqlite3.connect(db_path.as_posix()) as conn:
            # Save each schedule field
            for field, value in schedule_data.items():
                key = f'{hostname}::schedule::{field}'
                conn.execute("""
                    INSERT INTO config (key, value) VALUES (?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = CURRENT_TIMESTAMP
                """, (key, value, value))

        return {'success': True, 'message': f'Schedule saved for {hostname}'}
    except Exception as e:
        logger.error(f"Error saving schedule for {hostname}: {e}")
        return {'success': False, 'error': str(e)}


def delete_worker_schedule(workdir, hostname):
    """Delete per-host schedule configuration (but keep enabled state)"""
    init_workers_db(workdir)
    workdir_path = Path(workdir)
    db_path = workdir_path / 'workers.db'

    try:
        with sqlite3.connect(db_path.as_posix()) as conn:
            # Delete weekday_start, weekday_end, weekend_allowed (but keep enabled)
            conn.execute("""
                DELETE FROM config
                WHERE key LIKE ? || '::schedule::%'
                AND key NOT LIKE ? || '::schedule::enabled'
            """, (hostname, hostname))

        return {'success': True, 'message': f'Schedule reset to global defaults for {hostname}'}
    except Exception as e:
        logger.error(f"Error deleting schedule for {hostname}: {e}")
        return {'success': False, 'error': str(e)}


def get_default_schedule(workdir):
    """Get default schedule configuration"""
    init_workers_db(workdir)
    workdir_path = Path(workdir)
    db_path = workdir_path / 'workers.db'

    try:
        with sqlite3.connect(db_path.as_posix()) as conn:
            cursor = conn.execute("""
                SELECT key, value FROM config
                WHERE key LIKE '::schedule::default::%'
            """)

            schedule = {}
            for row in cursor:
                key = row[0]
                # key format: ::schedule::default::field
                field = key.split('::')[-1]
                schedule[field] = row[1]

            return schedule if schedule else None
    except Exception as e:
        logger.error(f"Error getting default schedule: {e}")
        return None


def save_default_schedule(workdir, schedule_data):
    """Save default schedule configuration"""
    init_workers_db(workdir)
    workdir_path = Path(workdir)
    db_path = workdir_path / 'workers.db'

    try:
        with sqlite3.connect(db_path.as_posix()) as conn:
            # Save each schedule field
            for field, value in schedule_data.items():
                key = f'::schedule::default::{field}'
                conn.execute("""
                    INSERT INTO config (key, value) VALUES (?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = CURRENT_TIMESTAMP
                """, (key, value, value))

        return {'success': True, 'message': 'Default schedule saved'}
    except Exception as e:
        logger.error(f"Error saving default schedule: {e}")
        return {'success': False, 'error': str(e)}


def get_scheduled_workers(workdir):
    """Get all workers with enabled schedules"""
    init_workers_db(workdir)
    workdir_path = Path(workdir)
    db_path = workdir_path / 'workers.db'

    try:
        with sqlite3.connect(db_path.as_posix()) as conn:
            # Get all workers with schedule::enabled=true
            cursor = conn.execute("""
                SELECT key FROM config
                WHERE key LIKE '%::schedule::enabled' AND value = 'true'
            """)

            scheduled_workers = {}
            for row in cursor:
                key = row[0]
                # key format: hostname::schedule::enabled
                hostname = key.split('::')[0]

                # Get full schedule for this worker
                schedule = get_worker_schedule(workdir, hostname)
                if schedule:
                    scheduled_workers[hostname] = schedule

            return scheduled_workers
    except Exception as e:
        logger.error(f"Error getting scheduled workers: {e}")
        return {}


# ---------------------------------------------------------------------------
# Performance visualization
# ---------------------------------------------------------------------------

PLOTLY_CDN_URL = (
    'https://cdn.jsdelivr.net/npm/plotly.js-basic-dist@2.35.2/plotly.basic.min.js'
)


def _ensure_plotly_cache(workdir: str) -> None:
    """Download Plotly.js to scratch/webcache/ if not already present."""
    import urllib.request
    cache_path = Path(workdir) / 'scratch' / 'webcache' / 'plotly.basic.min.js'
    if cache_path.exists():
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info('Downloading Plotly.js to %s', cache_path)
    urllib.request.urlretrieve(PLOTLY_CDN_URL, cache_path)
    logger.info('Plotly.js cached (%d bytes)', cache_path.stat().st_size)


def get_perf_archs(workdir: str) -> list[str]:
    """Return sorted list of arches with best_tuning_results data."""
    try:
        conn_params = get_db_connection_params(Path(workdir))
        with psycopg.connect(**conn_params, autocommit=True) as conn:
            return get_available_archs(conn)
    except Exception as e:
        logger.error('get_perf_archs failed: %s', e)
        return []


def query_perf_data(workdir: str, arch: str, kernel: str, mode: str = 'kernel',
                    seqlen_min: int = 0, seqlen_max: int = 65536) -> dict:
    """Query best results for the Perf tab API endpoint."""
    try:
        conn_params = get_db_connection_params(Path(workdir))
        with psycopg.connect(**conn_params, autocommit=True) as conn:
            return query_best_results(conn, arch, kernel, mode, seqlen_min, seqlen_max)
    except Exception as e:
        logger.error('query_perf_data failed: %s', e)
        return {'error': str(e), 'rows': [], 'axes': {}}


def export_visperf(workdir: str, dry_run: bool = False) -> dict:
    """Export self-contained perf.html to <workdir>/perf.html."""
    if dry_run:
        return {'status': 'dry_run', 'message': '[DRY RUN] Would export perf.html'}
    output = Path(workdir) / 'perf.html'
    try:
        conn_params = get_db_connection_params(Path(workdir))
        with psycopg.connect(**conn_params, autocommit=True) as conn:
            _do_export_visperf(conn, output)
        return {'status': 'ok', 'message': f'Exported to {output}'}
    except Exception as e:
        logger.error('export_visperf failed: %s', e)
        return {'status': 'error', 'message': str(e)}
