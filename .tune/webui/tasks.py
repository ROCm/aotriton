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

def run_command(cmd, cwd, workdir, description=None):
    """
    Execute shell command with per-action tracker

    Args:
        cmd: Command as list of Path/str objects (e.g., [script_path, arg1, arg2])
        cwd: Current working directory for command execution (Path object)
        workdir: Workdir path where logs should be stored (str)
        description: Human-readable description

    Returns:
        dict with action_id, status, message
    """
    # Convert cmd list to strings
    cmd_parts = [str(p) for p in cmd]
    cmd_str = ' '.join(cmd_parts)

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


def get_tuning_progress(workdir):
    """Get tuning progress from queue_progress view with speed calculation"""
    try:
        conn_params = get_db_connection_params(Path(workdir))
        with psycopg.connect(**conn_params, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                # Get progress from view
                cur.execute("SELECT * FROM queue_progress ORDER BY arch")
                progress_rows = cur.fetchall()

                # Calculate speed: tasks completed in last 5 minutes
                cur.execute("""
                    SELECT
                        arch,
                        COUNT(*) as recent_completions
                    FROM task_queue
                    WHERE status = 'completed'
                      AND completed_at > NOW() - INTERVAL '5 minutes'
                    GROUP BY arch
                """)
                speed_rows = cur.fetchall()

                # Count stale tasks (running > 2 hours)
                cur.execute("""
                    SELECT
                        arch,
                        COUNT(*) as stale_count
                    FROM task_queue
                    WHERE status = 'running'
                      AND EXTRACT(EPOCH FROM (NOW() - started_at)) > 7200
                    GROUP BY arch
                """)
                stale_rows = cur.fetchall()

                # Build speed and stale maps
                speed_map = {row['arch']: row['recent_completions'] / 5.0 for row in speed_rows}
                stale_map = {row['arch']: row['stale_count'] for row in stale_rows}

                # Merge data
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
    except Exception as e:
        logging.error(f"Failed to get tuning progress: {e}")
        return []


_TUNE_V3BIS_MARKER = 'TUNE_V3BIS testrun Item: '


def resolve_tune_entry(workdir, line: str) -> dict:
    """
    Parse a TUNE_V3BIS testrun line and return the matching task_queue id.

    Accepts the full line or just the payload after the marker.
    Returns {'task_id': <int>} or {'error': <str>}.
    """
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

    clauses = ["task_config->>'arch' = %s"]
    params: list = [arch]
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
                    " error, gpu_id, created_at FROM tuning_results"
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

        return {
            'task': task,
            'tuning_results': tuning_results,
            'best_results': best_results,
            'accurate_results': accurate_results,
        }
    except Exception as e:
        logging.error('Failed to get debug data for task %s: %s', task_id, e)
        return {'error': str(e)}


# Command execution helpers

class CommandBuilder:
    """Base class for building commands"""

    def _run(self, script_relative_path, args, workdir, description):
        """Execute command with proper paths"""
        cmd = [script_relative_path] + list(args)
        return run_command(cmd, cwd=AOTRITON_ROOT, workdir=workdir, description=description)


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

    def exec(self, workdir, hostname, options=None):
        """Execute command with script at RELATIVE path"""
        args = _build_worker_args(workdir, hostname, options)
        return self._run(self.RELATIVE, args, workdir, f'{self.ACTION_NAME} worker {hostname}')


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

    def exec(self, workdir):
        """Execute wkctl with action"""
        return self._run(self.RELATIVE, [workdir, self.ACTION], workdir, f'{self.ACTION.capitalize()} all workers')


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

    def exec(self, workdir):
        """Execute srvctl with action"""
        return self._run(self.RELATIVE, [workdir, self.ACTION], workdir, f'{self.ACTION.capitalize()} servers')


class StartServersCommand(ServerCommand):
    ACTION = 'start'


class StopServersCommand(ServerCommand):
    ACTION = 'stop'


class RestartServersCommand(ServerCommand):
    ACTION = 'restart'


class InitDatabaseCommand(CommandBuilder):
    """Initialize database schema"""
    RELATIVE = '.tune/bin/initdb'

    def exec(self, workdir):
        """Execute initdb script"""
        return self._run(self.RELATIVE, [workdir], workdir, 'Initialize database schema')


class RecreateSchemaCommand(CommandBuilder):
    """Recreate database schema (drop all tables first)"""
    RELATIVE = '.tune/bin/initdb'

    def exec(self, workdir):
        """Execute initdb script with --recreate flag"""
        return self._run(self.RELATIVE, [workdir, '--recreate'], workdir, 'Recreate database schema')


class ComputeBestResultsCommand(CommandBuilder):
    """Compute best_tuning_results table from raw tuning results"""
    RELATIVE = '.tune/bin/compute_best_results'

    def exec(self, workdir):
        return self._run(self.RELATIVE, [workdir], workdir, 'Compute best tuning results')


class ExportBestResultsCommand(CommandBuilder):
    """Export best results to centralized SQLite database"""
    RELATIVE = '.tune/bin/export_best_results'

    def exec(self, workdir):
        return self._run(self.RELATIVE, [workdir], workdir, 'Export best results to centraldb')


class BuildCommand(CommandBuilder):
    """Base class for build operations"""
    RELATIVE = None  # Subclass must define
    DESCRIPTION = None  # Subclass must define

    def exec(self, workdir):
        """Execute build script"""
        return self._run(self.RELATIVE, [workdir], workdir, self.DESCRIPTION)


class BuildLibrariesCommand(BuildCommand):
    RELATIVE = '.tune/bin/libbld'
    DESCRIPTION = 'Build libraries'


class BuildImagesCommand(BuildCommand):
    RELATIVE = '.tune/bin/imgbld'
    DESCRIPTION = 'Build Docker images'


class BuildImageOnWorkerCommand(CommandBuilder):
    """Build Docker image on a single worker"""
    RELATIVE = '.tune/single/build_image.sh'

    def exec(self, workdir, hostname):
        """Execute build_image.sh for a specific worker with --follow for web UI"""
        return self._run(self.RELATIVE, [workdir, hostname, '--follow'], workdir, f'Build image on {hostname}')


class DeployCommand(CommandBuilder):
    """Base class for deployment operations"""
    RELATIVE = None  # Subclass must define
    DESCRIPTION = None  # Subclass must define

    def exec(self, workdir):
        """Execute deployment script"""
        return self._run(self.RELATIVE, [workdir], workdir, self.DESCRIPTION)


class DeployAllCommand(DeployCommand):
    RELATIVE = '.tune/bin/deploy'
    DESCRIPTION = 'Deploy to all workers'


class PrepareWorkdirCommand(DeployCommand):
    RELATIVE = '.tune/bin/prepwkdir'
    DESCRIPTION = 'Prepare workdir'

    def exec(self, workdir):
        # Ensure log directory exists (use /scratch which is excluded from sync)
        log_dir = Path(workdir) / 'scratch' / 'webui-commands'
        log_dir.mkdir(parents=True, exist_ok=True)

        return super().exec(workdir)


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

_build_libraries = BuildLibrariesCommand()
_build_images = BuildImagesCommand()
_build_image_on_worker = BuildImageOnWorkerCommand()

_deploy_all = DeployAllCommand()
_prepare_workdir = PrepareWorkdirCommand()


# Worker control functions

def start_worker_single(workdir, hostname, options=None):
    """Start single worker"""
    return _start_worker.exec(workdir, hostname, options)


def stop_worker_single(workdir, hostname):
    """Stop single worker"""
    return _stop_worker.exec(workdir, hostname)


def restart_worker_single(workdir, hostname, options=None):
    """Restart single worker"""
    return _restart_worker.exec(workdir, hostname, options)


def stop_start_worker_single(workdir, hostname, options=None):
    """Stop then start single worker"""
    return _stopstart_worker.exec(workdir, hostname, options)


def start_all_workers(workdir):
    """Start all workers"""
    return _start_all_workers.exec(workdir)


def stop_all_workers(workdir):
    """Stop all workers"""
    return _stop_all_workers.exec(workdir)


def restart_all_workers(workdir):
    """Restart all workers"""
    return _restart_all_workers.exec(workdir)


def stop_start_all_workers(workdir):
    """Stop then start all workers (using shell ; operator)"""
    # Special case: need sequential execution with ; operator
    wkctl = '.tune/bin/wkctl'
    cmd = ['/bin/bash', '-c', f'{wkctl} {workdir} stop ; {wkctl} {workdir} start']
    return run_command(cmd, cwd=AOTRITON_ROOT, workdir=workdir, description='Stop & start all workers')


# Server control functions

def start_servers(workdir):
    """Start servers"""
    return _start_servers.exec(workdir)


def stop_servers(workdir):
    """Stop servers"""
    return _stop_servers.exec(workdir)


def restart_servers(workdir):
    """Restart servers"""
    return _restart_servers.exec(workdir)


def init_database(workdir):
    """Initialize database schema"""
    return _init_database.exec(workdir)


def recreate_schema(workdir):
    """Recreate database schema (drop all tables first)"""
    return _recreate_schema.exec(workdir)


def compute_best_results(workdir):
    """Compute best_tuning_results table from raw tuning results"""
    return _compute_best_results.exec(workdir)


def export_best_results(workdir):
    """Export best results to centralized SQLite database"""
    return _export_best_results.exec(workdir)


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

def build_libraries(workdir):
    """Build libraries"""
    return _build_libraries.exec(workdir)


def build_images(workdir):
    """Build Docker images"""
    return _build_images.exec(workdir)


def build_image_on_worker(workdir, hostname):
    """Build Docker image on specific worker"""
    return _build_image_on_worker.exec(workdir, hostname)


# Deploy functions

def deploy_workdir(workdir):
    """Deploy to all workers"""
    return _deploy_all.exec(workdir)


def deploy_workdir_single(workdir, hostname):
    """Deploy to single worker"""
    return _deploy_worker.exec(workdir, hostname)


def prepare_workdir(workdir):
    """Prepare workdir"""
    return _prepare_workdir.exec(workdir)


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
                'status_display': status.get('display', 'Unknown')
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


def detect_gpu_for_worker(workdir, hostname):
    """Detect GPU metadata for a specific worker"""
    cmd = ['.tune/single/detect_gpu.sh', workdir, hostname]
    return run_command(cmd, cwd=AOTRITON_ROOT, workdir=workdir, description=f'Detect GPU info for {hostname}')


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
