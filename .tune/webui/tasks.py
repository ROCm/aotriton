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

# Import supported GPU architectures
_tune_root = Path(__file__).parent.parent
sys.path.insert(0, (_tune_root.parent / 'v3python').as_posix())
from gpu_targets import AOTRITON_ARCH_TO_PACK

# Import tuning architectures configuration
from .config import TUNING_ARCHITECTURES


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
    from flask import current_app

    # Convert cmd list to strings
    cmd_parts = [str(p) for p in cmd]
    cmd_str = ' '.join(cmd_parts)

    # Get log directory from workdir
    log_dir = Path(workdir) / 'logs' / 'commands'

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

    return {
        'worker_count': len(workers),
        'architecture_count': len(archs),
        'architectures': archs,
        'rabbitmq_status': 'unknown',
        'postgres_status': 'unknown',
    }


# Command execution helpers

class CommandBuilder:
    """Base class for building commands"""

    def __init__(self):
        self.aotriton_root = Path(__file__).parent.parent.parent.resolve()

    def _run(self, script_path, args, workdir, description):
        """Execute command with proper paths"""
        cmd = [script_path] + list(args)
        return run_command(cmd, cwd=self.aotriton_root, workdir=workdir, description=description)


class SingleWorkerCommand(CommandBuilder):
    """Base class for single worker operations"""
    RELATIVE = None  # Subclass must define
    ACTION_NAME = None  # Subclass must define

    def exec(self, workdir, hostname):
        """Execute command with script at RELATIVE path"""
        script = self.aotriton_root / self.RELATIVE
        return self._run(script, [workdir, hostname], workdir, f'{self.ACTION_NAME} worker {hostname}')


class StartWorkerCommand(SingleWorkerCommand):
    RELATIVE = '.tune/single/start_worker.sh'
    ACTION_NAME = 'Start'


class StopWorkerCommand(SingleWorkerCommand):
    RELATIVE = '.tune/single/stop_worker.sh'
    ACTION_NAME = 'Stop'


class RestartWorkerCommand(SingleWorkerCommand):
    RELATIVE = '.tune/single/restart_worker.sh'
    ACTION_NAME = 'Restart'


class DeployWorkerCommand(SingleWorkerCommand):
    RELATIVE = '.tune/single/sync_workdir.sh'
    ACTION_NAME = 'Deploy to'


class BulkWorkerCommand(CommandBuilder):
    """Base class for bulk worker operations"""
    RELATIVE = '.tune/bin/wkctl'
    ACTION = None  # Subclass must define

    def exec(self, workdir):
        """Execute wkctl with action"""
        script = self.aotriton_root / self.RELATIVE
        return self._run(script, [workdir, self.ACTION], workdir, f'{self.ACTION.capitalize()} all workers')


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
        script = self.aotriton_root / self.RELATIVE
        return self._run(script, [workdir, self.ACTION], workdir, f'{self.ACTION.capitalize()} servers')


class StartServersCommand(ServerCommand):
    ACTION = 'start'


class StopServersCommand(ServerCommand):
    ACTION = 'stop'


class RestartServersCommand(ServerCommand):
    ACTION = 'restart'


class BuildCommand(CommandBuilder):
    """Base class for build operations"""
    RELATIVE = None  # Subclass must define
    DESCRIPTION = None  # Subclass must define

    def exec(self, workdir):
        """Execute build script"""
        script = self.aotriton_root / self.RELATIVE
        return self._run(script, [workdir], workdir, self.DESCRIPTION)


class BuildLibrariesCommand(BuildCommand):
    RELATIVE = '.tune/bin/libbld'
    DESCRIPTION = 'Build libraries'


class BuildImagesCommand(BuildCommand):
    RELATIVE = '.tune/bin/imgbld'
    DESCRIPTION = 'Build Docker images'


class DeployCommand(CommandBuilder):
    """Base class for deployment operations"""
    RELATIVE = None  # Subclass must define
    DESCRIPTION = None  # Subclass must define

    def exec(self, workdir):
        """Execute deployment script"""
        script = self.aotriton_root / self.RELATIVE
        return self._run(script, [workdir], workdir, self.DESCRIPTION)


class DeployAllCommand(DeployCommand):
    RELATIVE = '.tune/bin/deploy'
    DESCRIPTION = 'Deploy to all workers'


class PrepareWorkdirCommand(DeployCommand):
    RELATIVE = '.tune/bin/prepwkdir'
    DESCRIPTION = 'Prepare workdir'

    def exec(self, workdir):
        # Ensure logs directory exists
        log_dir = Path(workdir) / 'logs' / 'commands'
        log_dir.mkdir(parents=True, exist_ok=True)

        return super().exec(workdir)


# Global command instances
_start_worker = StartWorkerCommand()
_stop_worker = StopWorkerCommand()
_restart_worker = RestartWorkerCommand()
_deploy_worker = DeployWorkerCommand()

_start_all_workers = StartAllWorkersCommand()
_stop_all_workers = StopAllWorkersCommand()
_restart_all_workers = RestartAllWorkersCommand()

_start_servers = StartServersCommand()
_stop_servers = StopServersCommand()
_restart_servers = RestartServersCommand()

_build_libraries = BuildLibrariesCommand()
_build_images = BuildImagesCommand()

_deploy_all = DeployAllCommand()
_prepare_workdir = PrepareWorkdirCommand()


# Worker control functions

def start_worker_single(workdir, hostname):
    """Start single worker"""
    return _start_worker.exec(workdir, hostname)


def stop_worker_single(workdir, hostname):
    """Stop single worker"""
    return _stop_worker.exec(workdir, hostname)


def restart_worker_single(workdir, hostname):
    """Restart single worker"""
    return _restart_worker.exec(workdir, hostname)


def stop_start_worker_single(workdir, hostname):
    """Stop then start single worker (using shell ; operator)"""
    # Special case: need sequential execution with ; operator
    aotriton_root = Path(__file__).parent.parent.parent.resolve()
    stop_script = aotriton_root / '.tune' / 'single' / 'stop_worker.sh'
    start_script = aotriton_root / '.tune' / 'single' / 'start_worker.sh'
    cmd = ['/bin/bash', '-c', f'{stop_script} {workdir} {hostname} ; {start_script} {workdir} {hostname}']
    return run_command(cmd, cwd=aotriton_root, workdir=workdir, description=f'Stop & start worker {hostname}')


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
    aotriton_root = Path(__file__).parent.parent.parent.resolve()
    wkctl = aotriton_root / '.tune' / 'bin' / 'wkctl'
    cmd = ['/bin/bash', '-c', f'{wkctl} {workdir} stop ; {wkctl} {workdir} start']
    return run_command(cmd, cwd=aotriton_root, workdir=workdir, description='Stop & start all workers')


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


# Build functions

def build_libraries(workdir):
    """Build libraries"""
    return _build_libraries.exec(workdir)


def build_images(workdir):
    """Build Docker images"""
    return _build_images.exec(workdir)


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

    # Group workers by architecture
    workers_by_arch = {arch: [] for arch in TUNING_ARCHITECTURES}

    for hostname, arch, workdir_override in workers:
        if arch in workers_by_arch:
            workers_by_arch[arch].append({
                'hostname': hostname,
                'arch': arch,
                'workdir': workdir_override or ''
            })

    return workers_by_arch


def add_worker(workdir, hostname, arch, workdir_override=None):
    """Add a worker to the database"""
    # Validate architecture
    if arch not in AOTRITON_ARCH_TO_PACK:
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
