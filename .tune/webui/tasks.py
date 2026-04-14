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

# Import command tracker
from .command_tracker import get_tracker


def run_command(cmd, workdir=None, description=None):
    """
    Execute shell command with per-action tracker
    Returns action_id for tracking
    """
    from flask import current_app

    # Get log directory from workdir
    workdir_path = Path(workdir) if workdir else Path.cwd()
    log_dir = workdir_path / 'logs' / 'commands'

    # Create tracker
    tracker = current_app.tracker_registry.create(
        command=cmd,
        description=description or cmd,
        workdir=workdir,
        log_dir=log_dir.as_posix()
    )

    # Start execution in background
    tracker.start()

    # Also log to global tracker (backup)
    global_tracker = get_tracker()
    global_tracker.record_action(tracker)

    return {
        'action_id': tracker.action_id,
        'status': 'running',
        'message': f'Command started: {description or cmd}'
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


# Worker control functions (DEBUG mode - just return messages)

def start_worker_single(workdir, hostname):
    """Start single worker via .tune/single/start_worker.sh"""
    tune_root = Path(__file__).parent.parent.resolve()
    script = tune_root / 'single' / 'start_worker.sh'
    cmd = f"{script.as_posix()} {workdir} {hostname}"
    return run_command(cmd)


def stop_worker_single(workdir, hostname):
    """Stop single worker via .tune/single/stop_worker.sh"""
    tune_root = Path(__file__).parent.parent.resolve()
    script = tune_root / 'single' / 'stop_worker.sh'
    cmd = f"{script.as_posix()} {workdir} {hostname}"
    return run_command(cmd)


def restart_worker_single(workdir, hostname):
    """Restart single worker via .tune/single/restart_worker.sh"""
    tune_root = Path(__file__).parent.parent.resolve()
    script = tune_root / 'single' / 'restart_worker.sh'
    cmd = f"{script.as_posix()} {workdir} {hostname}"
    return run_command(cmd)


def stop_start_worker_single(workdir, hostname):
    """Stop then start single worker"""
    tune_root = Path(__file__).parent.parent.resolve()
    start_script = tune_root / 'single' / 'start_worker.sh'
    stop_script = tune_root / 'single' / 'stop_worker.sh'
    cmd = f"{stop_script.as_posix()} {workdir} {hostname} ; {start_script.as_posix()} {workdir} {hostname}"
    return run_command(cmd)


def start_all_workers(workdir):
    """Start all workers via .tune/bin/wkctl"""
    tune_root = Path(__file__).parent.parent.resolve()
    script = tune_root / 'bin' / 'wkctl'
    cmd = f"{script.as_posix()} {workdir} start"
    return run_command(cmd)


def stop_all_workers(workdir):
    """Stop all workers via .tune/bin/wkctl"""
    tune_root = Path(__file__).parent.parent.resolve()
    script = tune_root / 'bin' / 'wkctl'
    cmd = f"{script.as_posix()} {workdir} stop"
    return run_command(cmd)


def restart_all_workers(workdir):
    """Restart all workers via .tune/bin/wkctl"""
    tune_root = Path(__file__).parent.parent.resolve()
    script = tune_root / 'bin' / 'wkctl'
    cmd = f"{script.as_posix()} {workdir} restart"
    return run_command(cmd)


def stop_start_all_workers(workdir):
    """Stop then start all workers via .tune/bin/wkctl"""
    tune_root = Path(__file__).parent.parent.resolve()
    script = tune_root / 'bin' / 'wkctl'
    cmd = f"{script.as_posix()} {workdir} stop ; {script.as_posix()} {workdir} start"
    return run_command(cmd)


# Server control functions

def start_servers(workdir):
    """Start RabbitMQ and PostgreSQL via .tune/bin/srvctl"""
    tune_root = Path(__file__).parent.parent.resolve()
    script = tune_root / 'bin' / 'srvctl'
    cmd = f"{script.as_posix()} {workdir} start"
    return run_command(cmd)


def stop_servers(workdir):
    """Stop RabbitMQ and PostgreSQL via .tune/bin/srvctl"""
    tune_root = Path(__file__).parent.parent.resolve()
    script = tune_root / 'bin' / 'srvctl'
    cmd = f"{script.as_posix()} {workdir} stop"
    return run_command(cmd)


def restart_servers(workdir):
    """Restart RabbitMQ and PostgreSQL via .tune/bin/srvctl"""
    tune_root = Path(__file__).parent.parent.resolve()
    script = tune_root / 'bin' / 'srvctl'
    cmd = f"{script.as_posix()} {workdir} restart"
    return run_command(cmd)


# Build functions

def build_libraries(workdir):
    """Build AOTriton libraries for all architectures via .tune/bin/libbld"""
    tune_root = Path(__file__).parent.parent.resolve()
    script = tune_root / 'bin' / 'libbld'
    cmd = f"{script.as_posix()} {workdir}"
    return run_command(cmd)


def build_images(workdir):
    """Build Docker images on all workers via .tune/bin/imgbld"""
    tune_root = Path(__file__).parent.parent.resolve()
    script = tune_root / 'bin' / 'imgbld'
    cmd = f"{script.as_posix()} {workdir}"
    return run_command(cmd)


# Deploy functions

def deploy_workdir(workdir):
    """Deploy workdir to all workers via .tune/bin/deploy"""
    tune_root = Path(__file__).parent.parent.resolve()
    script = tune_root / 'bin' / 'deploy'
    cmd = f"{script.as_posix()} {workdir}"
    return run_command(cmd)


def deploy_workdir_single(workdir, hostname):
    """Deploy workdir to a single worker via .tune/single/sync_workdir.sh"""
    tune_root = Path(__file__).parent.parent.resolve()
    script = tune_root / 'single' / 'sync_workdir.sh'
    cmd = f"{script.as_posix()} {workdir} {hostname}"
    return run_command(cmd)


def prepare_workdir(workdir):
    """Prepare workdir via .tune/bin/prepwkdir"""
    tune_root = Path(__file__).parent.parent.resolve()
    script = tune_root / 'bin' / 'prepwkdir'
    cmd = f"{script.as_posix()} {workdir}"

    # Ensure logs directory exists
    log_dir = Path(workdir) / 'logs' / 'commands'
    log_dir.mkdir(parents=True, exist_ok=True)

    return run_command(cmd)


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
