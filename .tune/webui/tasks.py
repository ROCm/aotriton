# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Task execution module for web dashboard.
Provides functions to execute CLI commands and query the database.
"""

import subprocess
import sqlite3
import os


def run_command(cmd, workdir=None):
    """
    Execute shell command, return (stdout, stderr, returncode)
    For MVP: just return debug message instead of executing
    """
    # DEBUG: Not executing command yet
    return {
        'stdout': f'[DEBUG] Would execute: {cmd}',
        'stderr': '',
        'returncode': 0
    }


def get_workers(workdir):
    """Query workers from database"""
    db_path = os.path.join(workdir, 'workers.db')
    if not os.path.exists(db_path):
        return []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT hostname, arch, COALESCE(workdir_override, '') FROM workers ORDER BY hostname")
    workers = cursor.fetchall()
    conn.close()
    return workers


def get_worker_by_hostname(workdir, hostname):
    """Get single worker info by hostname"""
    db_path = os.path.join(workdir, 'workers.db')
    if not os.path.exists(db_path):
        return None

    conn = sqlite3.connect(db_path)
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
    db_path = os.path.join(workdir, 'workers.db')
    if not os.path.exists(db_path):
        return []

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT arch FROM workers ORDER BY arch")
    archs = [row[0] for row in cursor.fetchall()]
    conn.close()
    return archs


def get_hostnames(workdir):
    """Get list of distinct hostnames"""
    db_path = os.path.join(workdir, 'workers.db')
    if not os.path.exists(db_path):
        return []

    conn = sqlite3.connect(db_path)
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
    tune_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(tune_root, 'single', 'start_worker.sh')
    cmd = f"{script} {workdir} {hostname}"
    return run_command(cmd)


def stop_worker_single(workdir, hostname):
    """Stop single worker via .tune/single/stop_worker.sh"""
    tune_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(tune_root, 'single', 'stop_worker.sh')
    cmd = f"{script} {workdir} {hostname}"
    return run_command(cmd)


def restart_worker_single(workdir, hostname):
    """Restart single worker via .tune/single/restart_worker.sh"""
    tune_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(tune_root, 'single', 'restart_worker.sh')
    cmd = f"{script} {workdir} {hostname}"
    return run_command(cmd)


def start_all_workers(workdir):
    """Start all workers via .tune/bin/wkctl"""
    tune_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(tune_root, 'bin', 'wkctl')
    cmd = f"{script} {workdir} start"
    return run_command(cmd)


def stop_all_workers(workdir):
    """Stop all workers via .tune/bin/wkctl"""
    tune_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(tune_root, 'bin', 'wkctl')
    cmd = f"{script} {workdir} stop"
    return run_command(cmd)


def restart_all_workers(workdir):
    """Restart all workers via .tune/bin/wkctl"""
    tune_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(tune_root, 'bin', 'wkctl')
    cmd = f"{script} {workdir} restart"
    return run_command(cmd)


# Server control functions

def start_servers(workdir):
    """Start RabbitMQ and PostgreSQL via .tune/bin/srvctl"""
    tune_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(tune_root, 'bin', 'srvctl')
    cmd = f"{script} {workdir} start"
    return run_command(cmd)


def stop_servers(workdir):
    """Stop RabbitMQ and PostgreSQL via .tune/bin/srvctl"""
    tune_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(tune_root, 'bin', 'srvctl')
    cmd = f"{script} {workdir} stop"
    return run_command(cmd)


def restart_servers(workdir):
    """Restart RabbitMQ and PostgreSQL via .tune/bin/srvctl"""
    tune_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(tune_root, 'bin', 'srvctl')
    cmd = f"{script} {workdir} restart"
    return run_command(cmd)


# Build functions

def build_libraries(workdir):
    """Build AOTriton libraries for all architectures via .tune/bin/libbld"""
    tune_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(tune_root, 'bin', 'libbld')
    cmd = f"{script} {workdir}"
    return run_command(cmd)


def build_images(workdir):
    """Build Docker images on all workers via .tune/bin/imgbld"""
    tune_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(tune_root, 'bin', 'imgbld')
    cmd = f"{script} {workdir}"
    return run_command(cmd)


# Deploy functions

def deploy_workdir(workdir):
    """Deploy workdir to all workers via .tune/bin/deploy"""
    tune_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(tune_root, 'bin', 'deploy')
    cmd = f"{script} {workdir}"
    return run_command(cmd)


def prepare_workdir(workdir):
    """Prepare workdir via .tune/bin/prepwkdir"""
    tune_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(tune_root, 'bin', 'prepwkdir')
    cmd = f"{script} {workdir}"
    return run_command(cmd)
