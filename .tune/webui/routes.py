# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
URL routes and handlers for the web dashboard
"""

from flask import Blueprint, render_template, request, jsonify, current_app
from . import tasks

bp = Blueprint('main', __name__)


@bp.route('/')
def dashboard():
    """Dashboard overview page"""
    workdir = current_app.config['WORKDIR']
    status = tasks.get_status_summary(workdir)
    return render_template('dashboard.html', status=status)


@bp.route('/workers')
def workers():
    """Worker management page"""
    workdir = current_app.config['WORKDIR']
    workers = tasks.get_workers(workdir)
    return render_template('workers.html', workers=workers)


@bp.route('/servers')
def servers():
    """Server control page"""
    workdir = current_app.config['WORKDIR']
    return render_template('servers.html')


@bp.route('/builds')
def builds():
    """Build management page"""
    workdir = current_app.config['WORKDIR']
    archs = tasks.get_architectures(workdir)
    hostnames = tasks.get_hostnames(workdir)
    return render_template('builds.html', archs=archs, hostnames=hostnames)


@bp.route('/deploy')
def deploy():
    """Deployment page"""
    return render_template('deploy.html')


@bp.route('/slurm')
def slurm():
    """SLURM management page"""
    return render_template('slurm.html')


# API endpoints for worker actions

@bp.route('/api/workers/all/start', methods=['POST'])
def api_start_all_workers():
    """Start all workers"""
    workdir = current_app.config['WORKDIR']
    result = tasks.start_all_workers(workdir)
    return jsonify(result)


@bp.route('/api/workers/all/stop', methods=['POST'])
def api_stop_all_workers():
    """Stop all workers"""
    workdir = current_app.config['WORKDIR']
    result = tasks.stop_all_workers(workdir)
    return jsonify(result)


@bp.route('/api/workers/all/restart', methods=['POST'])
def api_restart_all_workers():
    """Restart all workers"""
    workdir = current_app.config['WORKDIR']
    result = tasks.restart_all_workers(workdir)
    return jsonify(result)


@bp.route('/api/workers/<hostname>/start', methods=['POST'])
def api_start_worker(hostname):
    """Start single worker"""
    workdir = current_app.config['WORKDIR']
    result = tasks.start_worker_single(workdir, hostname)
    return jsonify(result)


@bp.route('/api/workers/<hostname>/stop', methods=['POST'])
def api_stop_worker(hostname):
    """Stop single worker"""
    workdir = current_app.config['WORKDIR']
    result = tasks.stop_worker_single(workdir, hostname)
    return jsonify(result)


@bp.route('/api/workers/<hostname>/restart', methods=['POST'])
def api_restart_worker(hostname):
    """Restart single worker"""
    workdir = current_app.config['WORKDIR']
    result = tasks.restart_worker_single(workdir, hostname)
    return jsonify(result)


@bp.route('/api/workers/<hostname>/status', methods=['GET'])
def api_worker_status(hostname):
    """Get worker status"""
    workdir = current_app.config['WORKDIR']
    worker = tasks.get_worker_by_hostname(workdir, hostname)
    if worker:
        return '<span style="color: gray">Unknown</span>'
    return '<span style="color: red">Not Found</span>'


# API endpoints for server actions

@bp.route('/api/servers/start', methods=['POST'])
def api_start_servers():
    """Start RabbitMQ and PostgreSQL"""
    workdir = current_app.config['WORKDIR']
    result = tasks.start_servers(workdir)
    return jsonify(result)


@bp.route('/api/servers/stop', methods=['POST'])
def api_stop_servers():
    """Stop RabbitMQ and PostgreSQL"""
    workdir = current_app.config['WORKDIR']
    result = tasks.stop_servers(workdir)
    return jsonify(result)


@bp.route('/api/servers/restart', methods=['POST'])
def api_restart_servers():
    """Restart RabbitMQ and PostgreSQL"""
    workdir = current_app.config['WORKDIR']
    result = tasks.restart_servers(workdir)
    return jsonify(result)


# API endpoints for build actions

@bp.route('/api/builds/libraries', methods=['POST'])
def api_build_libraries():
    """Build AOTriton libraries"""
    workdir = current_app.config['WORKDIR']
    result = tasks.build_libraries(workdir)
    return jsonify(result)


@bp.route('/api/builds/images', methods=['POST'])
def api_build_images():
    """Build Docker images"""
    workdir = current_app.config['WORKDIR']
    result = tasks.build_images(workdir)
    return jsonify(result)


# API endpoints for deployment actions

@bp.route('/api/deploy/workdir', methods=['POST'])
def api_deploy_workdir():
    """Deploy workdir to all workers"""
    workdir = current_app.config['WORKDIR']
    result = tasks.deploy_workdir(workdir)
    return jsonify(result)


@bp.route('/api/deploy/prepare', methods=['POST'])
def api_prepare_workdir():
    """Prepare workdir"""
    workdir = current_app.config['WORKDIR']
    result = tasks.prepare_workdir(workdir)
    return jsonify(result)
