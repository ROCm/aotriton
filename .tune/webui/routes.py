# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
URL routes and handlers for the web dashboard
"""

from flask import Blueprint, render_template, request, jsonify, current_app, Response
import time
import json

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
    workers_by_arch = tasks.get_workers_by_architecture(workdir)
    supported_archs = tasks.get_supported_architectures()
    default_workdir = tasks.get_default_workdir(workdir) or '<not set>'
    return render_template('workers.html',
                          workers_by_arch=workers_by_arch,
                          supported_archs=supported_archs,
                          default_workdir=default_workdir)


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


@bp.route('/commands')
def commands():
    """Commands page (mobile)"""
    return render_template('commands.html')


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


@bp.route('/api/workers/all/stop-start', methods=['POST'])
def api_stop_start_all_workers():
    """Stop then start all workers"""
    workdir = current_app.config['WORKDIR']
    result = tasks.stop_start_all_workers(workdir)
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


@bp.route('/api/workers/<hostname>/stop-start', methods=['POST'])
def api_stop_start_worker(hostname):
    """Stop then start single worker"""
    workdir = current_app.config['WORKDIR']
    result = tasks.stop_start_worker_single(workdir, hostname)
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


@bp.route('/api/deploy/<hostname>', methods=['POST'])
def api_deploy_single(hostname):
    """Deploy workdir to a single worker"""
    workdir = current_app.config['WORKDIR']
    result = tasks.deploy_workdir_single(workdir, hostname)
    return jsonify(result)


@bp.route('/api/deploy/prepare', methods=['POST'])
def api_prepare_workdir():
    """Prepare workdir"""
    workdir = current_app.config['WORKDIR']
    result = tasks.prepare_workdir(workdir)
    return jsonify(result)


# API endpoints for worker management

@bp.route('/api/workers/add', methods=['POST'])
def api_add_worker():
    """Add a new worker"""
    workdir = current_app.config['WORKDIR']
    hostname = request.form.get('hostname', '').strip()
    arch = request.form.get('arch', '').strip()
    workdir_override = request.form.get('workdir_override', '').strip() or None

    if not hostname or not arch:
        return jsonify({'success': False, 'error': 'Hostname and architecture are required'}), 400

    result = tasks.add_worker(workdir, hostname, arch, workdir_override)
    return jsonify(result)


@bp.route('/api/workers/<hostname>/remove', methods=['POST'])
def api_remove_worker(hostname):
    """Remove a worker"""
    workdir = current_app.config['WORKDIR']
    result = tasks.remove_worker(workdir, hostname)
    return jsonify(result)


@bp.route('/api/workers/<hostname>/workdir', methods=['POST'])
def api_update_worker_workdir(hostname):
    """Update worker's custom workdir"""
    workdir = current_app.config['WORKDIR']
    workdir_override = request.form.get('workdir_override', '').strip() or None
    result = tasks.update_worker_workdir(workdir, hostname, workdir_override)
    return jsonify(result)


@bp.route('/api/config/default-workdir', methods=['GET', 'POST'])
def api_default_workdir():
    """Get or set default working directory"""
    workdir = current_app.config['WORKDIR']

    if request.method == 'GET':
        default_wd = tasks.get_default_workdir(workdir)
        return jsonify({'default_workdir': default_wd})

    # POST - update default workdir
    path = request.form.get('path', '').strip()
    if not path:
        return jsonify({'success': False, 'error': 'Path is required'}), 400

    result = tasks.set_default_workdir(workdir, path)
    return jsonify(result)


@bp.route('/api/architectures', methods=['GET'])
def api_get_architectures():
    """Get list of supported GPU architectures"""
    archs = tasks.get_supported_architectures()
    return jsonify(archs)


# API endpoints for command output tracking

@bp.route('/api/stream/<action_id>')
def stream_output(action_id):
    """Server-Sent Events endpoint for real-time output streaming"""
    def generate():
        tracker = current_app.tracker_registry.get(action_id)
        if not tracker:
            yield 'event: error\ndata: {"error": "Tracker not found"}\n\n'
            return

        sent = 0
        while True:
            output = tracker.get_output(from_line=sent)

            # Send new stdout lines
            for line in output['stdout']:
                yield f'event: stdout\ndata: {json.dumps({"line": line})}\n\n'

            # Send new stderr lines
            for line in output['stderr']:
                yield f'event: stderr\ndata: {json.dumps({"line": line})}\n\n'

            sent = output['total_stdout']

            # Send status update
            yield f'event: status\ndata: {json.dumps({"status": output["status"], "returncode": output["returncode"]})}\n\n'

            # If completed, send final event and close
            if output['status'] in ['completed', 'failed']:
                yield f'event: complete\ndata: {json.dumps({"status": output["status"], "returncode": output["returncode"]})}\n\n'
                break

            time.sleep(0.1)  # Poll every 100ms

    return Response(generate(), mimetype='text/event-stream')


@bp.route('/api/actions/<action_id>/status', methods=['GET'])
def get_action_status(action_id):
    """Get current status and output of an action"""
    tracker = current_app.tracker_registry.get(action_id)
    if not tracker:
        return jsonify({'error': 'Action not found'}), 404

    output = tracker.get_output()
    info = tracker.to_dict()

    return jsonify({
        **info,
        'stdout_lines': output['stdout'],
        'stderr_lines': output['stderr']
    })


@bp.route('/api/actions', methods=['GET'])
def list_actions():
    """List all active action trackers"""
    trackers = current_app.tracker_registry.get_all()
    return jsonify({
        'actions': [t.to_dict() for t in trackers]
    })


@bp.route('/api/actions/<action_id>', methods=['DELETE'])
def remove_action(action_id):
    """Remove a specific action tracker"""
    current_app.tracker_registry.remove(action_id)
    return jsonify({'success': True})


@bp.route('/api/actions/clear', methods=['POST'])
def clear_all_actions():
    """Clear all action trackers"""
    current_app.tracker_registry.clear_all()
    return jsonify({'success': True})
