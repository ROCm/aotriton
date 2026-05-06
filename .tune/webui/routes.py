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
    workers_by_arch = tasks.get_workers_by_architecture(workdir)
    # Flatten workers from all architectures for status display
    workers = []
    for arch_workers in workers_by_arch.values():
        workers.extend(arch_workers)
    tuning_progress = tasks.get_tuning_progress(workdir)
    refresh_interval = current_app.config['REFRESH_INTERVAL']
    return render_template('dashboard.html', status=status, workers=workers,
                           tuning_progress=tuning_progress, refresh_interval=refresh_interval)


@bp.route('/workers')
def workers():
    """Worker management page"""
    workdir = current_app.config['WORKDIR']
    workers_by_arch = tasks.get_workers_by_architecture(workdir)
    supported_archs = tasks.get_supported_architectures()
    default_workdir = tasks.get_default_workdir(workdir) or '<not set>'
    git_status = tasks.get_git_status(workdir)
    return render_template('workers.html',
                          workers_by_arch=workers_by_arch,
                          supported_archs=supported_archs,
                          default_workdir=default_workdir,
                          git_status=git_status)


@bp.route('/servers')
def servers():
    """Server control page"""
    workdir = current_app.config['WORKDIR']
    config_vars = tasks.get_config_vars(workdir)
    hostnames = tasks.get_hostnames(workdir)
    return render_template('servers.html', config_vars=config_vars, hostnames=hostnames)


@bp.route('/builds')
def builds():
    """Build management page"""
    workdir = current_app.config['WORKDIR']
    archs = tasks.get_architectures(workdir)
    hostnames = tasks.get_hostnames(workdir)
    build_node_config = tasks.get_build_node_config(workdir)
    default_workdir = tasks.get_default_workdir(workdir) or '(not set)'
    git_status = tasks.get_git_status(workdir)
    return render_template('builds.html', archs=archs, hostnames=hostnames,
                           build_node_config=build_node_config,
                           default_workdir=default_workdir,
                           git_status=git_status)


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
    options = request.get_json() or {}
    result = tasks.start_worker_single(workdir, hostname, options=options)
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
    options = request.get_json() or {}
    result = tasks.restart_worker_single(workdir, hostname, options=options)
    return jsonify(result)


@bp.route('/api/workers/<hostname>/stop-start', methods=['POST'])
def api_stop_start_worker(hostname):
    """Stop then start single worker"""
    workdir = current_app.config['WORKDIR']
    options = request.get_json() or {}
    result = tasks.stop_start_worker_single(workdir, hostname, options=options)
    return jsonify(result)


@bp.route('/api/workers/<hostname>/status', methods=['GET'])
def api_worker_status(hostname):
    """Get cached worker status"""
    workdir = current_app.config['WORKDIR']
    status = tasks.get_cached_worker_status(workdir, hostname)
    if status:
        return status.get('display', 'Unknown')
    return 'Unknown'


@bp.route('/api/workers/<hostname>/probe-status', methods=['POST'])
def api_probe_worker_status(hostname):
    """Probe worker status and cache result"""
    workdir = current_app.config['WORKDIR']
    status = tasks.probe_worker_status(workdir, hostname)
    return jsonify(status)


@bp.route('/api/workers/<hostname>/build-image', methods=['POST'])
def api_build_image_on_worker(hostname):
    """Build Docker image on specific worker"""
    workdir = current_app.config['WORKDIR']
    result = tasks.build_image_on_worker(workdir, hostname)
    return jsonify(result)


@bp.route('/api/workers/<hostname>/detect-gpu', methods=['POST'])
def api_detect_gpu(hostname):
    """Detect GPU metadata for a specific worker"""
    workdir = current_app.config['WORKDIR']
    result = tasks.detect_gpu_for_worker(workdir, hostname)
    return jsonify(result)


@bp.route('/api/workers/<hostname>/gpu-selection', methods=['POST'])
def api_save_gpu_selection(hostname):
    """Save GPU selection for a worker"""
    workdir = current_app.config['WORKDIR']
    data = request.get_json() or {}
    gpu_ids = data.get('gpu_ids', [-1])
    result = tasks.save_worker_gpu_selection(workdir, hostname, gpu_ids)
    return jsonify(result)


@bp.route('/api/workers/probe-all-status', methods=['POST'])
def api_probe_all_workers_status():
    """Probe status for all workers"""
    workdir = current_app.config['WORKDIR']
    result = tasks.probe_all_workers_status(workdir)
    return jsonify(result)


# API endpoints for server actions

@bp.route('/api/servers/start', methods=['POST'])
def api_start_servers():
    """Start PostgreSQL"""
    workdir = current_app.config['WORKDIR']
    result = tasks.start_servers(workdir)
    return jsonify(result)


@bp.route('/api/servers/stop', methods=['POST'])
def api_stop_servers():
    """Stop PostgreSQL"""
    workdir = current_app.config['WORKDIR']
    result = tasks.stop_servers(workdir)
    return jsonify(result)


@bp.route('/api/servers/restart', methods=['POST'])
def api_restart_servers():
    """Restart PostgreSQL"""
    workdir = current_app.config['WORKDIR']
    result = tasks.restart_servers(workdir)
    return jsonify(result)


@bp.route('/api/servers/initdb', methods=['POST'])
def api_init_database():
    """Initialize database schema"""
    workdir = current_app.config['WORKDIR']
    result = tasks.init_database(workdir)
    return jsonify(result)


@bp.route('/api/servers/recreate-schema', methods=['POST'])
def api_recreate_schema():
    """Recreate database schema (drops all tables first)"""
    workdir = current_app.config['WORKDIR']
    result = tasks.recreate_schema(workdir)
    return jsonify(result)


@bp.route('/api/servers/compute-best-results', methods=['POST'])
def api_compute_best_results():
    """Compute best_tuning_results table from raw tuning results"""
    workdir = current_app.config['WORKDIR']
    result = tasks.compute_best_results(workdir)
    return jsonify(result)


@bp.route('/api/servers/export-best-results', methods=['POST'])
def api_export_best_results():
    """Export best results to centralized SQLite database"""
    workdir = current_app.config['WORKDIR']
    result = tasks.export_best_results(workdir)
    return jsonify(result)


@bp.route('/api/servers/recreate-materialized-view', methods=['POST'])
def api_recreate_materialized_view():
    """Recreate most_accurate_tuning_results via DROP + CREATE"""
    workdir = current_app.config['WORKDIR']
    result = tasks.recreate_materialized_view(workdir)
    return jsonify(result)


@bp.route('/api/servers/sancheck', methods=['POST'])
def api_sancheck():
    """Run LUT sanity check against the exported centralized database"""
    workdir = current_app.config['WORKDIR']
    result = tasks.sancheck(workdir)
    return jsonify(result)


@bp.route('/api/servers/bake-lut', methods=['POST'])
def api_bake_lut():
    """Bake LUT — full, incremental, or fix-broken depending on form fields.

    Form fields (all optional):
      incremental=1          → passes --incremental
      fix=1 + pass_num=N     → passes --fix [hostname:]N  (hostname optional)
      hostname=<h>           → used only when fix=1; sentinel '__all__' means all hosts
    """
    workdir = current_app.config['WORKDIR']
    extra_args: list = []

    if request.form.get('fix') == '1':
        pass_num = request.form.get('pass_num', '').strip()
        if not pass_num:
            return jsonify({'status': 'error', 'message': 'pass_num is required for fix mode'}), 400
        hostname = request.form.get('hostname', '').strip()
        fix_spec = f'{hostname}:{pass_num}' if hostname and hostname != '__all__' else pass_num
        extra_args = ['--fix', fix_spec]
    elif request.form.get('incremental') == '1':
        extra_args = ['--incremental']

    result = tasks.bake_lut(workdir, extra_args or None)
    return jsonify(result)


@bp.route('/api/servers/update-materialized-view', methods=['POST'])
def api_update_materialized_view():
    workdir = current_app.config['WORKDIR']
    result = tasks.update_materialized_view(workdir)
    return jsonify(result)


@bp.route('/api/servers/decomposedb', methods=['POST'])
def api_decomposedb():
    """Decompose centraldb.sqlite3 into per-arch/kernel shards"""
    workdir = current_app.config['WORKDIR']
    result = tasks.decomposedb(workdir)
    return jsonify(result)


@bp.route('/api/servers/status', methods=['GET'])
def api_server_status():
    """Get server status (returns HTML for HTMX)"""
    workdir = current_app.config['WORKDIR']
    result = tasks.get_server_status(workdir)
    return render_template('partials/server_status.html', **result)


@bp.route('/api/workers/git-status', methods=['GET'])
def api_git_status():
    """Get git status (returns HTML for HTMX)"""
    workdir = current_app.config['WORKDIR']
    git_status = tasks.get_git_status(workdir)
    return render_template('partials/git_status.html', git_status=git_status)


# API endpoints for build actions

@bp.route('/api/builds/libraries', methods=['POST'])
def api_build_libraries():
    """Build tuning version of AOTriton libraries (all arches or single arch)."""
    workdir = current_app.config['WORKDIR']
    single_arch = request.form.get('single_arch', '').strip() or None
    result = tasks.build_libraries(workdir, single_arch)
    return jsonify(result)


@bp.route('/api/builds/test-libraries', methods=['POST'])
def api_build_test_libraries():
    """Build testing version of AOTriton libraries inside container (all or single arch)."""
    workdir = current_app.config['WORKDIR']
    single_arch = request.form.get('single_arch', '').strip() or None
    result = tasks.build_test_libraries(workdir, single_arch)
    return jsonify(result)


@bp.route('/api/builds/build-node', methods=['GET'])
def api_get_build_node_config():
    """Get build node configuration"""
    workdir = current_app.config['WORKDIR']
    config = tasks.get_build_node_config(workdir)
    return jsonify(config)


@bp.route('/api/builds/build-node', methods=['POST'])
def api_set_build_node_config():
    """Save build node configuration"""
    workdir = current_app.config['WORKDIR']
    hostname = request.form.get('hostname', '').strip()
    workdir_override = request.form.get('workdir_override', '').strip()
    enabled = request.form.get('enabled') == 'on'
    result = tasks.set_build_node_config(workdir, hostname, workdir_override, enabled)
    return jsonify(result)


@bp.route('/api/builds/deploy-build-node', methods=['POST'])
def api_sync_build_node():
    """Sync local workdir to the remote build node"""
    workdir = current_app.config['WORKDIR']
    result = tasks.sync_build_node(workdir)
    return jsonify(result)


@bp.route('/api/builds/fetch-tuning', methods=['POST'])
def api_fetch_tuning_build():
    """Fetch tuning build artifacts from remote build node"""
    workdir = current_app.config['WORKDIR']
    result = tasks.fetch_tuning_build(workdir)
    return jsonify(result)


@bp.route('/api/builds/fetch-test', methods=['POST'])
def api_fetch_test_build():
    """Fetch test build artifacts from remote build node"""
    workdir = current_app.config['WORKDIR']
    result = tasks.fetch_test_build(workdir)
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
    """Deploy workdir to a single worker; pass testnode=1 to add --testnode flag"""
    workdir = current_app.config['WORKDIR']
    extra_args = ['--testnode'] if request.form.get('testnode') == '1' else None
    result = tasks.deploy_workdir_single(workdir, hostname, extra_args=extra_args)
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


@bp.route('/api/workers/<hostname>/role/<role>', methods=['POST'])
def api_set_worker_role(hostname, role):
    """Set or clear a role for a worker (body: enabled=true|false)"""
    workdir = current_app.config['WORKDIR']
    enabled_raw = request.form.get('enabled', 'false').strip().lower()
    enabled = enabled_raw in ('1', 'true', 'yes', 'on')
    result = tasks.set_worker_role(workdir, hostname, role, enabled)
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
    # Get tracker registry before entering generator (to avoid app context issues)
    tracker_registry = current_app.tracker_registry

    def generate():
        try:
            tracker = tracker_registry.get(action_id)
            if not tracker:
                yield 'event: error\ndata: {"error": "Tracker not found"}\n\n'
                return

            sent_stdout = 0
            sent_stderr = 0

            # Send initial connection message
            yield ':ping\n\n'

            while True:
                try:
                    output = tracker.get_output(from_line=0)  # Get all lines

                    # Send new stdout lines (only what we haven't sent yet)
                    stdout_list = output['stdout']
                    for i in range(sent_stdout, len(stdout_list)):
                        line = stdout_list[i]
                        data = json.dumps({"line": line})
                        yield f'event: stdout\ndata: {data}\n\n'
                    sent_stdout = len(stdout_list)

                    # Send new stderr lines (only what we haven't sent yet)
                    stderr_list = output['stderr']
                    for i in range(sent_stderr, len(stderr_list)):
                        line = stderr_list[i]
                        data = json.dumps({"line": line})
                        yield f'event: stderr\ndata: {data}\n\n'
                    sent_stderr = len(stderr_list)

                    # Send status update
                    status_data = json.dumps({"status": output["status"], "returncode": output["returncode"]})
                    yield f'event: status\ndata: {status_data}\n\n'

                    # If completed, send final event and close
                    if output['status'] in ['completed', 'failed']:
                        complete_data = json.dumps({"status": output["status"], "returncode": output["returncode"]})
                        yield f'event: complete\ndata: {complete_data}\n\n'
                        break

                    time.sleep(0.1)  # Poll every 100ms
                except Exception as e:
                    import traceback
                    error_msg = f'{str(e)}\n{traceback.format_exc()}'
                    yield f'event: error\ndata: {json.dumps({"error": error_msg})}\n\n'
                    break
        except Exception as e:
            import traceback
            error_msg = f'{str(e)}\n{traceback.format_exc()}'
            yield f'event: error\ndata: {json.dumps({"error": error_msg})}\n\n'

    response = Response(generate(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response


@bp.route('/api/actions/<action_id>/output', methods=['GET'])
def get_action_output(action_id):
    """Get action output as HTML for polling (incremental)"""
    try:
        tracker = current_app.tracker_registry.get(action_id)
        if not tracker:
            return '(tracker not found)', 404

        # Get offset from query parameter
        offset = int(request.args.get('offset', 0))

        output = tracker.get_output()

        # Combine stdout and stderr with line numbers to maintain order
        all_lines = []
        for i, line in enumerate(output['stdout']):
            all_lines.append((i, 'stdout', line))
        for i, line in enumerate(output['stderr']):
            all_lines.append((i, 'stderr', line))

        # Get only new lines starting from offset
        new_lines = all_lines[offset:]

        # If no new lines
        if not new_lines:
            if offset == 0 and output['status'] in ['completed', 'failed']:
                return '(no output)\n'
            return ''  # No new content

        # Build plain text for new lines only
        lines_text = []
        for idx, line_type, line in new_lines:
            lines_text.append(line)

        # Return new lines with updated offset in response header
        response = '\n'.join(lines_text) + '\n'
        return response, 200, {'X-Output-Offset': str(len(all_lines))}

    except Exception as e:
        import traceback
        return f'<div style="color: red;">Error: {str(e)}<br><pre>{traceback.format_exc()}</pre></div>', 500


@bp.route('/api/actions/<action_id>/status', methods=['GET'])
def get_action_status(action_id):
    """Get current status and output of an action"""
    try:
        tracker = current_app.tracker_registry.get(action_id)
        if not tracker:
            return jsonify({'error': 'Action not found'}), 404

        output = tracker.get_output()
        info = tracker.to_dict()

        return jsonify({
            **info,
            'stdout_lines': output['stdout'],
            'stderr_lines': output['stderr'],
            'total_stdout': len(output['stdout']),
            'total_stderr': len(output['stderr'])
        })
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


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
    return '', 200


@bp.route('/api/actions/<action_id>/kill', methods=['POST'])
def kill_action(action_id):
    """Kill a running action"""
    tracker = current_app.tracker_registry.get(action_id)
    if not tracker:
        return jsonify({'success': False, 'error': 'Action not found'}), 404

    success = tracker.kill()
    if success:
        return jsonify({'success': True, 'message': 'Process killed'})
    else:
        return jsonify({'success': False, 'error': 'Process not running or already terminated'}), 400


@bp.route('/api/actions/clear', methods=['POST'])
def clear_all_actions():
    """Clear all action trackers"""
    current_app.tracker_registry.clear_all()
    return jsonify({'success': True})


# Schedule management routes

@bp.route('/api/workers/<hostname>/schedule', methods=['GET'])
def api_get_worker_schedule(hostname):
    """Get schedule configuration for a worker"""
    workdir = current_app.config['WORKDIR']
    schedule = tasks.get_worker_schedule(workdir, hostname)
    return jsonify(schedule or {})


@bp.route('/api/workers/<hostname>/schedule', methods=['POST'])
def api_save_worker_schedule(hostname):
    """Save schedule configuration for a worker"""
    workdir = current_app.config['WORKDIR']
    data = request.get_json() or {}
    result = tasks.save_worker_schedule(workdir, hostname, data)
    return jsonify(result)


@bp.route('/api/workers/<hostname>/schedule', methods=['DELETE'])
def api_delete_worker_schedule(hostname):
    """Delete schedule configuration for a worker"""
    workdir = current_app.config['WORKDIR']
    result = tasks.delete_worker_schedule(workdir, hostname)
    return jsonify(result)


@bp.route('/api/workers/schedule/default', methods=['GET'])
def api_get_default_schedule():
    """Get default schedule configuration"""
    workdir = current_app.config['WORKDIR']
    schedule = tasks.get_default_schedule(workdir)
    return jsonify(schedule or {})


@bp.route('/api/workers/schedule/default', methods=['POST'])
def api_save_default_schedule():
    """Save default schedule configuration"""
    workdir = current_app.config['WORKDIR']
    data = request.get_json() or {}
    result = tasks.save_default_schedule(workdir, data)
    return jsonify(result)


@bp.route('/api/workers/schedule/arm', methods=['POST'])
def api_arm_scheduled_workers():
    """Arm timers for all scheduled workers"""
    count = current_app.scheduler.arm_all_scheduled_workers()
    return jsonify({'armed_count': count, 'message': f'Armed timers for {count} workers'})


@bp.route('/api/workers/schedule/status', methods=['GET'])
def api_get_scheduler_status():
    """Get status of all armed timers"""
    status = current_app.scheduler.get_scheduler_status()
    return jsonify(status)


@bp.route('/api/tuning-progress', methods=['GET'])
def api_get_tuning_progress():
    """Get tuning progress data (for HTMX polling)"""
    workdir = current_app.config['WORKDIR']
    tuning_progress = tasks.get_tuning_progress(workdir)
    return render_template('_tuning_progress_table.html', tuning_progress=tuning_progress, last_refresh=None)


@bp.route('/api/debug/resolve_entry', methods=['POST'])
def api_debug_resolve_entry():
    """Resolve a TUNE_V3BIS testrun line to a task_queue id."""
    workdir = current_app.config['WORKDIR']
    line = (request.get_json() or {}).get('line', '')
    if not line:
        return jsonify({'error': 'No line provided'}), 400
    result = tasks.resolve_tune_entry(workdir, line)
    return jsonify(result)


@bp.route('/testing')
def testing():
    """Testing tab"""
    workdir = current_app.config['WORKDIR']
    workers = tasks.get_workers_for_testing(workdir)
    git_status = tasks.get_git_status(workdir)
    return render_template('testing.html', workers=workers, git_status=git_status)


@bp.route('/api/testing/<hostname>/role', methods=['POST'])
def api_set_tester_role(hostname):
    """Set or clear Tester role for a host"""
    workdir = current_app.config['WORKDIR']
    enabled_raw = request.form.get('enabled', 'false').strip().lower()
    enabled = enabled_raw in ('1', 'true', 'yes', 'on')
    result = tasks.set_worker_role(workdir, hostname, 'Tester', enabled)
    return jsonify(result)


@bp.route('/api/testing/<hostname>/workdir', methods=['POST'])
def api_update_tester_workdir(hostname):
    """Update custom workdir for a tester host"""
    workdir = current_app.config['WORKDIR']
    workdir_override = request.form.get('workdir_override', '').strip() or None
    result = tasks.update_worker_workdir(workdir, hostname, workdir_override)
    return jsonify(result)


@bp.route('/api/testing/<hostname>/signature', methods=['GET'])
def api_tester_signature(hostname):
    """Get __signature__ file content from a tester host"""
    workdir = current_app.config['WORKDIR']
    result = tasks.get_tester_signature(workdir, hostname)
    return jsonify(result)



@bp.route('/api/testing/<hostname>/run-test', methods=['POST'])
def api_run_test(hostname):
    """Run .ci/run-test.sh on a tester host"""
    workdir = current_app.config['WORKDIR']
    pass_num = str(request.form.get('pass_num', '0'))
    test_level = str(request.form.get('test_level', '0'))
    backend = request.form.get('backend', 'split')
    variant = request.form.get('variant') or request.args.get('variant') or None
    if backend not in ('split', 'fused', 'aiter', 'v3'):
        return jsonify({'status': 'error', 'message': f'Invalid backend: {backend}'}), 400
    if variant and variant not in ('partial', 'partial_adiffs'):
        return jsonify({'status': 'error', 'message': f'Invalid variant: {variant}'}), 400
    result = tasks.run_test_on_host(workdir, hostname, pass_num, test_level, backend, variant=variant)
    return jsonify(result)


@bp.route('/api/testing/<hostname>/tail-output', methods=['GET'])
def api_tail_output(hostname):
    """Fetch tail -n 5 of test output files for a tester host"""
    workdir = current_app.config['WORKDIR']
    pass_num = request.args.get('pass', '0')
    backend = request.args.get('backend', 'split')
    variant = request.args.get('variant') or None
    result = tasks.get_tail_output(workdir, hostname, pass_num, backend, variant=variant)
    return jsonify(result)


@bp.route('/api/testing/<hostname>/failures', methods=['GET'])
def api_testing_failures(hostname):
    """Return a full HTML page listing FAILED test lines from the output file."""
    workdir = current_app.config['WORKDIR']
    pass_num = request.args.get('pass', '0')
    backend = request.args.get('backend', 'split')
    variant = request.args.get('variant') or None
    result = tasks.get_failed_tests(workdir, hostname, pass_num, backend, variant=variant)

    variant_label = f' ({variant})' if variant else ''
    title = f'Failures — {hostname} pass {pass_num}{variant_label}'

    if result['status'] == 'not_found':
        body = f'<p class="err">File not found: <code>{result["filename"]}</code></p>'
    elif result['status'] == 'error':
        msg = result.get('message', 'Unknown error')
        body = f'<p class="err">Error: {msg}</p>'
    elif not result['lines']:
        body = '<p class="ok">No failures found.</p>'
    else:
        import html as _html
        escaped = _html.escape('\n'.join(result['lines']))
        body = (
            f'<p class="meta">{len(result["lines"])} failure(s) in'
            f' <code>{result["filename"]}</code></p>'
            f'<pre class="failures">{escaped}</pre>'
        )

    page = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{
    margin: 0;
    padding: 1rem 1.4rem;
    background: #1a1a1a;
    color: #d4d4d4;
    font-family: system-ui, sans-serif;
    font-size: 0.95rem;
  }}
  h1 {{
    font-size: 1.1rem;
    margin: 0 0 0.8rem 0;
    color: #e0e0e0;
    border-bottom: 1px solid #333;
    padding-bottom: 0.4rem;
  }}
  code {{
    font-family: monospace;
    background: #2a2a2a;
    padding: 0.1em 0.3em;
    border-radius: 3px;
  }}
  .meta {{ color: #999; font-size: 0.85em; margin: 0 0 0.5rem 0; }}
  .err {{ color: #f88; }}
  .ok  {{ color: #8f8; }}
  pre.failures {{
    margin: 0;
    padding: 0.7rem;
    background: #111;
    color: #f88;
    border: 1px solid #333;
    border-radius: 4px;
    white-space: pre-wrap;
    overflow-wrap: break-word;
    line-height: 1.45;
    font-size: 0.88rem;
  }}
</style>
</head>
<body>
<h1>{title}</h1>
{body}
</body>
</html>'''
    return page, 200, {'Content-Type': 'text/html; charset=utf-8'}


@bp.route('/api/testing/download_adiffs', methods=['GET'])
def api_download_adiffs():
    """SSH-read adiffs.txt from a tester host, save it locally, and serve as a download."""
    workdir = current_app.config['WORKDIR']
    hostname = request.args.get('hostname', '').strip()
    arch = request.args.get('arch', '').strip()
    if not hostname or not arch:
        return jsonify({'status': 'error', 'message': 'hostname and arch are required'}), 400
    result = tasks.get_adiffs_file(workdir, hostname, arch)
    if result['status'] == 'not_found':
        return result['message'], 404, {'Content-Type': 'text/plain; charset=utf-8'}
    if result['status'] == 'error':
        return result['message'], 500, {'Content-Type': 'text/plain; charset=utf-8'}
    content = result['content']
    filename = f'{arch}.txt'
    return content, 200, {
        'Content-Type': 'text/plain; charset=utf-8',
        'Content-Disposition': f'attachment; filename="{filename}"',
    }


@bp.route('/debug')
def debug():
    """Debug page - inspect task_queue entries and all related rows"""
    task_id_str = request.args.get('task_id', '').strip()
    if task_id_str:
        try:
            task_id = int(task_id_str)
        except ValueError:
            task_id = None
    else:
        task_id = None

    debug_data = None
    if task_id is not None:
        workdir = current_app.config['WORKDIR']
        debug_data = tasks.get_debug_task_data(workdir, task_id)

    return render_template('debug.html', task_id=task_id, debug_data=debug_data)
