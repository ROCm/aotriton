# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Read-only guest Flask app — no auth, no TLS, no write endpoints.
"""

import time
import threading
import datetime
from dataclasses import dataclass
from flask import Flask, render_template
from pathlib import Path


@dataclass
class _ProgressCache:
    progress: list | None = None
    last_refresh: str | None = None
    timestamp: float = 0.0


def create_guest_app(workdir, refresh_interval: int = 120):
    workdir_path = Path(workdir).resolve()
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.config['WORKDIR'] = workdir_path.as_posix()

    from . import tasks

    # Server-side cache: only query DB when refresh_interval has elapsed.
    _cache_lock = threading.Lock()
    _cache = _ProgressCache()

    def get_cached_progress():
        now = time.monotonic()
        with _cache_lock:
            if _cache.progress is None or (now - _cache.timestamp) >= refresh_interval:
                _cache.progress = tasks.get_tuning_progress(workdir_path.as_posix())
                _cache.last_refresh = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                _cache.timestamp = now
            return _cache.progress, _cache.last_refresh

    # Client polls 2 seconds after the server cache expires so data is always fresh.
    client_interval = refresh_interval + 2

    @app.route('/')
    def dashboard():
        status = tasks.get_status_summary(workdir_path.as_posix())
        tuning_progress, last_refresh = get_cached_progress()
        return render_template(
            'guest_dashboard.html',
            status=status,
            tuning_progress=tuning_progress,
            last_refresh=last_refresh,
            client_interval=client_interval,
            config=app.config,
        )

    @app.route('/api/tuning-progress')
    def api_tuning_progress():
        tuning_progress, last_refresh = get_cached_progress()
        return render_template(
            '_tuning_progress_table.html',
            tuning_progress=tuning_progress,
            last_refresh=last_refresh,
        )

    return app
