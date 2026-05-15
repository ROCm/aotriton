# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from flask import Flask
from pathlib import Path
import ssl


def create_app(workdir, refresh_interval: int = 10):
    """Create and configure the Flask application"""
    workdir_path = Path(workdir).resolve()
    app = Flask(__name__)
    app.config['WORKDIR'] = workdir_path.as_posix()
    app.config['TUNE_ROOT'] = Path(__file__).parent.parent.resolve().as_posix()
    app.config['REFRESH_INTERVAL'] = refresh_interval

    # Configure mTLS
    secrets_dir = workdir_path / 'secrets'
    app.config['SSL_CERT'] = (secrets_dir / 'server.crt').as_posix()
    app.config['SSL_KEY'] = (secrets_dir / 'server.key').as_posix()
    app.config['SSL_CA'] = (secrets_dir / 'ca.crt').as_posix()

    # Initialize tracker registry
    from .action_tracker import TrackerRegistry
    app.tracker_registry = TrackerRegistry()

    # Initialize worker scheduler
    from .scheduler import WorkerScheduler
    app.scheduler = WorkerScheduler(workdir_path.as_posix())

    from . import routes
    app.register_blueprint(routes.bp)

    return app


def create_ssl_context(app):
    """Create SSL context for mTLS"""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.verify_mode = ssl.CERT_REQUIRED  # Require client certificate
    ctx.load_cert_chain(app.config['SSL_CERT'], app.config['SSL_KEY'])
    ctx.load_verify_locations(app.config['SSL_CA'])
    return ctx
