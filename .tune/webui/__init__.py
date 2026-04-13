# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from flask import Flask
import os
import ssl


def create_app(workdir):
    """Create and configure the Flask application"""
    app = Flask(__name__)
    app.config['WORKDIR'] = os.path.abspath(workdir)
    app.config['TUNE_ROOT'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Configure mTLS
    secrets_dir = os.path.join(workdir, 'secrets')
    app.config['SSL_CERT'] = os.path.join(secrets_dir, 'server.crt')
    app.config['SSL_KEY'] = os.path.join(secrets_dir, 'server.key')
    app.config['SSL_CA'] = os.path.join(secrets_dir, 'ca.crt')

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
