#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
AOTriton Tuning Dashboard
Web interface for managing distributed tuning infrastructure
"""

import argparse
import sys
import os

# Add .tune to path so we can import webui
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from webui import create_app, create_ssl_context

try:
    from cheroot.wsgi import Server as WSGIServer
    from cheroot.ssl.builtin import BuiltinSSLAdapter
except ImportError:
    print("Error: cheroot not installed. Install with: pip install cheroot", file=sys.stderr)
    sys.exit(1)
import ssl


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AOTriton Tuning Dashboard')
    parser.add_argument('workdir', help='Working directory path')
    parser.add_argument('--port', type=int, default=8888, help='Port to listen on (default: 8888)')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    args = parser.parse_args()

    # Validate workdir
    if not os.path.isdir(args.workdir):
        print(f"Error: Working directory not found: {args.workdir}", file=sys.stderr)
        sys.exit(1)

    # Check certificates exist
    secrets_dir = os.path.join(args.workdir, 'secrets')
    required_certs = ['ca.crt', 'server.crt', 'server.key']
    missing = [cert for cert in required_certs if not os.path.exists(os.path.join(secrets_dir, cert))]

    if missing:
        print(f"Error: Missing certificates in {secrets_dir}: {', '.join(missing)}", file=sys.stderr)
        print(f"\nGenerate certificates with:", file=sys.stderr)
        tune_root = os.path.dirname(os.path.abspath(__file__))
        print(f"  {tune_root}/webui/gencerts/generate_all_certs.sh {args.workdir}", file=sys.stderr)
        sys.exit(1)

    app = create_app(args.workdir)

    # Get SSL certificate paths
    server_cert = os.path.join(secrets_dir, 'server.crt')
    server_key = os.path.join(secrets_dir, 'server.key')
    ca_cert = os.path.join(secrets_dir, 'ca.crt')

    print(f"=" * 70)
    print(f"AOTriton Tuning Dashboard")
    print(f"=" * 70)
    print(f"URL:     https://{args.host}:{args.port}")
    print(f"Workdir: {args.workdir}")
    print(f"Certificates: {secrets_dir}")
    print(f"")
    print(f"Client certificate required: {secrets_dir}/client.p12")
    print(f"Import client.p12 into your browser to access the dashboard.")
    print(f"=" * 70)

    # Configure Cheroot WSGI server with mTLS
    server = WSGIServer(
        (args.host, args.port),
        app,
        numthreads=10
    )

    # Setup SSL with client certificate verification
    ssl_adapter = BuiltinSSLAdapter(server_cert, server_key, ca_cert)
    ssl_adapter.context.verify_mode = ssl.CERT_REQUIRED
    server.ssl_adapter = ssl_adapter

    try:
        server.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        # Cancel all scheduled timers before stopping server
        app.scheduler.cancel_all_timers()
        server.stop()
