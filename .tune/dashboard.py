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
    parser.add_argument('--port', type=int, default=None, help='Port to listen on (default: 8888, or 9999 in guest mode)')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--guest', action='store_true', help='Read-only guest mode: no auth, plain HTTP, port 9999')
    parser.add_argument('--demo', action='store_true', help='Demo mode: mTLS on port 8899, all mutating actions blocked')
    parser.add_argument('--refresh_interval', type=int, default=None,
                        help='Tuning Progress refresh interval in seconds (default: 120 for guest, 10 for admin)')
    args = parser.parse_args()

    # Validate workdir
    if not os.path.isdir(args.workdir):
        print(f"Error: Working directory not found: {args.workdir}", file=sys.stderr)
        sys.exit(1)

    if args.guest:
        from webui.guest import create_guest_app
        port = args.port if args.port is not None else 9999
        refresh_interval = args.refresh_interval if args.refresh_interval is not None else 120
        app = create_guest_app(args.workdir, refresh_interval=refresh_interval)

        print(f"=" * 70)
        print(f"AOTriton Tuning Dashboard (Guest / Read-Only)")
        print(f"=" * 70)
        print(f"URL:     http://{args.host}:{port}")
        print(f"Workdir: {args.workdir}")
        print(f"Refresh: every {refresh_interval}s")
        print(f"=" * 70)

        server = WSGIServer((args.host, port), app, numthreads=4)
        try:
            server.start()
        except KeyboardInterrupt:
            print("\nShutting down...")
            server.stop()

    else:
        from webui import create_app

        is_demo = args.demo
        if is_demo:
            port = args.port if args.port is not None else 8899
        else:
            port = args.port if args.port is not None else 8888

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

        refresh_interval = args.refresh_interval if args.refresh_interval is not None else 10
        app = create_app(args.workdir, refresh_interval=refresh_interval, demo_mode=is_demo)

        server_cert = os.path.join(secrets_dir, 'server.crt')
        server_key = os.path.join(secrets_dir, 'server.key')
        ca_cert = os.path.join(secrets_dir, 'ca.crt')

        print(f"=" * 70)
        if is_demo:
            print(f"AOTriton Tuning Dashboard (DEMO MODE)")
        else:
            print(f"AOTriton Tuning Dashboard")
        print(f"=" * 70)
        print(f"URL:     https://{args.host}:{port}")
        print(f"Workdir: {args.workdir}")
        print(f"Certificates: {secrets_dir}")
        print(f"")
        if is_demo:
            print(f"Demo certificate required: {secrets_dir}/demo.p12")
            print(f"Import demo.p12 into your browser to access the dashboard.")
            print(f"All mutating operations are BLOCKED in demo mode.")
        else:
            print(f"Client certificate required: {secrets_dir}/client.p12")
            print(f"Import client.p12 into your browser to access the dashboard.")
        print(f"=" * 70)

        server = WSGIServer((args.host, port), app, numthreads=10)

        ssl_adapter = BuiltinSSLAdapter(server_cert, server_key, ca_cert)
        ssl_adapter.context.verify_mode = ssl.CERT_REQUIRED
        server.ssl_adapter = ssl_adapter

        try:
            server.start()
        except KeyboardInterrupt:
            print("\nShutting down...")
            app.scheduler.cancel_all_timers()
            server.stop()
