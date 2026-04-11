#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Manage GPU worker configurations in SQLite database."""

import argparse
import sqlite3
import sys
from pathlib import Path
from datetime import datetime

# Import supported architectures
sys.path.insert(0, str(Path(__file__).parent.parent / 'v3python'))
from gpu_targets import AOTRITON_ARCH_TO_PACK

SUPPORTED_ARCHS = set(AOTRITON_ARCH_TO_PACK.keys())


class WorkerManager:
    def __init__(self, workdir):
        self.workdir = Path(workdir)
        if not self.workdir.is_dir():
            sys.exit(f"Error: Working directory '{workdir}' does not exist")

        self.rcfile = self.workdir / "config.rc"
        if not self.rcfile.exists():
            sys.exit(f"Error: Config file '{self.rcfile}' not found. Did you run create-project-directory.sh?")

        self.db_file = self.workdir / "workers.db"
        self.init_db()

    def init_db(self):
        """Initialize database schema if it doesn't exist."""
        with sqlite3.connect(self.db_file) as conn:
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
                    gres TEXT NOT NULL UNIQUE,
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

    def add_workers(self, arch, hostnames, workdir_override=None):
        """Add GPU workers for a given architecture."""
        if arch not in SUPPORTED_ARCHS:
            sys.exit(f"Error: Unsupported architecture '{arch}'. Supported: {', '.join(sorted(SUPPORTED_ARCHS))}")

        added = []
        errors = []

        for hostname in hostnames:
            try:
                with sqlite3.connect(self.db_file) as conn:
                    conn.execute("INSERT INTO workers (hostname, arch, workdir_override) VALUES (?, ?, ?)",
                               (hostname, arch, workdir_override))
                added.append(hostname)
            except sqlite3.IntegrityError:
                errors.append(f"Worker '{hostname}' already exists")

        # Report results
        if added:
            workdir_msg = f" with custom workdir: {workdir_override}" if workdir_override else " (will use default workdir)"
            print(f"Successfully added {len(added)} worker(s) for arch '{arch}'{workdir_msg}:")
            for hostname in added:
                print(f"  - {hostname}")

        if errors:
            print("\nErrors:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            sys.exit(1)

    def remove_worker(self, hostname):
        """Remove a GPU worker."""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.execute("DELETE FROM workers WHERE hostname = ?", (hostname,))
            if cursor.rowcount == 0:
                sys.exit(f"Error: Worker '{hostname}' not found")
            print(f"Successfully removed worker '{hostname}'")

    def list_workers(self):
        """List all GPU workers."""
        with sqlite3.connect(self.db_file) as conn:
            # Get default workdir
            cursor = conn.execute("SELECT value FROM config WHERE key = 'default_workdir'")
            row = cursor.fetchone()
            default_workdir = row[0] if row else "<not set>"

            print("=== GPU Worker Configuration ===\n")
            print(f"Default Working Directory: {default_workdir}\n")

            # Get workers
            cursor = conn.execute("""
                SELECT hostname, arch, COALESCE(workdir_override, '<default>'), created_at
                FROM workers ORDER BY arch, hostname
            """)
            workers = cursor.fetchall()

            if not workers:
                print("No workers registered")
                return

            print("Registered Workers:")
            print("-" * 100)
            print(f"{'Hostname':<35} {'Arch':<12} {'Working Directory':<30} {'Created':<20}")
            print("-" * 100)
            for hostname, arch, workdir, created in workers:
                print(f"{hostname:<35} {arch:<12} {workdir:<30} {created:<20}")
            print(f"\nTotal: {len(workers)} worker(s)")

    def set_default_workdir(self, path):
        """Set default working directory."""
        with sqlite3.connect(self.db_file) as conn:
            conn.execute("""
                INSERT INTO config (key, value) VALUES ('default_workdir', ?)
                ON CONFLICT(key) DO UPDATE SET value = ?, updated_at = CURRENT_TIMESTAMP
            """, (path, path))
        print(f"Successfully set default working directory to: {path}")

    def get_default_workdir(self):
        """Get default working directory."""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.execute("SELECT value FROM config WHERE key = 'default_workdir'")
            row = cursor.fetchone()
            if not row:
                sys.exit("Error: Default working directory not set. Use 'set-default-workdir' command.")
            print(row[0])

    def slurm_add(self, gres):
        """Add SLURM batch configuration."""
        try:
            with sqlite3.connect(self.db_file) as conn:
                conn.execute("INSERT INTO slurm_batch (gres) VALUES (?)", (gres,))
            print(f"Successfully added SLURM configuration with gres '{gres}'")
        except sqlite3.IntegrityError:
            sys.exit(f"Error: SLURM configuration with gres '{gres}' already exists. Use slurm-remove first.")

    def slurm_list(self):
        """List all SLURM batch configurations."""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.execute("""
                SELECT gres, created_at
                FROM slurm_batch ORDER BY gres
            """)
            configs = cursor.fetchall()

            if not configs:
                print("No SLURM batch configurations registered")
                return

            print("=== SLURM Batch Configurations ===\n")
            print("-" * 60)
            print(f"{'GRES':<40} {'Created':<20}")
            print("-" * 60)
            for gres, created in configs:
                print(f"{gres:<40} {created:<20}")
            print(f"\nTotal: {len(configs)} configuration(s)")

    def slurm_remove(self, gres):
        """Remove SLURM batch configuration."""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.execute("DELETE FROM slurm_batch WHERE gres = ?", (gres,))
            if cursor.rowcount == 0:
                sys.exit(f"Error: SLURM configuration with gres '{gres}' not found")
            print(f"Successfully removed SLURM configuration with gres '{gres}'")

    def slurm_bad_add(self, hostnames, reason=None):
        """Mark SLURM nodes as bad (to be excluded from job submissions)."""
        added = []
        errors = []

        for hostname in hostnames:
            try:
                with sqlite3.connect(self.db_file) as conn:
                    conn.execute("INSERT INTO slurm_bad_nodes (hostname, reason) VALUES (?, ?)",
                               (hostname, reason))
                added.append(hostname)
            except sqlite3.IntegrityError:
                errors.append(f"Node '{hostname}' already marked as bad")

        if added:
            reason_msg = f" (reason: {reason})" if reason else ""
            print(f"Successfully marked {len(added)} node(s) as bad{reason_msg}:")
            for hostname in added:
                print(f"  - {hostname}")

        if errors:
            print("\nErrors:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            if not added:
                sys.exit(1)

    def slurm_bad_list(self):
        """List all bad SLURM nodes."""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.execute("""
                SELECT hostname, reason, created_at
                FROM slurm_bad_nodes ORDER BY hostname
            """)
            bad_nodes = cursor.fetchall()

            if not bad_nodes:
                print("No bad nodes registered")
                return

            print("=== Bad SLURM Nodes (Excluded from Jobs) ===\n")
            print("-" * 80)
            print(f"{'Hostname':<35} {'Reason':<25} {'Created':<20}")
            print("-" * 80)
            for hostname, reason, created in bad_nodes:
                reason_str = reason if reason else "<no reason>"
                print(f"{hostname:<35} {reason_str:<25} {created:<20}")
            print(f"\nTotal: {len(bad_nodes)} bad node(s)")

    def slurm_bad_remove(self, hostnames):
        """Remove nodes from bad node list (unmark as bad)."""
        removed = []
        not_found = []

        for hostname in hostnames:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.execute("DELETE FROM slurm_bad_nodes WHERE hostname = ?", (hostname,))
                if cursor.rowcount > 0:
                    removed.append(hostname)
                else:
                    not_found.append(hostname)

        if removed:
            print(f"Successfully removed {len(removed)} node(s) from bad node list:")
            for hostname in removed:
                print(f"  - {hostname}")

        if not_found:
            print("\nNot found:", file=sys.stderr)
            for hostname in not_found:
                print(f"  - {hostname}", file=sys.stderr)
            if not removed:
                sys.exit(1)


def main():
    arch_choices = sorted(SUPPORTED_ARCHS)

    parser = argparse.ArgumentParser(
        description="Manage GPU worker configurations stored in SQLite database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Description:
  This script manages GPU worker configurations for the AOTriton distributed
  tuning framework. Worker configurations are stored in <workdir>/workers.db.

  Each worker has:
  - An architecture (GPU arch: {', '.join(arch_choices[:5])}, ...)
  - A hostname (must be unique and accessible via SSH)
  - An optional custom working directory path

  A global default working directory can be configured. Workers without a
  custom path will use this default directory.

Typical Workflow:
  1. Create a project directory: .celery/create-project-directory.sh <workdir>
  2. Set default workdir: %(prog)s <workdir> set-default-workdir /opt/aotriton/workdir
  3. Add GPU workers: %(prog)s <workdir> add <arch> <hostname1> [<hostname2> ...]
  4. List workers to verify: %(prog)s <workdir> list

Examples:
  # Set the default working directory for all workers
  %(prog)s /path/to/workdir set-default-workdir /opt/aotriton/workdir

  # Add multiple gfx90a GPU workers (will use default workdir)
  %(prog)s /path/to/workdir add gfx90a gpu-node-01.example.com gpu-node-02.example.com

  # Add a single gfx942 worker
  %(prog)s /path/to/workdir add gfx942 gpu-node-03.example.com

  # Add workers with custom working directory
  %(prog)s /path/to/workdir add gfx1100 gpu-node-04.example.com --workdir /custom/path

  # List all registered workers
  %(prog)s /path/to/workdir list

  # Remove a worker
  %(prog)s /path/to/workdir remove gpu-node-01.example.com

  # Get the default working directory
  %(prog)s /path/to/workdir get-default-workdir

Supported Architectures:
  {', '.join(arch_choices)}

Notes:
  - Worker hostnames must be accessible from the dev node via SSH
  - The working directory path should exist on each GPU worker node
  - Worker configurations persist in <workdir>/workers.db
        """
    )

    parser.add_argument("workdir", help="Project working directory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Add command
    add_parser = subparsers.add_parser("add", help="Add GPU workers")
    add_parser.add_argument("arch", choices=arch_choices, help="GPU architecture")
    add_parser.add_argument("hostnames", nargs="+", help="One or more fully qualified hostnames")
    add_parser.add_argument("--workdir", dest="workdir_override", help="Optional custom working directory for these workers")

    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove a GPU worker")
    remove_parser.add_argument("hostname", help="Hostname to remove")

    # List command
    subparsers.add_parser("list", help="List all GPU workers")

    # Set default workdir
    set_parser = subparsers.add_parser("set-default-workdir", help="Set default working directory")
    set_parser.add_argument("path", help="Path for default working directory")

    # Get default workdir
    subparsers.add_parser("get-default-workdir", help="Get default working directory")

    # SLURM commands
    slurm_add_parser = subparsers.add_parser("slurm-add", help="Add SLURM batch configuration")
    slurm_add_parser.add_argument("gres", help="SLURM gres constraint (e.g., gpu:gfx942-mi300x:8 or gpu:gfx1100w:4)")

    subparsers.add_parser("slurm-list", help="List all SLURM batch configurations")

    slurm_remove_parser = subparsers.add_parser("slurm-remove", help="Remove SLURM batch configuration")
    slurm_remove_parser.add_argument("gres", help="SLURM gres constraint to remove")

    # SLURM bad node commands
    slurm_bad_add_parser = subparsers.add_parser("slurm-bad-add", help="Mark SLURM nodes as bad (exclude from jobs)")
    slurm_bad_add_parser.add_argument("hostnames", nargs="+", help="One or more hostnames to mark as bad")
    slurm_bad_add_parser.add_argument("--reason", help="Reason for marking nodes as bad")

    subparsers.add_parser("slurm-bad-list", help="List all bad SLURM nodes")

    slurm_bad_remove_parser = subparsers.add_parser("slurm-bad-remove", help="Remove nodes from bad node list")
    slurm_bad_remove_parser.add_argument("hostnames", nargs="+", help="One or more hostnames to unmark as bad")

    args = parser.parse_args()

    manager = WorkerManager(args.workdir)

    if args.command == "add":
        manager.add_workers(args.arch, args.hostnames, args.workdir_override)
    elif args.command == "remove":
        manager.remove_worker(args.hostname)
    elif args.command == "list":
        manager.list_workers()
    elif args.command == "set-default-workdir":
        manager.set_default_workdir(args.path)
    elif args.command == "get-default-workdir":
        manager.get_default_workdir()
    elif args.command == "slurm-add":
        manager.slurm_add(args.gres)
    elif args.command == "slurm-list":
        manager.slurm_list()
    elif args.command == "slurm-remove":
        manager.slurm_remove(args.gres)
    elif args.command == "slurm-bad-add":
        manager.slurm_bad_add(args.hostnames, getattr(args, 'reason', None))
    elif args.command == "slurm-bad-list":
        manager.slurm_bad_list()
    elif args.command == "slurm-bad-remove":
        manager.slurm_bad_remove(args.hostnames)


if __name__ == "__main__":
    main()
