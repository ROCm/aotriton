#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Compatibility shim: define sqlite3() using Python's built-in sqlite3 module
# when the sqlite3 CLI binary is not available (e.g. inside CELERY_WORKER_IMAGE).
# Supports the subset of sqlite3 CLI used in this project:
#   sqlite3 <db_path> <sql>
# Output format matches the default sqlite3 pipe-separated output.

if ! command -v sqlite3 &>/dev/null; then
    sqlite3() {
        local db="$1"
        local sql="$2"
        python3 -c "
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
for row in conn.execute(sys.argv[2]):
    print('|'.join('' if v is None else str(v) for v in row))
" "$db" "$sql"
    }
fi
