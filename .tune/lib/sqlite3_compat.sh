#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Compatibility shim: define sqlite3() using Python's built-in sqlite3 module
# when the sqlite3 CLI binary is not available (e.g. inside CELERY_WORKER_IMAGE).
# Supports the subset of sqlite3 CLI used in this project:
#   sqlite3 <db_path> <sql>
#   sqlite3 <db_path> <<SQL ... SQL   (sql read from stdin when <sql> is omitted)
# <sql> may hold multiple ';'-separated statements (e.g. several INSERTs).
# Multi-statement scripts are run via executescript() (SQLite's own parser,
# so values containing a literal ';' are handled correctly) rather than a
# naive string split; single statements use execute() so SELECT rows are
# still returned. Output format matches the default sqlite3 pipe-separated
# output.

if ! command -v sqlite3 &>/dev/null; then
    sqlite3() {
        local db="$1"
        local sql="$2"
        if [ -z "$sql" ] && [ ! -t 0 ]; then
            sql="$(cat)"
        fi
        python3 -c "
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
sql = sys.argv[2]
try:
    cur = conn.execute(sql)
    for row in cur:
        print('|'.join('' if v is None else str(v) for v in row))
except sqlite3.ProgrammingError:
    # More than one statement -- execute() only accepts one; executescript()
    # handles the rest via SQLite's real parser but never returns rows.
    conn.executescript(sql)
conn.commit()
" "$db" "$sql"
    }
fi
