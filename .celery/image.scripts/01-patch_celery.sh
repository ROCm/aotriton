#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Apply patch to Celery to use JSON instead of PickleType for database storage.
#
# This script patches the installed Celery package to use SQLAlchemy's JSON type
# instead of PickleType for storing task results and tasksets in the database.
# This is necessary to use PostgreSQL as a schemaless database.
#
# Usage:
#   bash 01-patch_celery.sh
#   CONFIG_RC=/path/to/config.rc bash 01-patch_celery.sh
#   CELERY_TO_PATCH_PYTHON=/path/to/python bash 01-patch_celery.sh
#
# Environment Variables:
#   CONFIG_RC (optional):
#     Path to config.rc file. Defaults to /config.rc.
#     The config.rc should define CELERY_WORKER_PYTHON.
#
#   CELERY_TO_PATCH_PYTHON (optional):
#     Python executable path whose celery installation should be patched.
#     If not set, uses CELERY_WORKER_PYTHON from config.rc.
#     This allows overriding the python path without modifying config.rc.
#
# Examples:
#   # Use default config.rc at /config.rc
#   bash 01-patch_celery.sh
#
#   # Use custom config.rc
#   CONFIG_RC=/workdir/config.rc bash 01-patch_celery.sh
#
#   # Patch a specific python's celery installation
#   CELERY_TO_PATCH_PYTHON=/venv/bin/python bash 01-patch_celery.sh
#
#   # Use custom config but override the python to patch
#   CONFIG_RC=/workdir/config.rc CELERY_TO_PATCH_PYTHON=/custom/venv/bin/python bash 01-patch_celery.sh

CONFIG_RC="${CONFIG_RC:-/config.rc}"

if [ ! -f "$CONFIG_RC" ]; then
  echo "Error: config.rc not found at $CONFIG_RC" >&2
  exit 1
fi

# Source config to get CELERY_WORKER_PYTHON
. "$CONFIG_RC"

# CELERY_TO_PATCH_PYTHON can override CELERY_WORKER_PYTHON
CELERY_TO_PATCH_PYTHON="${CELERY_TO_PATCH_PYTHON:-$CELERY_WORKER_PYTHON}"

if [ -z "$CELERY_TO_PATCH_PYTHON" ]; then
  echo "Error: CELERY_TO_PATCH_PYTHON not set" >&2
  exit 1
fi

# Get celery installation location
CELERY_LOCATION=$($CELERY_TO_PATCH_PYTHON -c 'import celery; from pathlib import Path; print((Path(celery.__file__).parent.parent).as_posix())')

if [ -z "$CELERY_LOCATION" ]; then
  echo "Error: Could not find celery installation location" >&2
  exit 1
fi

echo "Found celery at: $CELERY_LOCATION"

# Apply patch
apply_patch() {
git apply "$@" - 2>/dev/null << 'EOF'
From 7849070ce374321810be855a17d08418a69d110c Mon Sep 17 00:00:00 2001
From: Xinya Zhang <Xinya.Zhang@amd.com>
Date: Tue, 31 Mar 2026 22:09:37 +0000
Subject: [PATCH] db: use sqlalchemy.dialects.postgresql.JSONB as database type
 for result

This hack breaks compatibility with non-pg databases
---
 celery/backends/database/models.py | 6 +++---
 1 file changed, 3 insertions(+), 3 deletions(-)

diff --git a/celery/backends/database/models.py b/celery/backends/database/models.py
index f8ee62393..8237a77ff 100644
--- a/celery/backends/database/models.py
+++ b/celery/backends/database/models.py
@@ -2,7 +2,7 @@
 from datetime import datetime, timezone

 import sqlalchemy as sa
-from sqlalchemy.types import PickleType
+from sqlalchemy.dialects.postgresql import JSONB

 from celery import states

@@ -34,7 +34,7 @@ class Task(ResultModelBase):
                    primary_key=True, autoincrement=True)
     task_id = sa.Column(sa.String(155), unique=True)
     status = sa.Column(sa.String(50), default=states.PENDING)
-    result = sa.Column(PickleType, nullable=True)
+    result = sa.Column(JSONB, nullable=True)
     date_done = sa.Column(sa.DateTime, default=_get_utc_now,
                           onupdate=_get_utc_now, nullable=True, index=True)
     traceback = sa.Column(sa.Text, nullable=True)
@@ -96,7 +96,7 @@ class TaskSet(ResultModelBase):
     id = sa.Column(DialectSpecificInteger, sa.Sequence('taskset_id_sequence'),
                    autoincrement=True, primary_key=True)
     taskset_id = sa.Column(sa.String(155), unique=True)
-    result = sa.Column(PickleType, nullable=True)
+    result = sa.Column(JSONB, nullable=True)
     date_done = sa.Column(sa.DateTime, default=_get_utc_now,
                           nullable=True, index=True)

--
2.34.1

EOF
}

cd "$CELERY_LOCATION"
if apply_patch --check; then
  apply_patch
  echo "Patch applied successfully"
elif apply_patch --reverse --check; then
  echo "Patch already applied, skipping"
else
  echo "Error: Patch cannot be applied (conflicts or already modified)" >&2
  exit 1
fi
