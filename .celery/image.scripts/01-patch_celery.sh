#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

CONFIG_RC="/config.rc"

if [ ! -f "$CONFIG_RC" ]; then
  echo "Error: config.rc not found at $CONFIG_RC" >&2
  exit 1
fi

# Source config to get CELERY_WORKER_PYTHON
. "$CONFIG_RC"

if [ -z "$CELERY_WORKER_PYTHON" ]; then
  echo "Error: CELERY_WORKER_PYTHON not set in config.rc" >&2
  exit 1
fi

# Get celery installation location
CELERY_LOCATION=$($CELERY_WORKER_PYTHON -c 'import celery; from pathlib import Path; print((Path(celery.__file__).parent.parent).as_posix())')

if [ -z "$CELERY_LOCATION" ]; then
  echo "Error: Could not find celery installation location" >&2
  exit 1
fi

echo "Found celery at: $CELERY_LOCATION"

# Apply patch
apply_patch() {
git apply "$@" - 2>/dev/null << 'EOF'
From 64ae6737ef4ec49d02f62e55545e491264df367b Mon Sep 17 00:00:00 2001
From: Xinya Zhang <Xinya.Zhang@amd.com>
Date: Tue, 31 Mar 2026 22:09:37 +0000
Subject: [PATCH] db: use sa.JSON as database type for result

---
 celery/backends/database/models.py | 6 +++---
 1 file changed, 3 insertions(+), 3 deletions(-)

diff --git a/celery/backends/database/models.py b/celery/backends/database/models.py
index f8ee62393..6e8470da8 100644
--- a/celery/backends/database/models.py
+++ b/celery/backends/database/models.py
@@ -2,7 +2,7 @@
 from datetime import datetime, timezone

 import sqlalchemy as sa
-from sqlalchemy.types import PickleType
+from sqlalchemy.types import JSON

 from celery import states

@@ -34,7 +34,7 @@ class Task(ResultModelBase):
                    primary_key=True, autoincrement=True)
     task_id = sa.Column(sa.String(155), unique=True)
     status = sa.Column(sa.String(50), default=states.PENDING)
-    result = sa.Column(PickleType, nullable=True)
+    result = sa.Column(JSON, nullable=True)
     date_done = sa.Column(sa.DateTime, default=_get_utc_now,
                           onupdate=_get_utc_now, nullable=True, index=True)
     traceback = sa.Column(sa.Text, nullable=True)
@@ -96,7 +96,7 @@ class TaskSet(ResultModelBase):
     id = sa.Column(DialectSpecificInteger, sa.Sequence('taskset_id_sequence'),
                    autoincrement=True, primary_key=True)
     taskset_id = sa.Column(sa.String(155), unique=True)
-    result = sa.Column(PickleType, nullable=True)
+    result = sa.Column(JSON, nullable=True)
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
