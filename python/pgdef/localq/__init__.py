# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Per-node runtime: a Unix-socket broker, pull-based workers and PG readers.

Messages are length-prefixed JSON (`protocol.py`). The broker routes them
between named queues and resolves dependencies; it knows no application.
"""
