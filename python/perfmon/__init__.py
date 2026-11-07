# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
perfmon: end-to-end performance of AOTriton's public APIs, release over
release, as an application on pgdef (aotriton.pgdef).

Entry points: python -m aotriton.perfmon.{node,dispatch,manifest,export,perfdb}.
The handler graph is handlers.py; the PostgreSQL layer is pq/ (schema
`perfmon`); the repository format is perfdb.py, see
docs/Human Readable Performance Database Format.md.

This package file imports nothing on purpose: family plugins import
`aotriton.perfmon.pdesc`, and must not pull in the engine by doing so.
"""
