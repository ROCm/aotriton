# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""perfmon test fixtures. The helpers of this directory (perfmon_fakes.py)
are importable by name; the pgdef test fakes (localq_harness, pgfake) come
from python/test/conftest.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
