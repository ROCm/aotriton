# SPDX-License-Identifier: Apache-2.0

"""Compile from source: FlyDSL's JIT cache key leaves out helpers outside the
kernel's own directory (`fmha_utils.py`, `philox.py`), so a cached binary can
outlive an edit to them."""

import os

os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
