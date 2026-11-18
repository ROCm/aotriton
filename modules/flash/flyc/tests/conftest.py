# SPDX-License-Identifier: Apache-2.0

"""Setup for the arch-neutral JIT suites.

Compile from source: FlyDSL's JIT cache key leaves out helpers outside the
kernel's own directory (`fmha_utils.py`, `philox.py`), so a cached binary can
outlive an edit to them.

`flyc/` goes on `sys.path` for the dispatchers and shared modules (this
directory has no `__init__.py`, so pytest adds only `tests/`), and the FlyDSL
checkout for the `kernels.*` modules the arch kernels import.
"""

import os
import sys
from pathlib import Path

os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"

_FLYC = str(Path(__file__).resolve().parents[1])
if _FLYC not in sys.path:
    sys.path.insert(0, _FLYC)

_KERNEL_ROOT = os.environ.get("AOTRITON_FLYDSL_KERNEL_ROOT")
if _KERNEL_ROOT and _KERNEL_ROOT not in sys.path:
    sys.path.append(_KERNEL_ROOT)
