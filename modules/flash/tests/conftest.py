# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""pytest configuration for the flash AOT/pyaotriton integration+perf suite.

Putting the suite dir on sys.path lets pytest collect the whole directory in one
invocation (`pytest modules/flash/tests`) — the suite's sibling modules
(_common_test, attn_torch_function, aotriton_flash, fwd_kernel, tuned_bwd, …)
import cleanly without naming individual test files.

test_forward.py is intentionally EXCLUDED from directory collection: the forward
kernel is already exercised (with broader coverage) as part of test_backward.py,
which runs the forward pass to produce the inputs it differentiates. test_forward.py
is kept only as a standalone entry point for users who want to check the forward
kernel in isolation — run it explicitly (`pytest .../test_forward.py`); its coverage
is a subset of test_backward.py's.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Exclude the forward-only suite from `pytest modules/flash/tests` (see docstring).
collect_ignore = ['test_forward.py']
