# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""pytest configuration for the flash AOT/pyaotriton integration+perf suite.

Putting the suite dir on sys.path lets pytest collect the whole directory in one
invocation (`pytest modules/flash/tests`) — the suite's sibling modules
(_common_test, attn_torch_function, aotriton_flash, fwd_kernel, tuned_bwd, …)
import cleanly without naming individual test files.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
