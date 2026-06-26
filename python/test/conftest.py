# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""pytest configuration for ATI unit tests.

Sets up sys.path so tests can import aotriton, the flash kernel sources
(modules/flash, tritonsrc), and the local registry helper — without each
test file needing to know its depth below the repo root.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # python/test/
_REPO = _HERE.parents[2]                          # repo root

sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / 'tritonsrc'))
sys.path.insert(0, str(_REPO / 'modules' / 'flash'))
sys.path.insert(0, str(_REPO / 'modules' / 'flash' / 'kernel'))
sys.path.insert(0, str(_HERE))                   # for registry.py
