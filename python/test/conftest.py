# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""pytest configuration for the ATI generator unit tests.

The suite is self-contained: it exercises the ATI machinery against fake, minimal
kernels (fakekernels.py) and a fake flash family (fakefamily/), with NO dependency
on the real flash sources under modules/. We only put the test dir on sys.path so
`import registry` / `import fakekernels` resolve.
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # python/test/

sys.path.insert(0, str(_HERE))                   # for registry.py / fakekernels.py
