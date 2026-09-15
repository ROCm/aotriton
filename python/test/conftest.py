# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""pytest configuration for the ATI generator unit tests.

The suite is self-contained: it exercises the ATI machinery against fake, minimal
kernels (fakekernels.py) and a fake flash family (fakefamily/), with NO dependency
on the real flash sources under modules/. We only put the test dir on sys.path so
`import registry` / `import fakekernels` resolve.

The exceptions are test_gpu_targets.py, which shells out to the real installed
`aotriton.gpu_targets` CLI rather than a fake, and test_buildable_decl.py, which
links the real modules/ tree for a genuine FlycDecl (the fake family has no flyc
description). Neither needs a GPU.
"""

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent          # python/test/

sys.path.insert(0, str(_HERE))                   # for registry.py / fakekernels.py


@pytest.fixture(autouse=True)
def _one_tree_per_test():
    """A family binds to one tree per process (codegen/parser.py), but this suite
    links both the fake family and the real modules/ tree. Reset between tests so
    the pair is order-independent rather than whichever ran first winning."""
    from aotriton.codegen.parser import reset_loaded_aot
    reset_loaded_aot()
    yield
    reset_loaded_aot()
