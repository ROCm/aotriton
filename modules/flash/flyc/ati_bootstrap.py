# SPDX-License-Identifier: Apache-2.0

"""Make `aotriton` importable from the checkout this flyc tree sits in.

The tuning modules write their per-call rules with ATI (`ati.derives`), but a
JIT run -- the tests, the coverage harness, the benches -- has flydsl and this
checkout, not an installed `aotriton`. Importing this module registers the
checkout's `python/` as the `aotriton` package, unless one is already loaded:
inside the code generator and `aotriton.flyc_compile` it is, and they keep the
`aotriton` they run as. Otherwise the checkout wins over an installed copy on
purpose, since a stale install would hand the tuning modules a different ATI.
"""

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

if 'aotriton' not in sys.modules:
    _spec = importlib.machinery.ModuleSpec('aotriton', None, is_package=True)
    _spec.submodule_search_locations = [str(Path(__file__).resolve().parents[3] / 'python')]
    sys.modules['aotriton'] = importlib.util.module_from_spec(_spec)
