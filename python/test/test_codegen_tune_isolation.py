# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression test for the codegen -> aotriton.tune back-edge invariant.

The code generator (python/codegen/, python/template_instantiation/,
modules/*/aot/) must never import anything from `aotriton.tune`: `aotriton`
and `aotriton-tune` are separate distributions, and the main wheel never
bundles tuning infrastructure.

Loading the real flash family's `aot` package (the same call `kdesc.py`'s
`_lut_sancheck` makes) is enough to reach `aot.sancheck.LutSancheck` -- the
class this back-edge existed for. If loading it pulls in any `aotriton.tune*`
module, the back-edge is back.

Runs the check in a FRESH subprocess rather than against this process's own
`sys.modules`: another test (python/tune/tests/test_tune_infra.py) may legitimately
import aotriton.tune first in the same pytest session, which would either
mask a real regression (already cached, so "nothing new" gets imported) or
falsely flag one (bare-presence check). A clean subprocess has neither problem.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULES_DIR = _REPO_ROOT / 'modules'

# Runs in a fresh `python -c` -- see module docstring for why.
_CHECK_SCRIPT = f'''
import inspect
import sys

sys.path.insert(0, {str(_REPO_ROOT)!r})

from aotriton.codegen.parser import Parser

aot = Parser({str(_MODULES_DIR)!r}).load_family_aot('flash')

# Not vacuous: must actually resolve LutSancheck, else the check below
# would pass for the wrong reason.
assert inspect.isclass(aot.sancheck.LutSancheck), (
    f"aot.sancheck.LutSancheck did not resolve to a class: {{aot.sancheck.LutSancheck!r}}")

tune_modules = sorted(m for m in sys.modules
                       if m == "aotriton.tune" or m.startswith("aotriton.tune."))
print("TUNE_MODULES:" + ",".join(tune_modules))
'''


def test_loading_flash_aot_does_not_import_tune():
    result = subprocess.run(
        [sys.executable, '-c', _CHECK_SCRIPT],
        capture_output=True, text=True, cwd=str(_REPO_ROOT))
    assert result.returncode == 0, (
        f"the check subprocess itself failed (LutSancheck likely did not "
        f"resolve): returncode={result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}")

    marker_lines = [l for l in result.stdout.splitlines() if l.startswith('TUNE_MODULES:')]
    assert marker_lines, (
        f"check subprocess did not print the expected marker line; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}")
    tune_modules = [m for m in marker_lines[-1][len('TUNE_MODULES:'):].split(',') if m]
    assert not tune_modules, (
        f"loading modules/flash/aot pulled in tune modules: {tune_modules} -- "
        f"the codegen -> aotriton.tune back-edge has regressed")


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
    print(f'OK: {len(fns)} codegen/tune isolation tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
