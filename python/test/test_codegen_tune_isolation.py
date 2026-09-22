# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression test for the codegen -> aotriton.tune back-edge invariant.

The code generator (python/codegen/, python/template_instantiation/,
modules/*/aot/) must never import anything from `aotriton.tune`: `aotriton`
and `aotriton-tune` (python/tune/) are separate installable distributions, and
the main `aotriton` wheel never bundles tuning infrastructure (see
modules/flash/aot/sancheck.py, which used to reach into
modules/flash/tune/sancheck.py through aotriton.tune.registry.load_family_tune
before that back-edge was severed).

Loading the real flash family's `aot` package (via
`aotriton.codegen.parser.Parser.load_family_aot`, the exact call
`ir/triton/kdesc.py`'s `_lut_sancheck` makes) is enough to reach
`aot.sancheck.LutSancheck` -- the class this whole back-edge existed for. If
loading it pulls in a single `aotriton.tune*` module, the back-edge is back.

The check runs in a FRESH subprocess, not against this process's own
`sys.modules`. Two reasons, both real:
  * This file can run in the same pytest session as test_tune_infra.py (see
    the combined invocation in the verification plan), which legitimately
    imports `aotriton.tune.*` for unrelated reasons. Checking THIS process's
    `sys.modules` for bare absence would then fail for a reason that has
    nothing to do with the invariant.
  * The reverse failure mode is just as real: if `aotriton.tune` is already
    cached in `sys.modules` (again, e.g. because test_tune_infra.py ran
    first), a regressed `_gen_missing_entries`/`_common.py` that reaches back
    into `aotriton.tune` would silently reuse that cached module instead of
    importing anything NEW -- a before/after diff of the CURRENT process's
    `sys.modules` would then miss the regression entirely (false negative).
A clean subprocess starts with neither problem: nothing has imported
`aotriton.tune` yet, so any import of it triggered by loading `aot` shows up,
every time, regardless of what else ran earlier in this pytest session.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODULES_DIR = _REPO_ROOT / 'modules'

# Runs in a fresh `python -c`, not imported: see the module docstring for why
# this must not just inspect the current process's sys.modules.
_CHECK_SCRIPT = f'''
import inspect
import sys

sys.path.insert(0, {str(_REPO_ROOT)!r})

from aotriton.codegen.parser import Parser

aot = Parser({str(_MODULES_DIR)!r}).load_family_aot('flash')

# Not vacuous: the load step must actually resolve LutSancheck, or the tune-
# module check below would pass for the wrong reason (nothing loaded at all).
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
