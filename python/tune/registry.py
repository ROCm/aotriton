# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Tuning module registry.

Loads `modules/<family>/tune/` BY PATH under a synthetic top-level name
(mirroring `python/codegen/parser.py`'s `load_family_aot`), and resolves the
flat module names used throughout the tuning CLI/queue (`'flash'`,
`'flash_op'`, ...) to the loaded submodule that exports `TuneDesc` /
`ImplSelector`.

Phase 1 (pure relocation, see modular-tune.md): `'flash'` and `'flash_op'`
are two independent subpackages under `modules/flash/tune/`, each still
exporting its own `TuneDesc`/`ImplSelector`. Both stay registered as separate
CLI subcommands / queue `module` keys here -- unifying them into one
`FlashTune` description with a `tuning_mode`/`level` axis is Phase 2 (§4.1-§4.3
of the plan) and is explicitly out of scope for this registry.
"""

import os
import sys
import importlib
import importlib.util
from pathlib import Path

# Flat module name (as used in the CLI / task_queue.module column) ->
# (family, submodule name under modules/<family>/tune/).
#
# This is a static list, not a filesystem scan (F8): 'flash'/'flash_op' now
# live under modules/flash/tune/, outside this package, so a directory glob
# rooted here can no longer discover them. Adding a new tuning module means
# adding an entry here.
_MODULE_TO_FAMILY = {
    'flash': ('flash', 'flash'),
    'flash_op': ('flash', 'flash_op'),
}


def available_module_names() -> list[str]:
    """Flat module names registered for the tuning CLI/queue."""
    return sorted(_MODULE_TO_FAMILY)


def default_modules_dir() -> Path:
    """Resolve the repo's `modules/` directory.

    `modules/` is DATA beside the package source (like codegen's --root_dir),
    never shipped inside the installed `aotriton` package (see setup.py), so
    it cannot be reliably derived from this file's location alone -- a
    non-editable install copies `python/` out of the checkout but never
    ships `modules/` alongside it.

    Resolution order:
      1. `AOTRITON_MODULES_DIR` env var, if set -- the explicit override.
      2. `<cwd>/modules` -- all `.tune/bin/*` / `.tune/remote/*` wrappers `cd`
         into `$AOTRITON_ROOT` before invoking `python3 -m aotriton.tune.*`
         (see e.g. `.tune/bin/dispatch`), so this covers every production
         call site.
      3. `<repo root inferred from this file>/modules` -- last-resort
         fallback for ad-hoc invocation (e.g. running pytest from the repo
         root with an editable install); only correct when `aotriton` is
         installed editable from this exact checkout.
    """
    env = os.environ.get('AOTRITON_MODULES_DIR')
    if env:
        return Path(env)
    cwd_candidate = Path.cwd() / 'modules'
    if cwd_candidate.is_dir():
        return cwd_candidate
    return Path(__file__).resolve().parent.parent.parent / 'modules'


def load_family_tune(family: str, modules_dir: 'Path | None' = None):
    """Import `<modules_dir>/<family>/tune/__init__.py` by path under a
    synthetic unique package name, so `modules/<family>` stays a plain
    directory (not a package) -- identical rationale to
    `python/codegen/parser.py`'s `load_family_aot` (see also
    `modules/flash/tune/__init__.py`). Cached in `sys.modules`.
    """
    modname = f'_aotriton_modules_{family}_tune'
    cached = sys.modules.get(modname)
    if cached is not None:
        return cached
    if modules_dir is None:
        modules_dir = default_modules_dir()
    tune_dir = Path(modules_dir) / family / 'tune'
    init_path = tune_dir / '__init__.py'
    if not init_path.is_file():
        raise ImportError(
            f"No tune block for family '{family}': {init_path} not found "
            f"(modules_dir={modules_dir}). Set AOTRITON_MODULES_DIR or run "
            f"from the repo root.")
    spec = importlib.util.spec_from_file_location(
        modname, init_path, submodule_search_locations=[str(tune_dir)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def load_tune_module(module_name: str, modules_dir: 'Path | None' = None):
    """Resolve a flat tuning module name (e.g. `'flash'`, `'flash_op'`) to
    its submodule under `modules/<family>/tune/`. The returned submodule
    exports `TuneDesc` and `ImplSelector`, exactly like the
    pre-modularization `v3python.tune.<name>` packages did.
    """
    try:
        family, submodule_name = _MODULE_TO_FAMILY[module_name]
    except KeyError:
        raise ImportError(
            f"Unknown tuning module '{module_name}'. "
            f"Available: {available_module_names()}")
    family_pkg = load_family_tune(family, modules_dir=modules_dir)
    return importlib.import_module(f'.{submodule_name}', package=family_pkg.__name__)


def load_flash_entry_module(modules_dir: 'Path | None' = None):
    """Return `modules/flash/tune/flash/module.py` -- the module defining
    `FlashEntry` / `FlashInputMetadata`.

    A handful of non-GPU tools (`.tune/bin/retry_missing_entries`,
    `.tune/libexec/broken_entries_to_db`, `.tune/libexec/pq_helpers.py`,
    `.tune/webui/tasks.py`) need `FlashEntry`'s dataclass shape
    (`as_text`/`parse_text`) without pulling in `KernelControl`. `FlashEntry`
    is torch-free at import time (see `module.py`), so importing it here is
    always safe, even outside a GPU container.

    This is the one case a caller cannot just do `from aotriton.tune.X import
    Y`: the family package lives outside the `aotriton` tree, under
    `modules/`, so it must go through the by-path loader above.
    """
    return load_tune_module('flash', modules_dir=modules_dir).module
