# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Tuning module registry.

Loads `modules/<family>/tune/` BY PATH under a synthetic top-level name
(mirroring `python/codegen/parser.py`'s `load_family_aot`), and resolves the
flat family names used throughout the tuning CLI/queue (`'flash'`, ...) to the
loaded package that exports `TuneDesc` / `ImplSelector`.

Phase 2 (modularization unification, modular-tune.md §4.1-§4.3): the
Phase-1 `flash`/`flash_op` module-name split is gone. Each family now exposes
ONE `TuneDesc` (a `TuningDescription` subclass) -- so the registry only needs
to resolve a bare family name, not a (family, submodule) pair.

Revision note 3: `TuneDesc` takes no `level=`/tuning-level constructor
argument -- it lists and resolves impls of every tuning level directly, keyed
by DSL name (see `aotriton.tune.tdesc.TuningDescription`).
"""

import os
import sys
import importlib
import importlib.util
from pathlib import Path

# Family names (as used in the CLI / task_queue.module column) registered
# under modules/<family>/tune/.
#
# This is a static list, not a filesystem scan (F8): family tune blocks live
# outside this package, so a directory glob rooted here can no longer
# discover them. Adding a new tuning family means adding an entry here.
_FAMILIES = ('flash',)


def available_module_names() -> list[str]:
    """Flat family names registered for the tuning CLI/queue."""
    return sorted(_FAMILIES)


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


def load_family_visperf(family: str, modules_dir: 'Path | None' = None):
    """Import `<modules_dir>/<family>/visperf/__init__.py` by path under a
    synthetic unique package name, mirroring `load_family_tune` above
    (identical rationale, F6: `modules/<family>` stays a plain directory,
    not a package). Cached in `sys.modules`.

    The returned module exports `DESCRIPTOR` (a dict consumed by
    `aotriton.tune.pq.visperf`'s query builder and the webui's perf page)
    and, by convention, ships its JS counterpart at
    `<modules_dir>/<family>/visperf/static/<family>.js` -- served through
    the webui's `/family_static/<family>/<path:filename>` route and
    inlined by `aotriton.tune.pq.export_visperf` for the standalone export
    (modular-tune.md §3d).
    """
    modname = f'_aotriton_modules_{family}_visperf'
    cached = sys.modules.get(modname)
    if cached is not None:
        return cached
    if modules_dir is None:
        modules_dir = default_modules_dir()
    visperf_dir = Path(modules_dir) / family / 'visperf'
    init_path = visperf_dir / '__init__.py'
    if not init_path.is_file():
        raise ImportError(
            f"No visperf block for family '{family}': {init_path} not found "
            f"(modules_dir={modules_dir}). Set AOTRITON_MODULES_DIR or run "
            f"from the repo root.")
    spec = importlib.util.spec_from_file_location(
        modname, init_path, submodule_search_locations=[str(visperf_dir)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def load_tune_module(module_name: str, modules_dir: 'Path | None' = None):
    """Resolve a flat family name (e.g. `'flash'`) to its tune package under
    `modules/<family>/tune/`. The returned package exports `TuneDesc` and
    `ImplSelector`.

    Kept as a thin alias of `load_family_tune` (same signature/behavior) for
    call-site compatibility with pre-unification code that spoke of "tuning
    modules" rather than "tuning families" -- now that `flash`/`flash_op`
    collapsed into one `flash` family (modular-tune.md §4.3), the two
    concepts are the same thing.
    """
    if module_name not in _FAMILIES:
        raise ImportError(
            f"Unknown tuning module '{module_name}'. "
            f"Available: {available_module_names()}")
    return load_family_tune(module_name, modules_dir=modules_dir)


def make_tune_desc(family: str, modules_dir: 'Path | None' = None):
    """Convenience: resolve `family` and construct its `TuneDesc()` in one
    call."""
    return load_family_tune(family, modules_dir=modules_dir).TuneDesc()


def load_flash_entry_module(modules_dir: 'Path | None' = None):
    """Return `modules/flash/tune/entry.py` -- the module defining
    `FlashEntry` / `FlashInputMetadata`.

    A handful of non-GPU tools (`.tune/bin/retry_missing_entries`,
    `.tune/libexec/broken_entries_to_db`, `.tune/libexec/pq_helpers.py`,
    `.tune/webui/tasks.py`) need `FlashEntry`'s dataclass shape
    (`as_text`/`parse_text`) without pulling in `KernelControl`. `FlashEntry`
    is torch-free at import time (see `entry.py`), so importing it here is
    always safe, even outside a GPU container.

    This is the one case a caller cannot just do `from aotriton.tune.X import
    Y`: the family package lives outside the `aotriton` tree, under
    `modules/`, so it must go through the by-path loader above.
    """
    family_pkg = load_family_tune('flash', modules_dir=modules_dir)
    return importlib.import_module('.entry', package=family_pkg.__name__)
