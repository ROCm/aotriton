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

import importlib
import os
from pathlib import Path

from aotriton.pgdef.utils import (default_modules_dir, load_family_block,  # noqa: F401  (re-exported)
                                  tree_cache_key as _tree_cache_key)


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


def default_tune_root() -> Path:
    '''Resolve the repo's `.tune/` directory.

    `.tune/` is operator tooling run straight from a checkout and never
    installed (setup.py ships `python/` only), so an installed package cannot
    derive it from its own location. Same resolution order as
    default_modules_dir(), for the same reason:

      1. `AOTRITON_TUNE_ROOT`, the explicit override.
      2. `<cwd>/.tune` -- the `.tune/bin/*` wrappers cd into $AOTRITON_ROOT
         before invoking `python3 -m aotriton.tune.*`.
      3. `<repo root inferred from this file>/.tune` -- only correct under an
         editable install from this exact checkout.
    '''
    env = os.environ.get('AOTRITON_TUNE_ROOT')
    if env:
        return Path(env)
    cwd_candidate = Path.cwd() / '.tune'
    if cwd_candidate.is_dir():
        return cwd_candidate
    return Path(__file__).resolve().parent.parent.parent / '.tune'


def load_family_tune(family: str, modules_dir: 'Path | None' = None):
    """Import `<modules_dir>/<family>/tune/__init__.py` by path under a
    synthetic unique package name, so `modules/<family>` stays a plain
    directory (not a package) -- identical rationale to
    `python/codegen/parser.py`'s `load_family_aot` (see also
    `modules/flash/tune/__init__.py`). Cached in `sys.modules`, under a key
    that includes the module TREE -- see `_tree_cache_key`.
    """
    if modules_dir is None:
        modules_dir = default_modules_dir()
    return load_family_block(family, 'tune', modules_dir)


def load_family_visperf(family: str, modules_dir: 'Path | None' = None):
    """Import `<modules_dir>/<family>/visperf/__init__.py` by path under a
    synthetic unique package name, mirroring `load_family_tune` above
    (identical rationale, F6: `modules/<family>` stays a plain directory,
    not a package). Cached in `sys.modules`, under the same tree-keyed name
    `load_family_tune` uses -- see `_tree_cache_key`.

    The returned module exports `DESCRIPTOR` (a dict consumed by
    `aotriton.tune.pq.visperf`'s query builder and the webui's perf page)
    and, by convention, ships its JS counterpart at
    `<modules_dir>/<family>/visperf/static/<family>.js` -- served through
    the webui's `/family_static/<family>/<path:filename>` route and
    inlined by `aotriton.tune.pq.export_visperf` for the standalone export
    (modular-tune.md §3d).
    """
    if modules_dir is None:
        modules_dir = default_modules_dir()
    return load_family_block(family, 'visperf', modules_dir)


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
