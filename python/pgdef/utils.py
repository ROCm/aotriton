# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Engine utilities that every application needs: logging setup and the by-path
loader for family plugins. (The defensive subprocess line reader is not
pgdef-specific; it lives in aotriton.utils.line_reader.)

Nothing here may import an application, torch or triton (see
python/test/pgdef/test_layering.py).
"""

import hashlib
import importlib.util
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def configure_logging_with_flush():
    """
    Configure logging with line-buffered output to ensure logs are written immediately.
    Critical for debugging blocking issues where buffered logs might never appear.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True
    )

    # Force line-buffered output for all handlers
    for handler in logging.root.handlers:
        handler.setStream(sys.stderr)
        if hasattr(handler.stream, 'reconfigure'):
            handler.stream.reconfigure(line_buffering=True)


def tree_cache_key(modules_dir, family: str, block: str) -> str:
    """Synthetic module name for `<modules_dir>/<family>/<block>`.

    Keyed on a digest of the RESOLVED `modules_dir`, not on the family name
    alone -- exactly as `python/codegen/parser.py`'s `load_family_aot` is, and
    for the same reason. Two trees can supply the same family (the real
    `modules/` and `python/test/fakefamily/` both have a `flash`), and a
    name-only key means whichever tree loaded first serves both. That is
    invisible while the two agree and silently wrong the moment they diverge.
    """
    digest = hashlib.blake2b(str(Path(modules_dir).resolve()).encode(),
                             digest_size=6).hexdigest()
    return f'_aotriton_modules_{digest}_{family}_{block}'


def load_family_block(family: str, block: str, modules_dir):
    """Import `<modules_dir>/<family>/<block>/__init__.py` by path.

    The package gets a synthetic unique name (see `tree_cache_key`), so
    `modules/<family>` stays a plain directory rather than a package -- the
    same rationale as `python/codegen/parser.py`'s `load_family_aot`.
    `block` is the application's own subdirectory name (`tune`, `visperf`, ...); the engine does not know any of them. Cached in
    `sys.modules` under a tree-keyed name.

    Raises ImportError if the block does not exist.
    """
    modname = tree_cache_key(modules_dir, family, block)
    cached = sys.modules.get(modname)
    if cached is not None:
        return cached
    block_dir = Path(modules_dir) / family / block
    init_path = block_dir / '__init__.py'
    if not init_path.is_file():
        raise ImportError(
            f"No {block} block for family '{family}': {init_path} not found "
            f"(modules_dir={modules_dir}). Set AOTRITON_MODULES_DIR or run "
            f"from the repo root.")
    spec = importlib.util.spec_from_file_location(
        modname, init_path, submodule_search_locations=[str(block_dir)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


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
         into `$AOTRITON_ROOT` before invoking `python3 -m aotriton.<app>.*`
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
