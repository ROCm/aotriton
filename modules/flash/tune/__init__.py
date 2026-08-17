# Copyright © 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Flash family tuning block.

Loaded BY PATH under the synthetic name `_aotriton_modules_flash_tune` (see
python/tune/registry.py's `load_family_tune`), mirroring how
python/codegen/parser.py's `load_family_aot` loads modules/<family>/aot/. This
keeps `modules/<family>` a plain directory rather than a package, so a stray
`flash.py` on sys.path cannot shadow it.

Phase 2 (modularization unification, modular-tune.md §4.1-§4.3): the former
Phase-1 `flash` (kernel-level) and `flash_op` (op-level) subpackages are
unified into a single `FlashTune` description. `ImplSelector` is the single
DSL replacing the old per-module FlashKernelSelector/FlashOpBackendSelector
pair (see aotriton.tune.tdesc).

Revision note 3 (modular-tune.md): `FlashTune` lists and resolves impls of
BOTH tuning levels directly, keyed by DSL name (`list_impls`/`get_impl`/
`probe_all_impls`/`probe_impl_desc`) -- there is no per-level `TuningLevel`
strategy object and no `level=` constructor argument. `TuneDesc` is the single
handle for tuning metadata; this package intentionally does not export a
second one (e.g. a standalone `LEVELS` mapping) -- extend
`TuningDescription`'s interface instead if something outside needs metadata
`TuneDesc` cannot supply.

Everything below except `sancheck` (needed eagerly by the codegen back-edge in
python/template_instantiation/ir/kdesc.py, and torch-free) stays lazily
resolved: importing this package must not pull in torch/pyaotriton (see
desc.py's module docstring for why).
"""

from .desc import FlashTune
from aotriton.tune.tdesc import ImplSelector
from . import sancheck

TuneDesc = FlashTune

__all__ = ['TuneDesc', 'ImplSelector', 'sancheck']
