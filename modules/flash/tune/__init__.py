# Copyright © 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Flash family tuning block.

Loaded BY PATH under the synthetic name `_aotriton_modules_flash_tune` (see
python/tune/registry.py's `load_family_tune`), mirroring how
python/codegen/parser.py's `load_family_aot` loads modules/<family>/aot/. This
keeps `modules/<family>` a plain directory rather than a package, so a stray
`flash.py` on sys.path cannot shadow it.

One `FlashTune` description covers both tuning levels. It lists and resolves
every impl directly, keyed by its DSL name (`list_impls` / `get_impl` /
`probe_all_impls` / `probe_impl_desc`), dispatching on the `op.` prefix --
there is no per-level strategy object and no `level=` constructor argument.
`ImplSelector` (see aotriton.tune.tdesc) is the sole parser of that DSL.

`TuneDesc` is the single handle for tuning metadata. This package
deliberately exports nothing alongside it for that purpose: if something
outside needs metadata `TuneDesc` cannot supply, extend
`TuningDescription`'s interface rather than adding a second handle.

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
