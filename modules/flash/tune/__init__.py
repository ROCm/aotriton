# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Flash family tuning block.

Loaded BY PATH under the synthetic name `_aotriton_modules_flash_tune` (see
python/tune/registry.py's `load_family_tune`), mirroring how
python/codegen/parser.py's `load_family_aot` loads modules/<family>/aot/. This
keeps `modules/<family>` a plain directory rather than a package, so a stray
`flash.py` on sys.path cannot shadow it.

Phase 1 (pure relocation, no unification): this package holds two independent
subpackages moved verbatim from v3python.tune -- `flash` (kernel-level tuning)
and `flash_op` (operator-level tuning) -- each still exporting its own
`TuneDesc`/`ImplSelector`. Nothing is re-exported here yet; the registry
resolves 'flash' -> this package's `flash` submodule and 'flash_op' -> its
`flash_op` submodule.

Phase 2 unifies the two into a single `FlashTune` description (one `TuneDesc`,
`LEVELS = {'kernel': ..., 'op': ...}`) — see modular-tune.md §4.1/§4.3. This
file's shape will change then.
"""
