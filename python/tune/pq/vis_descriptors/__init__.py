# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Registry of per-family visperf descriptors (modular-tune.md §3d.1).

This package used to hold the flash descriptor directly
(`vis_descriptors/flash.py`); it now holds only the registry shim. Each
family's descriptor lives at `modules/<family>/visperf/__init__.py` and is
loaded by path via `aotriton.tune.registry.load_family_visperf` -- the same
mechanism `load_family_tune` uses for `modules/<family>/tune/` (F6), so
`modules/<family>` stays a plain directory rather than a package.

Adding a family means registering it in `registry.py`'s family list; no
import here changes.

Keyed by FAMILY NAME (the `modules/<family>/` directory), not by whatever
`DESCRIPTOR['id']` happens to say: consumers turn these keys back into paths
-- the webui's `/family_static/<family>/<family>.js` route and
`perf.html`'s script loop both do -- so a descriptor whose `id` drifted from
its directory name would produce a 404 and a page with no descriptor
registered. The two are asserted equal instead of silently reconciled.
"""

from ...registry import available_module_names, load_family_visperf

def _load_descriptors() -> dict[str, dict]:
    """Build the registry in a function scope, so no loop variables leak into
    the module namespace and an empty family list is simply an empty dict."""
    out: dict[str, dict] = {}
    for family in available_module_names():
        try:
            descriptor = load_family_visperf(family).DESCRIPTOR
        except ImportError:
            # visperf is optional: a family may be tunable without shipping a
            # visualisation. Skipping keeps one such family from taking the
            # whole web UI down at import.
            continue
        if descriptor['id'] != family:
            raise ValueError(
                f"visperf DESCRIPTOR['id'] is {descriptor['id']!r} but the "
                f"family directory is {family!r}; they must match -- the key "
                f"is used as a path segment by /family_static and by "
                f"export_visperf.")
        out[family] = descriptor
    return out


# Registry: family name -> descriptor dict
DESCRIPTORS: dict[str, dict] = _load_descriptors()
