# SPDX-License-Identifier: Apache-2.0

"""Which arch the JIT suites build for, and when they skip.

The arch is the first GPU's, and a suite runs wherever its dispatcher has a
row for it. Knob overrides are arch-specific, so a test variant that sets
some names the arch it belongs to and skips elsewhere.
"""

import os

import pytest


def device_arch():
    """The first GPU's arch without feature suffixes ("gfx950"), or None."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]


def require(dispatcher):
    """This GPU's arch; skips the calling module unless `dispatcher` serves it.

    Call it before importing anything that imports flydsl: a skip mark does not
    stop module-level imports, so a host without FlyDSL would fail collection.
    """
    if not os.environ.get("AOTRITON_FLYDSL_KERNEL_ROOT"):
        pytest.skip("needs AOTRITON_FLYDSL_KERNEL_ROOT", allow_module_level=True)
    arch = device_arch()
    if arch is None:
        pytest.skip("needs a GPU", allow_module_level=True)
    if arch not in dispatcher.arches():
        pytest.skip(f"{dispatcher.__name__} has no {arch} row", allow_module_level=True)
    return arch


def overrides(arch, variant):
    """`variant`'s knob overrides on `arch`, or a skip.

    `variant` is None (the policy's choice, every arch) or `(owner, overrides)`.
    """
    if variant is None:
        return None
    owner, knobs = variant
    if owner != arch:
        pytest.skip(f"knob variant {knobs} is {owner}'s")
    return knobs


def build(dispatcher, arch, meta, variant=None):
    """`(fn, knobs)`: `meta` planned for `arch` and built, overrides checked."""
    pins = overrides(arch, variant)
    meta, knobs = dispatcher.plan(arch, meta, overrides=pins)
    for name, value in (pins or {}).items():
        assert getattr(knobs, name) == value, f"{name}={value} override ignored: {getattr(knobs, name)}"
    return dispatcher.builder(arch, meta, knobs), knobs
