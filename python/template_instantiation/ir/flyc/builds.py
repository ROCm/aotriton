# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
What an `@ati.flyc.kernel` builder function returns, normalized.

The function returns either

  (build, knobs)                  one build serves every call of the functional;
  a generator of (boxes, build, knobs)
                                  one build per distinct set of per-call knob
                                  overrides, `boxes` being the calls it serves:
                                  a list of `{runtime value: (lo, hi)}`, inclusive
                                  (see `ir/runtime_rules.py`).

`build` is deferred (it imports flydsl); `knobs` is the JSON-serialisable dict
recorded beside the hsaco.
"""


def flyc_builds(result):
    """`[(boxes, build, knobs)]`; a single build gets `boxes == [{}]` (every call)."""
    if isinstance(result, tuple):
        build, knobs = result
        return [([{}], build, knobs)]
    return list(result)


def flyc_shipped_build(result):
    """`(build, knobs)` of the one image the code generator ships per functional.

    TODO(flyc conditional builds): the code generator does not emit the
    per-call selector yet, so a functional with several builds ships only the
    first, which `runtime_rules.builds` orders as the build for the smallest
    runtime values -- for the gfx950 forward, today's build (no D floor, the
    exact KV walk). Remove once flytune.py emits every build and the selector.
    """
    boxes, build, knobs = flyc_builds(result)[0]
    return build, knobs
