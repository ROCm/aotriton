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

The code generator emits one image per build and a selector over the boxes.
Each image's `Fly.compile` row names its build to `aotriton.flyc_compile` by
its *build knobs*: the knobs whose values differ among the functional's
builds, which both sides compute from the same description.
"""

from aotriton.utils import parse_pon, render_pon


def flyc_builds(result):
    """`[(boxes, build, knobs)]`; a single build gets `boxes == [{}]` (every call)."""
    if isinstance(result, tuple):
        build, knobs = result
        return [([{}], build, knobs)]
    return list(result)


def build_knobs(all_knobs):
    """For each build's knobs, the ones whose values differ among the builds.

    `{}` for a functional with one build. Distinct builds have distinct knobs,
    so the result is distinct per build.
    """
    differ = sorted({k for knobs in all_knobs for k in knobs
                     if len({repr(other.get(k)) for other in all_knobs}) > 1})
    return [{k: knobs.get(k) for k in differ} for knobs in all_knobs]


def render_build_knobs(knobs) -> str:
    """`--build_knobs` text: space-separated `k=v`, like `--signature`."""
    return render_pon(knobs, sep=' ')


def flyc_build_for(result, build_knobs_text):
    """`(build, knobs)` of the build `build_knobs_text` names; `''` names the only build."""
    found = flyc_builds(result)
    wanted = parse_pon(build_knobs_text, sep=' ') if build_knobs_text else {}
    named = build_knobs([knobs for _, _, knobs in found])
    matches = [(build, knobs) for (_, build, knobs), bk in zip(found, named) if bk == wanted]
    if len(matches) != 1:
        raise ValueError(f'--build_knobs {build_knobs_text!r} names {len(matches)} of this functional\'s '
                         f'{len(found)} builds; it has {[render_build_knobs(bk) for bk in named]}')
    return matches[0]
