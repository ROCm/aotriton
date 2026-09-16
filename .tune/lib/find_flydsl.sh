#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# THIS FILE IS TEMPORARY AND SHOULD GO AWAY.
#
# Deciding how AOTriton gets a FlyDSL compiler is not .tune's business. It
# belongs under .ci/, next to the scripts that already build the artifacts --
# and the shape it should take there is still open: `pip install -r
# third_party/flydsl-compiler.txt` may turn out to be sufficient on its own, or
# a from-source wheel build may stay necessary. That question is deferred, not
# answered here.
#
# What is NOT deferred is that .tune needs an answer today: with
# third_party/flydsl-llvm.txt non-empty, CMakeLists.txt's tripwire fails every
# image-mode configure that does not pass AOTRITON_USE_LOCAL_FLYDSL_WHEEL, so
# libbld and testbld do not build at all. This file is the smallest thing that
# unblocks them. When .ci/ takes ownership, delete it and repoint the three
# callers (.tune/single/prebuild_wheel.sh, .tune/bin/testbld,
# .tune/single/build_arch.sh).
#
# Source this; do not execute it.

# The SINGLE decision point for "does this tree need a locally built FlyDSL
# wheel". Every caller asks this rather than reading a pin file itself, so the
# policy has exactly one body to replace when it grows.
#
# Today the answer is "iff third_party/flydsl-llvm.txt is non-empty", which is
# the same condition CMakeLists.txt:206 fails on and the same one
# .ci/releasesuite-git-head.sh:320 uses to imply --flydsl_commit. A non-empty
# pin means the released wheel was built against an LLVM known to miscompile
# register spills, so the pinned wheel must not be used and a local build is
# the only thing that can finish.
flydsl_required() {
  local aotriton_root="$1"
  local pin
  pin="$(pin_line "$aotriton_root/third_party/flydsl-llvm.txt")" || return 2
  [ -n "$pin" ]
}

# The FlyDSL ref to build, spelled as a git tag. FlyDSL releases are tagged
# vX.Y.Z, so `flydsl==0.3.1` is `v0.3.1`. Any other requirement shape is not
# something to guess at -- same rule, and the same refusal to guess, as
# .ci/releasesuite-git-head.sh:325-333.
flydsl_ref_from_pin() {
  local aotriton_root="$1"
  local req
  req="$(pin_line "$aotriton_root/third_party/flydsl-compiler.txt")" || return 1
  if [[ "$req" =~ ^flydsl[[:space:]]*==[[:space:]]*([0-9]+\.[0-9]+\.[0-9]+)$ ]]; then
    printf 'v%s' "${BASH_REMATCH[1]}"
    return 0
  fi
  echo "Error: cannot derive a FlyDSL ref from third_party/flydsl-compiler.txt ('$req')." >&2
  echo "  Expected 'flydsl==X.Y.Z'. Build the wheel by hand with" >&2
  echo "  .ci/build_flydsl_wheel.sh --flydsl_commit <ref> if the pin has moved to" >&2
  echo "  a requirement shape this cannot read." >&2
  return 1
}

# Print the cached FlyDSL wheel for this workdir and Python ABI, or return 1.
#
# The ABI tag is part of the match, not a detail: flydsl wheels are CPython-ABI
# specific and CMakeLists.txt:451-452 makes a cp-tag mismatch a FATAL_ERROR, so
# a wheel built for another Python is not a hit -- finding it here would only
# move the failure an hour downstream. Same reasoning as
# .ci/build_flydsl_wheel.sh's own cache key.
find_flydsl_wheel() {
  local workdir="$1"
  local pyver="$2"       # X.Y, as `python --version` reports it
  local wheel
  # `|| true`: callers run under `set -euo pipefail` (testbld, build_arch.sh),
  # where a missing cache dir makes ls exit 2, pipefail carries it through
  # `| head`, and the assignment kills the CALLER instead of returning 1 here.
  # A miss has to be reportable -- it is how "run the prebuild first" gets said.
  wheel=$(ls "$workdir/scratch/flydsl"/flydsl-*"cp${pyver/./}"*.whl 2>/dev/null | head -n 1) || true
  [ -n "$wheel" ] && [ -f "$wheel" ] || return 1
  echo "$wheel"
}
