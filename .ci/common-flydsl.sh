#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Shared by .ci/build_flydsl_wheel.sh and .tune/lib/find_flydsl.sh, the two
# readers of the FlyDSL wheel cache. Source this; do not execute it.

# Print the first <wheel> whose .p<N> equals the number of patches in
# <patch_root>/<version>/, where <version> is the FlyDSL base version in that
# wheel's own name; print nothing if none does.
#
# The count cannot be part of a plain glob: which patch set applies depends on
# the checkout's __version__, which runc-build-flydsl-wheel.sh reads inside the
# container and a caller holding only a SHA does not know. Every wheel records
# its version, though, so a candidate can be checked against the patch set it
# was built with. Adding a patch for one version then misses only that
# version's wheels.
flydsl_pick_cached_wheel() {
  local patch_root="$1" wheel version count patches
  shift
  for wheel in "$@"; do
    [ -f "$wheel" ] || continue
    [[ "$(basename "$wheel")" =~ ^flydsl-([^+]+)\+.*\.p([0-9]+)-cp ]] || continue
    version="${BASH_REMATCH[1]}"
    count="${BASH_REMATCH[2]}"
    # A glob, not `ls | wc -l`: no directory for <version> is the common case
    # (0 patches), and ls failing on it kills a caller under pipefail.
    patches=("$patch_root/$version"/*.patch)
    [ -e "${patches[0]}" ] || patches=()
    if [[ "$count" -eq "${#patches[@]}" ]]; then
      printf '%s\n' "$wheel"
      return 0
    fi
  done
}
