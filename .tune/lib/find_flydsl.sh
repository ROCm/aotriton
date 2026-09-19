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
#
# A malformed pin file ABORTS the caller rather than returning an answer. There
# are three outcomes here, not two, and the third has no sensible recovery: a
# second non-comment line means nobody knows which LLVM is wanted, and every
# caller would have to stop anyway. Reading it as "not required" would turn a
# typo into a skipped FlyDSL build and then, minutes later, a cmake FATAL_ERROR
# naming the tripwire instead of the typo. pin_line has already said what is
# wrong with the file.
flydsl_required() {
  local aotriton_root="$1"
  local pin
  pin="$(pin_line "$aotriton_root/third_party/flydsl-llvm.txt")" || exit 1
  [ -n "$pin" ]
}

# The FlyDSL commit to build, read from the git URL pin in
# third_party/flydsl-compiler.txt by .ci/common-pin.sh's parser. One reader for
# that file, shared with .ci/releasesuite-git-head.sh, so the two cannot
# disagree about which FlyDSL a build means.
flydsl_ref_from_pin() {
  local aotriton_root="$1"
  local pin_text
  pin_text="$(parse_flydsl_pin "$aotriton_root/third_party/flydsl-compiler.txt")" || return 1
  # Second line is the commit; the origin on the first is not ours to pick,
  # .ci/build_flydsl_wheel.sh reads the same pin for it.
  printf '%s' "$(printf '%s' "$pin_text" | tail -n 1)"
}

# The LLVM commit this tree is pinned to, as the 12 hex digits every artifact
# name is keyed on.
flydsl_llvm_sha12() {
  local aotriton_root="$1"
  local pin_text sha
  pin_text="$(parse_llvm_pin "$aotriton_root/third_party/flydsl-llvm.txt")" || return 1
  sha="$(printf '%s' "$pin_text" | tail -n 1)"
  [ -n "$sha" ] || return 1
  printf '%s' "${sha:0:12}"
}

# Where the LLVM stage records the tarball it produced. A stamp, because
# .ci/build_llvm_tarball.sh prints the exact path (diagnostics go to stderr) and
# the distro component of the name is its business, not ours.
flydsl_llvm_stamp() { printf '%s/.current-tarball' "$1"; }

# Print the LLVM tarball the CURRENT pin names, or return 1.
#
# The stamp is checked against the pinned SHA rather than trusted: it survives
# in the workdir across pin moves, and `--select flydsl-wheel` on its own would
# otherwise hand an artifact from the previous pin to a wheel build that claims
# to be this one. The llvm-<sha12> prefix is the same identity
# .ci/build_flydsl_wheel.sh already regexes out of the name for its cache key.
find_flydsl_llvm_tarball() {
  local aotriton_root="$1" tarball_dir="$2"
  local stamp tarball sha12
  sha12="$(flydsl_llvm_sha12 "$aotriton_root")" || return 1
  stamp="$(flydsl_llvm_stamp "$tarball_dir")"
  [ -f "$stamp" ] || return 1
  tarball="$(cat "$stamp")"
  [ -n "$tarball" ] && [ -f "$tarball" ] || return 1
  [[ "$(basename "$tarball")" == llvm-"$sha12"-* ]] || return 1
  printf '%s' "$tarball"
}

# The wheel name .ci/build_flydsl_wheel.sh would produce for the CURRENT pins,
# as a glob. Every component of its cache key, in its order:
#
#   flydsl-*+git<flydsl sha8><version suffix>.llvm<llvm sha12>.p<patches>-*<abi>*.whl
#
# All five matter, and a wheel missing any one of them is a different wheel:
# the FlyDSL commit and the LLVM commit are what was compiled and what it was
# compiled against (the wrong LLVM miscompiles register spills and returns
# wrong numbers rather than failing); .ci/flydsl-patch/ is part of what the
# wheel IS, so its count is in the name; and CMakeLists.txt makes a cp-tag
# mismatch a FATAL_ERROR.
#
# Kept in step with that script's cached_wheel() by hand. The alternative --
# trusting a path the prebuild recorded -- cannot notice a pin moving under it,
# which is precisely when a stale wheel is installed and nothing says so.
flydsl_wheel_glob() {
  local aotriton_root="$1" pyver="$2"
  local flydsl_sha llvm_sha12 patches major minor
  flydsl_sha="$(flydsl_ref_from_pin "$aotriton_root")" || return 1
  llvm_sha12="$(flydsl_llvm_sha12 "$aotriton_root")" || return 1
  # Both pins must name a commit, because the cache is keyed on the RESOLVED
  # SHA and only a build can resolve a moving ref. Refusing here names the pin
  # to fix; globbing on the first 8 characters of a branch name would instead
  # match nothing and read as "run the prebuild", which would not help.
  if [[ ! "$flydsl_sha" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "Error: third_party/flydsl-compiler.txt pins '$flydsl_sha', not a commit SHA." >&2
    return 1
  fi
  if [[ ! "$llvm_sha12" =~ ^[0-9a-fA-F]{12}$ ]]; then
    echo "Error: third_party/flydsl-llvm.txt does not pin a commit SHA." >&2
    return 1
  fi
  patches=$(ls "$aotriton_root"/.ci/flydsl-patch/*.patch 2>/dev/null | wc -l)
  read -r major minor <<< "$(get_aotriton_major_minor "$aotriton_root")"
  printf 'flydsl-*+git%s.aotriton%s.%s.llvm%s.p%s-*cp%s*.whl' \
    "${flydsl_sha:0:8}" "$major" "$minor" "$llvm_sha12" "$patches" "${pyver//./}"
}

# Print the cached FlyDSL wheel matching the current pins and Python ABI, or
# return 1. A miss is how "run the prebuild first" gets said.
find_flydsl_wheel() {
  local aotriton_root="$1"
  local workdir="$2"
  local pyver="$3"       # X.Y, as `python --version` reports it
  local glob wheel
  glob="$(flydsl_wheel_glob "$aotriton_root" "$pyver")" || return 1
  # Unquoted on purpose: $glob is a pattern. `|| true` because callers run under
  # `set -euo pipefail`, where ls exiting 2 on a missing cache dir would kill
  # them at the assignment instead of letting this return 1.
  wheel=$(ls "$workdir/scratch/flydsl"/$glob 2>/dev/null | head -n 1) || true
  [ -n "$wheel" ] && [ -f "$wheel" ] || return 1
  printf '%s' "$wheel"
}
