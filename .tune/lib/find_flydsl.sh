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

# The FlyDSL commit to build, read from the PEP 508 direct reference in
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

# Where each prebuild stage records the artifact it produced.
#
# A STAMP, not a glob. Both .ci build scripts print the exact path of the
# artifact -- cache hit or fresh build, diagnostics go to stderr -- so the
# identity is already known at the only moment it is unambiguous. Recomputing
# it later means re-deriving a cache key that lives in
# .ci/build_flydsl_wheel.sh (FlyDSL sha8, the --version_suffix, the LLVM sha12,
# the patch count, the ABI tag), and a second copy of that key is a second
# thing to get wrong.
#
# The wheel stamp is per Python ABI because the wheels are: CMakeLists.txt
# makes a cp-tag mismatch a FATAL_ERROR, so one stamp per workdir would let two
# images overwrite each other's answer and move the failure an hour downstream.
flydsl_llvm_stamp()  { printf '%s/.current-tarball' "$1"; }
flydsl_wheel_stamp() { printf '%s/.current-wheel-cp%s' "$1" "${2/./}"; }

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

# Print the cached FlyDSL wheel for this workdir and Python ABI, or return 1.
find_flydsl_wheel() {
  local workdir="$1"
  local pyver="$2"       # X.Y, as `python --version` reports it
  local stamp wheel
  stamp="$(flydsl_wheel_stamp "$workdir/scratch/flydsl" "$pyver")"
  [ -f "$stamp" ] || return 1
  wheel="$(cat "$stamp")"
  [ -n "$wheel" ] && [ -f "$wheel" ] || return 1
  printf '%s' "$wheel"
}
