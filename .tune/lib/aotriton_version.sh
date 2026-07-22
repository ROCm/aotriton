#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Compute the AOTriton VERSION_DIR string: matches .ci/Makefile's convention
# (major.minor[.patch], with a trailing "b" while major < 1, e.g. "0.13.1b"
# for 0.13.1). Used to locate the optional per-version altwheel YAML at
# .ci/<VERSION_DIR>.yaml -- shared here instead of recomputed in each of
# .tune/single/prebuild_wheel.sh, .tune/single/build_arch.sh, .tune/bin/testbld.
get_aotriton_version_dir() {
  local aotriton_root="$1"
  local major minor patch version_dir
  major=$(grep '^set(AOTRITON_VERSION_MAJOR_INT' "$aotriton_root/CMakeLists.txt" | cut -d ' ' -f 2 | cut -d ')' -f 1)
  minor=$(grep '^set(AOTRITON_VERSION_MINOR_INT' "$aotriton_root/CMakeLists.txt" | cut -d ' ' -f 2 | cut -d ')' -f 1)
  patch=$(grep '^set(AOTRITON_VERSION_PATCH_INT' "$aotriton_root/CMakeLists.txt" | cut -d ' ' -f 2 | cut -d ')' -f 1)
  version_dir="${major}.${minor}"
  if [ "$patch" != "0" ]; then
    version_dir="${version_dir}.${patch}"
  fi
  if [ "$major" -lt 1 ]; then
    version_dir="${version_dir}b"
  fi
  echo "$version_dir"
}

# Print the path to a resolved altwheel YAML for this workdir, if one
# applies -- i.e. .ci/<VERSION_DIR>.yaml exists AND a resolved copy (hashes
# replaced with actual wheel paths) has already been produced by
# .tune/single/prebuild_wheel.sh. Prints nothing otherwise: either this
# version has no altwheel config, or prebuild_wheel.sh never ran (e.g.
# bare-metal libbld usage that bypasses remotebld) -- callers should treat
# that as "no altwheel config" and build with just the default wheel, same
# as today.
get_resolved_altwheel_yaml() {
  local aotriton_root="$1"
  local workdir="$2"
  local version_dir resolved
  version_dir="$(get_aotriton_version_dir "$aotriton_root")"
  resolved="$workdir/scratch/triton/resolved_altwheel.yaml"
  if [ -f "$aotriton_root/.ci/${version_dir}.yaml" ] && [ -f "$resolved" ]; then
    echo "$resolved"
  fi
}
