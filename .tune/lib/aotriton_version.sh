#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Print AOTriton's MAJOR and MINOR version ints (space-separated), read once
# from CMakeLists.txt's set(AOTRITON_VERSION_*_INT ...) lines.
get_aotriton_major_minor() {
  local aotriton_root="$1"
  local major minor
  major=$(grep '^set(AOTRITON_VERSION_MAJOR_INT' "$aotriton_root/CMakeLists.txt" | cut -d ' ' -f 2 | cut -d ')' -f 1)
  minor=$(grep '^set(AOTRITON_VERSION_MINOR_INT' "$aotriton_root/CMakeLists.txt" | cut -d ' ' -f 2 | cut -d ')' -f 1)
  echo "$major $minor"
}

# AOTriton VERSION_DIR string, matching .ci/Makefile's convention
# (e.g. "0.13.1b"). Locates the optional .ci/<VERSION_DIR>.yaml altwheel config.
get_aotriton_version_dir() {
  local aotriton_root="$1"
  local major minor patch version_dir
  read -r major minor <<< "$(get_aotriton_major_minor "$aotriton_root")"
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

# Print the resolved altwheel YAML path for this workdir, if
# prebuild_wheel.sh already produced one; nothing otherwise.
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
