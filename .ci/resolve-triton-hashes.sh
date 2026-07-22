#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Resolve the set of Triton commit hashes that need building: the default
# (submodule-pinned commit, or a .venvs.default override) plus every other
# .venvs.* entry in an optional altwheel YAML. Extracted from
# releasesuite-git-head.sh so both .ci's release suite and .tune share one
# implementation instead of two independent copies.
#
# Usage: resolve-triton-hashes.sh <aotriton_src_dir> [<altwheel_yaml>]
# Prints the resolved hashes to stdout, one per line, default hash first.

set -e

AOTRITON_SRC_DIR="$1"
ALTWHEEL_YAML="${2:-}"

if [ -z "$AOTRITON_SRC_DIR" ]; then
  echo "Usage: $0 <aotriton_src_dir> [<altwheel_yaml>]" >&2
  exit 1
fi

# .venvs.default in the YAML replaces the embedded submodule hash;
# otherwise the submodule-pinned commit is the mandatory default.
DEFAULT_HASH=""
if [[ -n "${ALTWHEEL_YAML}" ]]; then
  DEFAULT_HASH=$(yq -r '.venvs.default // ""' "${ALTWHEEL_YAML}")
fi
if [[ -z "${DEFAULT_HASH}" ]]; then
  DEFAULT_HASH=$(git -C "${AOTRITON_SRC_DIR}" rev-parse HEAD:third_party/triton)
fi

echo "${DEFAULT_HASH}"

if [[ -n "${ALTWHEEL_YAML}" ]]; then
  yq -r '.venvs | to_entries | .[] | select(.key != "default") | .value' "${ALTWHEEL_YAML}"
fi
