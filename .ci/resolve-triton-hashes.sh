#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Resolve the Triton commit hashes to build: the default (submodule-pinned,
# or .venvs.default) plus every other .venvs.* entry in an altwheel YAML.
# Shared by releasesuite-git-head.sh and .tune.
#
# Usage: resolve-triton-hashes.sh <aotriton_src_dir> [<altwheel_yaml>]
# Prints one hash per line, default first.

set -e

AOTRITON_SRC_DIR="$1"
ALTWHEEL_YAML="${2:-}"

if [ -z "$AOTRITON_SRC_DIR" ]; then
  echo "Usage: $0 <aotriton_src_dir> [<altwheel_yaml>]" >&2
  exit 1
fi

# .venvs.default overrides the submodule-pinned commit if set.
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
