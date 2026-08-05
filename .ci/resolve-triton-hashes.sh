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

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
# shellcheck disable=SC1091
. "${SCRIPT_DIR}/common-altwheel.sh"

# .venvs.default overrides the submodule-pinned commit if set.
DEFAULT_HASH=""
if [[ -n "${ALTWHEEL_YAML}" ]]; then
  DEFAULT_HASH=$(altwheel_venv_hash "${ALTWHEEL_YAML}" ".venvs.default")
fi
if [[ -z "${DEFAULT_HASH}" ]]; then
  DEFAULT_HASH=$(git -C "${AOTRITON_SRC_DIR}" rev-parse HEAD:third_party/triton)
fi

echo "${DEFAULT_HASH}"

if [[ -n "${ALTWHEEL_YAML}" ]]; then
  for key in $(yq -r '.venvs | keys | .[]' "${ALTWHEEL_YAML}"); do
    [ "$key" = "default" ] && continue
    altwheel_venv_hash "${ALTWHEEL_YAML}" ".venvs.${key}"
  done
fi
