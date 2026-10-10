#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Which pip index carries a given TheRock ROCm version. Source this; do not
# execute it.
#
# Two things pick the index, and both matter:
#
#   * A PEP 440 pre-release suffix -- 'a' followed by an 8-digit date, as in
#     10.2.0a20260918 -- marks a nightly. Nightlies and releases live on
#     separate indexes and pip does not fall back between them: get it wrong
#     and it resolves no wheel at all.
#   * The major version picks the HOST. ROCm 10 serves whl-next from
#     stable./nightly.repo.amd.com; 7.14 serves whl-multi-arch from
#     repo.amd.com. A 10.x version on the 7.x host resolves nothing.
#
# 7.x nightlies were retired when 7.15 was renamed to 10, so the 7.x row has no
# nightly entry; a 7.x version carrying a nightly suffix falls through to the
# ROCm 10 nightly index and finds nothing, which is what it would do anyway.
#
# It lives here rather than in common-vars.sh because that file runs
# rocm_agent_enumerator at source time and so needs a GPU, which a host-side
# wheel build (build_flydsl_wheel.sh) does not have.
THEROCK_NIGHTLY_INDEX_URL="https://nightly.repo.amd.com/rocm/whl-next/"
THEROCK_STABLE_INDEX_URL="https://stable.repo.amd.com/rocm/whl-next/"
THEROCK_LEGACY_STABLE_INDEX_URL="https://repo.amd.com/rocm/whl-multi-arch/"

therock_pip_index_url() {
  local version="$1"
  local major="${version%%.*}"
  if [[ "${version}" =~ a[0-9]{8} ]]; then
    printf '%s' "${THEROCK_NIGHTLY_INDEX_URL}"
  elif [[ "${major}" -ge 10 ]]; then
    printf '%s' "${THEROCK_STABLE_INDEX_URL}"
  else
    printf '%s' "${THEROCK_LEGACY_STABLE_INDEX_URL}"
  fi
}
