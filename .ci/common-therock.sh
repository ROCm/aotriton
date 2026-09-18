#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Which pip index carries a given TheRock ROCm version. Source this; do not
# execute it.
#
# A PEP 440 pre-release suffix -- 'a' followed by an 8-digit date, as in
# 10.2.0a20260918 -- marks a nightly, and nightlies exist ONLY on the nightly
# index. Everything else is a release (7.14.1) and exists ONLY on the release
# index. Getting it wrong does not fall back: pip resolves no wheel at all.
#
# Deliberately not keyed on the major version. ROCm 7.15 was renamed to 10 and
# its nightlies were retired, so there is no longer a 7.x nightly channel to
# distinguish from a 10.x one -- the suffix alone answers the question, and will
# keep answering it when 11 arrives.
#
# It lives here rather than in common-vars.sh because that file runs
# rocm_agent_enumerator at source time and so needs a GPU, which a host-side
# wheel build (build_flydsl_wheel.sh) does not have.
THEROCK_NIGHTLY_INDEX_URL="https://nightly.repo.amd.com/rocm/whl-next/"
THEROCK_RELEASE_INDEX_URL="https://repo.amd.com/rocm/whl-multi-arch/"

therock_pip_index_url() {
  local version="$1"
  if [[ "${version}" =~ a[0-9]{8} ]]; then
    printf '%s' "${THEROCK_NIGHTLY_INDEX_URL}"
  else
    printf '%s' "${THEROCK_RELEASE_INDEX_URL}"
  fi
}
