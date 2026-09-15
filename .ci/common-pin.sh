#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Readers for the one-line pin files in third_party/.
#
# "Empty" means no non-comment, non-blank line, which is how
# CMakeLists.txt reads third_party/flydsl-llvm.txt: the comment saying what a
# file is for survives the file being emptied.

# The single non-comment line of a pin file, or nothing when it has none.
# More than one is an error: these files pin ONE thing, and silently taking
# the first would make an accidental second line invisible.
pin_line() {
  local line count=0 pin=""
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"   # ltrim
    line="${line%"${line##*[![:space:]]}"}"   # rtrim
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    pin="${line}"
    count=$((count + 1))
  done < "$1"
  if [[ "${count}" -gt 1 ]]; then
    echo "Error: ${1} has ${count} non-comment lines; it must have at most one." >&2
    return 1
  fi
  printf '%s' "${pin}"
}

# One PEP 508 direct reference: git+<scheme>://<url>@<ref>. Split on the LAST
# '@', which is unambiguous for that form and stays unambiguous for
# git+ssh://git@host/org/repo@ref. The scp-style git@github.com:org/repo is the
# one spelling the rule cannot disambiguate, so it is rejected rather than
# guessed at.
parse_llvm_pin() {
  local pin
  pin="$(pin_line "$1")" || return 1
  [[ -z "${pin}" ]] && return 0
  if [[ "${pin}" != git+*://*@* ]]; then
    echo "Error: cannot parse '${pin}' in ${1}." >&2
    echo "Expected one PEP 508 direct reference, e.g." >&2
    echo "  git+https://github.com/ROCm/llvm-project@aotriton/0.14b/rc0" >&2
    echo "scp-style URLs (git@github.com:org/repo) are not accepted: the ref" >&2
    echo "separator and the user separator are the same character." >&2
    return 1
  fi
  printf '%s\n%s\n' "${pin%@*}" "${pin##*@}"
}
