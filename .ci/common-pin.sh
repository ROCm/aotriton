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

# One git URL pin, `[<name> @ ]git+<scheme>://<url>@<ref>[#<sha1>]`:
#
#   git+https://github.com/ROCm/llvm-project@aotriton/0.14b/rc0#<sha1>
#   flydsl @ git+https://github.com/ROCm/FlyDSL@<sha1>
#
# Prints two lines: the origin (git+ prefix intact) and the COMMIT TO BUILD.
#
# Deliberately NOT called a PEP 508 parser. The flydsl-compiler.txt form is one
# -- fragment included, PEP 508 URLs carry RFC 3986 fragments -- but
# flydsl-llvm.txt has no requirement name, and giving it one would buy the
# shape and nothing else: there is no `llvm` distribution for pip to install,
# so `llvm @ ...` would be PEP 508 that can never be a pip requirement.
# Unifying both files is deferred to the next cycle.
#
# The fragment is what makes a moving ref safe. `aotriton/0.14b/rc0` advances
# as the RC does, and every artifact cache downstream is keyed on the resolved
# SHA -- so without a pinned SHA the identity of a build depends on when the
# mirror was last fetched. When a fragment is present it IS the commit and the
# ref before it is documentation; only 40 hex digits are accepted, because a
# fragment that is anything else is a typo rather than a second spelling.
#
# Split on the LAST '@' of the URL, which is unambiguous for this form and
# stays unambiguous for git+ssh://git@host/org/repo@ref. The scp-style
# git@github.com:org/repo is the one spelling the rule cannot disambiguate, so
# it is rejected rather than guessed at.
parse_git_pin() {
  local pin="$1" src="$2" url ref sha=""
  # Optional `<name> @ ` prefix. Requires a space on either side, so it cannot
  # eat the URL's own user@host.
  if [[ "${pin}" == *" @ "* ]]; then
    pin="${pin#* @ }"
  fi
  if [[ "${pin}" != git+*://*@* ]]; then
    echo "Error: cannot parse '${pin}' in ${src}." >&2
    echo "Expected a git URL pin, e.g." >&2
    echo "  git+https://github.com/ROCm/llvm-project@aotriton/0.14b/rc0#<sha1>" >&2
    echo "  flydsl @ git+https://github.com/ROCm/FlyDSL@<sha1>" >&2
    echo "scp-style URLs (git@github.com:org/repo) are not accepted: the ref" >&2
    echo "separator and the user separator are the same character." >&2
    return 1
  fi
  if [[ "${pin}" == *"#"* ]]; then
    sha="${pin##*#}"
    pin="${pin%#*}"
    if [[ ! "${sha}" =~ ^[0-9a-fA-F]{40}$ ]]; then
      echo "Error: '#${sha}' in ${src} is not a 40-digit commit SHA." >&2
      echo "The fragment pins the exact commit of the ref before it; drop it" >&2
      echo "entirely to resolve that ref at build time instead." >&2
      return 1
    fi
  fi
  url="${pin%@*}"
  ref="${pin##*@}"
  printf '%s\n%s\n' "${url}" "${sha:-${ref}}"
}

parse_llvm_pin() {
  local pin
  pin="$(pin_line "$1")" || return 1
  [[ -z "${pin}" ]] && return 0
  parse_git_pin "${pin}" "$1"
}

# The FlyDSL compiler pin, same shape. Sole reader of
# third_party/flydsl-compiler.txt for callers that need the ref to BUILD, as
# opposed to the pip requirement line the file also is.
parse_flydsl_pin() {
  local pin
  pin="$(pin_line "$1")" || return 1
  if [[ -z "${pin}" ]]; then
    echo "Error: ${1} is empty; it must name the FlyDSL compiler to install." >&2
    return 1
  fi
  parse_git_pin "${pin}" "$1"
}
