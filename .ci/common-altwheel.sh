#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Read altwheel YAML venvs.* entries: either a bare hash string (old syntax)
# or a {hash, origin} map (new syntax, for a hash living in a different
# Triton origin than the default). yq's if/then/else and try/catch don't
# reliably branch on mixed scalar/map types across yq versions -- branch in
# bash on `tag` instead, which is stable everywhere.

altwheel_venv_hash() {
  local yaml="$1" path="$2"
  local t
  t=$(yq -r "${path} | tag" "$yaml" 2>/dev/null)
  if [ "$t" = "!!map" ]; then
    yq -r "${path}.hash" "$yaml"
  else
    yq -r "${path} // \"\"" "$yaml"
  fi
}

altwheel_venv_origin() {
  local yaml="$1" path="$2"
  local t
  t=$(yq -r "${path} | tag" "$yaml" 2>/dev/null)
  if [ "$t" = "!!map" ]; then
    yq -r "${path}.origin // \"\"" "$yaml"
  else
    echo ""
  fi
}
