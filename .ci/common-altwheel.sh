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

# Optional: names the environment variable holding a GitHub PAT for this venv
# entry's origin (see docs/AltWheelExample.yaml). Empty if unset -- no PAT is
# used during git clone or the Triton build in that case.
altwheel_venv_pat_environ() {
  local yaml="$1" path="$2"
  local t
  t=$(yq -r "${path} | tag" "$yaml" 2>/dev/null)
  if [ "$t" = "!!map" ]; then
    yq -r "${path}.pat_environ // \"\"" "$yaml"
  else
    echo ""
  fi
}

# Find the built wheel matching <hash> in <host_wheel_dir> (a directory of
# triton-*+git<hash8>*.whl files, e.g. .ci's WHEEL_CACHE_DIR or .tune's
# <workdir>/scratch/triton). Prints the absolute path, or nothing if missing.
altwheel_resolve_wheel_path() {
  local host_wheel_dir="$1" hash="$2" short
  short="${hash:0:8}"
  ls "${host_wheel_dir}"/triton-*+git${short}*.whl 2>/dev/null | head -n1
}

# Rewrite every venvs.* entry of <yaml> IN PLACE into a plain wheel-path
# string. AOTriton's own codegen (python/codegen/root.py's
# _load_altwheel_config) requires this -- it calls value.startswith(...)
# unconditionally on every entry, so a {hash, origin} map crashes it. The
# single source of truth for this: both .ci's release suite and .tune call
# this instead of maintaining separate resolution logic.
#
# <host_wheel_dir>      directory to find built wheels in (see above)
# <container_wheel_dir> path prefix to write into the yaml -- pass the same
#                       value as <host_wheel_dir> when the yaml is consumed
#                       from the same path it was resolved from (.tune's
#                       case); pass a container-internal mount point when
#                       the yaml crosses into a different container (.ci's
#                       release pipeline, where wheels are bind-mounted at
#                       /cache/wheels but built on the host at WHEEL_CACHE_DIR)
# <default_hash>        the hash to use for .venvs.default -- not a fallback
#                       in the sense of "least preferred": it's the actual
#                       submodule-pinned commit, the normal/standard Triton
#                       everything builds against. Altwheel exists to bump
#                       Triton independently for specific arches/kernels; an
#                       alt venv is the odd case, not the default one. Only
#                       used when the source yaml doesn't already set
#                       .venvs.default itself (e.g. an arch-only altwheel
#                       yaml like .ci/0.13.1b.yaml, which has no default key).
altwheel_resolve_config() {
  local yaml="$1" host_wheel_dir="$2" container_wheel_dir="$3" default_hash="$4"
  local key hash host_wheel

  local resolved_default_hash
  resolved_default_hash=$(altwheel_venv_hash "${yaml}" ".venvs.default")
  [ -z "${resolved_default_hash}" ] && resolved_default_hash="${default_hash}"
  host_wheel=$(altwheel_resolve_wheel_path "${host_wheel_dir}" "${resolved_default_hash}")
  if [ -z "${host_wheel}" ]; then
    echo "Error: no built wheel found for altwheel default venv (hash ${resolved_default_hash:0:8})" >&2
    return 1
  fi
  yq -i ".venvs.default = \"${container_wheel_dir}/$(basename "${host_wheel}")\"" "${yaml}"

  for key in $(yq -r '.venvs | keys | .[]' "${yaml}"); do
    [ "${key}" = "default" ] && continue
    hash=$(altwheel_venv_hash "${yaml}" ".venvs.${key}")
    host_wheel=$(altwheel_resolve_wheel_path "${host_wheel_dir}" "${hash}")
    if [ -z "${host_wheel}" ]; then
      echo "Error: no built wheel found for altwheel venv '${key}' (hash ${hash:0:8})" >&2
      return 1
    fi
    yq -i ".venvs.${key} = \"${container_wheel_dir}/$(basename "${host_wheel}")\"" "${yaml}"
  done
}
