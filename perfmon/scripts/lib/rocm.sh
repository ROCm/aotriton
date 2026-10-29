# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# ROCm discovery shared by build_core.sh and build_subject.sh. Source it.

# Exports ROCM_PATH: the environment's if set, else `rocm-sdk path --root`
# (theRock), else /opt/rocm -- the order AOTriton's cmake/LoadHIP.cmake uses.
# Old release tags predate that module and need ROCM_PATH in the environment,
# so it is resolved once here for every build step.
perfmon_resolve_rocm_path() {
  if [ -z "${ROCM_PATH:-}" ] && command -v rocm-sdk >/dev/null 2>&1; then
    ROCM_PATH="$(rocm-sdk path --root)" || {
      echo "Error: 'rocm-sdk path --root' failed. If the venv is read-only, run it" >&2
      echo "       once where it is writable, or set ROCM_PATH." >&2
      return 1
    }
  fi
  ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
  if [ ! -d "${ROCM_PATH}" ]; then
    echo "Error: ROCM_PATH=${ROCM_PATH} does not exist." >&2
    return 1
  fi
  export ROCM_PATH
}

# Prints the <major>.<minor>.<patch> of the ROCm in use, or nothing. Asks
# rocm-sdk, then $ROCM_PATH/.info/version. Not hipconfig: that reports the HIP
# version, which is not the ROCm release.
perfmon_probe_rocm_version() {
  local v=""
  if command -v rocm-sdk >/dev/null 2>&1; then
    v="$(rocm-sdk version 2>/dev/null | head -1 || true)"
  fi
  if [ -z "$v" ] && [ -f "${ROCM_PATH:-/nonexistent}/.info/version" ]; then
    v="$(head -1 "${ROCM_PATH}/.info/version" || true)"
  fi
  printf '%s' "$v" | grep -oE '^[0-9]+\.[0-9]+\.[0-9]+' || true
}

# Fails unless the ROCm in use is exactly <ver>. Every artifact carries <ver>
# in its name, so building with any other toolchain would mislabel it.
perfmon_require_rocm() {
  local want="$1" got
  if ! printf '%s' "${want}" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$'; then
    echo "Error: --rocm takes <major>.<minor>.<patch>, got '${want}'." >&2
    return 1
  fi
  got="$(perfmon_probe_rocm_version)"
  if [ -z "${got}" ]; then
    echo "Error: cannot determine the ROCm version (tried rocm-sdk version and" >&2
    echo "       ${ROCM_PATH}/.info/version)." >&2
    return 1
  fi
  if [ "${got}" != "${want}" ]; then
    echo "Error: --rocm ${want} was asked for, but the toolchain at ${ROCM_PATH}" >&2
    echo "       is ROCm ${got}." >&2
    return 1
  fi
  echo "[perfmon] ROCm ${got} at ${ROCM_PATH}" >&2
}
