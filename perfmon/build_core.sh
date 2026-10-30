#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Builds libperfmon_core for one ROCm, as one multi-arch binary, and installs
#   <prefix>/core/include/perfmon/   perfmon_abi.h and the headers adapters use
#   <prefix>/core/lib/libperfmon_core+rocm<ver>.so
# Every subject of that ROCm, of every arch, links this one library
# (build_subject.sh), so it is built once per ROCm, not per arch.
#
# Its arch list is every arch of AOTriton's python/gpu_targets.py that this
# ROCm's compiler accepts, not only the fleet's current arches, so a new fleet
# arch rarely needs a rebuild. The list and the source commit are embedded in
# the library, and the runner reports both and refuses a device outside the
# list.
#
# Usage: build_core.sh [--build_dir <dir>] [--without_amdsmi] --rocm <ver> <prefix>
#   --rocm <ver>      the ROCm the toolchain must be, e.g. 10.1.0
#   <prefix>          e.g. ${AOTRITON_WORKDIR}/installed/perfmon
#   --build_dir       cmake build tree (default ${TMPDIR:-/tmp}/perfmon-core-rocm<ver>);
#                     never under <prefix>, which holds products only
#   --without_amdsmi  build the thermal stub; the runner then refuses
#                     AOTRITON_THERMAL_GATE=on. For hosts without amd-smi.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
. "${SCRIPT_DIR}/scripts/lib/rocm.sh"

ROCM=""
BUILD_DIR=""
AMDSMI=ON
while [ "$#" -gt 0 ]; do
  case "$1" in
    --rocm)           ROCM="$2"; shift 2 ;;
    --build_dir)      BUILD_DIR="$2"; shift 2 ;;
    --without_amdsmi) AMDSMI=OFF; shift ;;
    --) shift; break ;;
    -*) echo "Error: unknown option: $1" >&2; exit 1 ;;
    *) break ;;
  esac
done
if [ -z "${ROCM}" ] || [ "$#" -ne 1 ]; then
  echo "Usage: $0 [--build_dir <dir>] [--without_amdsmi] --rocm <ver> <prefix>" >&2
  exit 1
fi
PREFIX="$1"

perfmon_resolve_rocm_path
perfmon_require_rocm "${ROCM}"

HIPCC="${CXX:-}"
if [ -z "${HIPCC}" ]; then
  if [ -x "${ROCM_PATH}/bin/hipcc" ]; then HIPCC="${ROCM_PATH}/bin/hipcc"; else HIPCC=hipcc; fi
fi

# --- Arch list ------------------------------------------------------------
# Read as a literal from gpu_targets.py rather than imported, so this needs
# nothing but python3.
CANDIDATES="$(python3 - "${REPO_ROOT}/python/gpu_targets.py" <<'PYEOF'
import ast, sys
tree = ast.parse(open(sys.argv[1]).read())
for node in tree.body:
    if isinstance(node, ast.Assign) and any(
            getattr(t, 'id', None) == 'AOTRITON_SUPPORTED_GPUS' for t in node.targets):
        gpus = ast.literal_eval(node.value)
        break
else:
    sys.exit('AOTRITON_SUPPORTED_GPUS not found in ' + sys.argv[1])
seen = []
for gpu in gpus:
    arch = gpu.split('_mod')[0]
    if arch not in seen:
        seen.append(arch)
print(' '.join(seen))
PYEOF
)"

PROBE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/perfmon-arch-probe.XXXXXX")"
trap 'rm -rf "${PROBE_DIR}"' EXIT
cat > "${PROBE_DIR}/probe.hip" <<'EOF'
#include <hip/hip_runtime.h>
__global__ void probe(float* p) { p[threadIdx.x] = 1.0f; }
EOF
ARCHES=()
REJECTED=()
for arch in ${CANDIDATES}; do
  if "${HIPCC}" -x hip --offload-arch="${arch}" -c "${PROBE_DIR}/probe.hip" \
       -o "${PROBE_DIR}/probe.o" >/dev/null 2>&1; then
    ARCHES+=("${arch}")
  else
    REJECTED+=("${arch}")
  fi
done
if [ "${#ARCHES[@]}" -eq 0 ]; then
  echo "Error: ${HIPCC} accepted none of: ${CANDIDATES}" >&2
  exit 1
fi
echo "[build_core] arches: ${ARCHES[*]}" >&2
if [ "${#REJECTED[@]}" -ne 0 ]; then
  echo "[build_core] WARNING: ROCm ${ROCM}'s compiler rejects ${REJECTED[*]};" \
       "runners will refuse those devices." >&2
fi
ARCH_LIST="$(IFS=';'; echo "${ARCHES[*]}")"

# --- Source commit ---------------------------------------------------------
COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
if [ "${COMMIT}" != unknown ] && \
   [ -n "$(git -C "${REPO_ROOT}" status --porcelain -- perfmon/core cmake/LoadHIP.cmake 2>/dev/null)" ]; then
  COMMIT="${COMMIT}-dirty"
fi

# --- Build -------------------------------------------------------------------
BUILD_DIR="${BUILD_DIR:-${TMPDIR:-/tmp}/perfmon-core-rocm${ROCM}}"
INSTALL_DIR="${PREFIX%/}/core"
echo "[build_core] commit ${COMMIT}, amd-smi ${AMDSMI}, build ${BUILD_DIR}, install ${INSTALL_DIR}" >&2

cmake -S "${SCRIPT_DIR}/core" -B "${BUILD_DIR}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="${HIPCC}" \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
  -DPERFMON_ROCM_VERSION="${ROCM}" \
  -DPERFMON_CORE_ARCHES="${ARCH_LIST}" \
  -DPERFMON_CORE_COMMIT="${COMMIT}" \
  -DPERFMON_ENABLE_AMDSMI="${AMDSMI}"
cmake --build "${BUILD_DIR}"
cmake --install "${BUILD_DIR}"

LIB="${INSTALL_DIR}/lib/libperfmon_core+rocm${ROCM}.so"
if [ ! -f "${LIB}" ] || [ ! -f "${INSTALL_DIR}/include/perfmon/perfmon_abi.h" ]; then
  echo "Error: the install did not produce ${LIB} and its headers." >&2
  exit 1
fi
echo "[build_core] done: ${LIB}" >&2
