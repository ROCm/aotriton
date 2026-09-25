#!/bin/bash
# Runs inside the AlmaLinux 8 ROCm Docker container (manylinux_2_28 environment).
# Bind-mounted by build_inside() in releasesuite-git-head.sh at /tmp/runc-manylinux-build-tar.sh
# and invoked as: bash -l /tmp/runc-manylinux-build-tar.sh <args>
# Positional args: $1=TARGETS $2=WHEEL_CFG $3=ASAN_MODE $4=ARCH_LIST
#   TARGETS: comma-separated subset of gpu,common,runtime naming what THIS
#   invocation (one container, one ROCm version) produces. Always exactly one
#   cmake build; the set only selects its options and what gets packaged.
#     gpu             GPU kernel images                   -> /output
#     runtime         libaotriton_v2.so, linking the       -> /output
#                     common archive cached by an earlier invocation
#     common          aotriton_common.a + headers          -> /cache/aotriton-common
#     gpu,common      the image build, whose build tree already contains
#                     aotriton_common.a, packaged to the cache as a by-product
#     common,runtime  no-cache fallback: an ordinary build of both halves,
#                     installing only the runtime and caching nothing
#   ARCH_LIST: "ALL" (default) or a ';'-separated GPU arch list forwarded
#   to build-release.sh as its arch_list arg (becomes AOTRITON_TARGET_ARCH).
#
# Container prerequisites:
#   Mounts:
#     /mirror        — bare AOTriton git mirror (read-only); the source is
#                      cloned from it into /src/aotriton at the requested commit
#     /output        — destination for output tar archives
#     /cache         — shared cache; /cache/wheels holds pre-built Triton wheels,
#                      /cache/aotriton-common the common archive that a
#                      `common` target writes and a `runtime` target reads
#   Environment variables:
#     AOTRITON_GIT_COMMIT    — commit to check out from /mirror
#     AOTRITON_BUILD_PATH    — build directory, e.g. /scratch/build/aotriton
#                              Recommended: mount /scratch as tmpfs with exec flag
#                              and sufficient memory to hold the full build tree.
#     AOTRITON_INSTALL_PREFIX — install prefix, e.g. /scratch/install
#                               AOTRITON_INSTALL_PATH is derived as $AOTRITON_INSTALL_PREFIX/aotriton
#     PIP_CACHE_DIR          — pip download cache (bind-mounted from host)
#     AOTRITON_USE_LOCAL_FLYDSL_WHEEL — optional; path (inside this container,
#                              normally under /cache) to a FlyDSL compiler
#                              wheel built by .ci/build_flydsl_wheel.sh. When
#                              unset the build installs the wheel
#                              third_party/flydsl-compiler.txt pins.
#   Tools (provided by the ROCm AlmaLinux 8 image):
#     hipconfig        — to locate ROCM_PATH
#     gcc-toolset-13   — C++17 compiler via scl enable (non-asan path only)
#     cpp              — preprocessor used to extract the HIP version number

set -ex

# --- Arguments ---
TARGETS="$1"
WHEEL_CFG="$2"
ASAN_MODE="${3:-OFF}"
ARCH_LIST="${4:-ALL}"

want_gpu=0
want_common=0
want_runtime=0
IFS=',' read -ra _targets <<< "${TARGETS}"
for _t in "${_targets[@]}"; do
  case "${_t}" in
    gpu)     want_gpu=1 ;;
    common)  want_common=1 ;;
    runtime) want_runtime=1 ;;
    *)
      echo "Error: unknown build target '${_t}' in '${TARGETS}'." >&2
      echo "       Valid targets: gpu, common, runtime (comma-separated)." >&2
      exit 1 ;;
  esac
done
if [ $((want_gpu + want_common + want_runtime)) -eq 0 ]; then
  echo "Error: \$1 is empty; expected a comma-separated subset of gpu,common,runtime." >&2
  exit 1
fi
# One invocation is one cmake build, and these two want incompatible options
# out of it (AOTRITON_NOIMAGE_MODE=OFF vs ON). Run them as separate invocations.
if [ ${want_gpu} -eq 1 ] && [ ${want_runtime} -eq 1 ]; then
  echo "Error: 'gpu' and 'runtime' cannot share one invocation ('${TARGETS}')." >&2
  exit 1
fi
# Only an image build compiles kernels; everything else runs with
# AOTRITON_NOIMAGE_MODE=ON.
if [ ${want_gpu} -eq 1 ]; then
  NOIMAGE_MODE=OFF
else
  NOIMAGE_MODE=ON
fi

# --- Validate environment ---
if [ -z "${AOTRITON_BUILD_PATH}" ]; then
  echo "Error: AOTRITON_BUILD_PATH is not set." >&2; exit 1
fi
if [ -z "${AOTRITON_INSTALL_PREFIX}" ]; then
  echo "Error: AOTRITON_INSTALL_PREFIX is not set." >&2; exit 1
fi
export AOTRITON_INSTALL_PATH="${AOTRITON_INSTALL_PREFIX}/aotriton"

# --- Materialize the AOTriton source from the local mirror ---
# Shallow-fetch only the requested commit from the read-only /mirror volume
# (offline, fast). The mirror sets uploadpack.allowAnySHA1InWant, so the SHA
# is fetchable without cloning the full history. Non-recursive:
# Triton (the only submodule) is installed from a pre-built wheel, so
# third_party/triton is never checked out here.
if [ -z "${AOTRITON_GIT_COMMIT}" ]; then
  echo "Error: AOTRITON_GIT_COMMIT is not set." >&2; exit 1
fi
git config --global --add safe.directory '*'
rm -rf /src/aotriton
git init /src/aotriton
git -C /src/aotriton remote add origin file:///mirror
git -C /src/aotriton fetch --depth=1 origin "${AOTRITON_GIT_COMMIT}"
git -C /src/aotriton checkout -f FETCH_HEAD

# pip (running as root) refuses a cache dir not owned by root and silently
# disables caching. PIP_CACHE_DIR=/cache/pip is bind-mounted from the host
# and owned by the host UID, so take ownership inside the container.
if [ -n "${PIP_CACHE_DIR}" ] && [ -d "${PIP_CACHE_DIR}" ]; then
  chown -R "$(id -u):$(id -g)" "${PIP_CACHE_DIR}" || true
fi

# --- Detect ROCm and HIP version ---
GIT_SHORT=$(git -C /src/aotriton rev-parse --short=12 HEAD)
export ROCM_PATH=$(hipconfig --rocmpath)
if [ -z "${ROCM_PATH}" ]; then
  echo "Error: ROCM_PATH is empty. hipconfig --rocmpath failed." >&2
  exit 1
fi
printf '#include <hip/hip_version.h>\nHIP_VERSION_MAJOR . HIP_VERSION_MINOR\n' > /tmp/print_hip_version.h
if [[ "${ASAN_MODE}" == "ON" ]]; then
  # No gcc-toolset/cpp in this path; use clang's preprocessor.
  hipver=$(${ROCM_PATH}/llvm/bin/clang -E -P -x c -I${ROCM_PATH}/include /tmp/print_hip_version.h | tail -n 1 | sed 's/ //g')
else
  hipver=$(scl enable gcc-toolset-13 "cpp -I${ROCM_PATH}/include /tmp/print_hip_version.h" | tail -n 1 | sed 's/ //g')
fi

# The tag states the minimum glibc these binaries need, so read it off the
# container's glibc instead of restating the base image's version here.
libc_version=$(getconf GNU_LIBC_VERSION | awk '{print $2}')
IFS=. read -r libc_major libc_minor _ <<< "${libc_version}"
if [ -z "${libc_major}" ] || [ -z "${libc_minor}" ]; then
  echo "Error: cannot parse a glibc version out of 'getconf GNU_LIBC_VERSION' (got '${libc_version}')." >&2
  exit 1
fi
MANYLINUX_TAG="manylinux_${libc_major}_${libc_minor}"

asan_suffix=""
if [[ "${ASAN_MODE}" == "ON" ]]; then
  asan_suffix="+asan"
fi

# --- Build ---
# Only image builds embed a Triton wheel. Runtime builds (NOIMAGE_MODE=ON)
# run with AOTRITON_NOIMAGE_MODE=ON and skip Triton entirely, so no wheel
# config is passed (WHEEL_CFG is "NONE" in that case).
build_args=("${NOIMAGE_MODE}" "${ARCH_LIST}")
if [ "${NOIMAGE_MODE}" == "OFF" ]; then
  if [[ "${WHEEL_CFG}" == *.yml || "${WHEEL_CFG}" == *.yaml ]]; then
    cmake_arg="-DAOTRITON_ALT_TRITON_WHEEL_CONFIG_FILE=${WHEEL_CFG}"
  else
    cmake_arg="-DAOTRITON_USE_LOCAL_TRITON_WHEEL=${WHEEL_CFG}"
  fi
  build_args+=("${cmake_arg}")
  # Only image builds compile kernels, so only they need a FlyDSL compiler at
  # all. Forwarded from the environment rather than a positional argument
  # because it is optional; see the caller in releasesuite-git-head.sh.
  if [ -n "${AOTRITON_USE_LOCAL_FLYDSL_WHEEL}" ]; then
    if [ ! -f "${AOTRITON_USE_LOCAL_FLYDSL_WHEEL}" ]; then
      echo "Error: AOTRITON_USE_LOCAL_FLYDSL_WHEEL points at a file this container cannot see: ${AOTRITON_USE_LOCAL_FLYDSL_WHEEL}" >&2
      exit 1
    fi
    build_args+=("-DAOTRITON_USE_LOCAL_FLYDSL_WHEEL=${AOTRITON_USE_LOCAL_FLYDSL_WHEEL}")
  fi
fi

# Naming follows the /output artifacts; only the location differs, because this
# is a cache rather than a release artifact. No -rocm${hipver}: the archive
# holds only translation units that never call a real hip*() function, which is
# what lets one build of it link into every ROCm version in the matrix. The
# manylinux tag and the asan suffix stay -- a static archive is sensitive to both.
COMMON_CACHE_DIR="/cache/aotriton-common"
COMMON_TARBALL="aotriton-${GIT_SHORT}${asan_suffix}-${MANYLINUX_TAG}_x86_64-common-static.tar.gz"

if [ ${want_common} -eq 1 ] && [ ${want_gpu} -eq 0 ] && [ ${want_runtime} -eq 0 ]; then
  build_args+=("-DAOTRITON_COMMON_LIBRARY_ONLY_MODE=ON")
elif [ ${want_runtime} -eq 1 ] && [ ${want_common} -eq 0 ]; then
  # Link the archive an earlier invocation cached. It is keyed by commit, so a
  # miss means no `common` target ran for THIS commit -- never a stale hit.
  if [ ! -f "${COMMON_CACHE_DIR}/${COMMON_TARBALL}" ]; then
    echo "Error: no cached common archive at ${COMMON_CACHE_DIR}/${COMMON_TARBALL}" >&2
    echo "       Run 'common' or 'gpu,common' for this commit first, or use" >&2
    echo "       'common,runtime' to build both halves in this invocation." >&2
    exit 1
  fi
  COMMON_STAGE="${AOTRITON_BUILD_PATH}/common-stage"
  rm -rf "${COMMON_STAGE}" && mkdir -p "${COMMON_STAGE}"
  tar xz -C "${COMMON_STAGE}" < "${COMMON_CACHE_DIR}/${COMMON_TARBALL}"
  build_args+=("-DAOTRITON_COMMON_LIBRARY=${COMMON_STAGE}/aotriton/lib/libaotriton_common.a")
fi

if [[ "${ASAN_MODE}" == "ON" ]]; then
  build_args+=(
    "-DAOTRITON_ENABLE_ASAN_CLANG=ON"
    "-DCMAKE_C_COMPILER=${ROCM_PATH}/llvm/bin/clang"
    "-DCMAKE_CXX_COMPILER=${ROCM_PATH}/llvm/bin/clang++"
  )
  # Use theRock clang directly — no scl gcc-toolset wrapper.
  bash /src/aotriton/.ci/build-release.sh "${build_args[@]}"
else
  scl enable gcc-toolset-13 -- bash /src/aotriton/.ci/build-release.sh "${build_args[@]}"
fi

# --- Package (every archive must have aotriton/ as the root directory) ---
if [ ${want_gpu} -eq 1 ]; then
  tarbase=aotriton-${GIT_SHORT}${asan_suffix}-images
  cd "${AOTRITON_INSTALL_PREFIX}"
  for d in $(ls aotriton/lib/aotriton.images/); do
    tarfile=${tarbase}-$d.tar.gz
    tar cz "aotriton/lib/aotriton.images/$d" > /output/${tarfile}
  done
fi

if [ ${want_runtime} -eq 1 ]; then
  tarfile=aotriton-${GIT_SHORT}${asan_suffix}-${MANYLINUX_TAG}_x86_64-rocm${hipver}-shared.tar.gz
  cd "${AOTRITON_INSTALL_PREFIX}" && tar cz aotriton > /output/${tarfile}
fi

# Skipped when a runtime was also requested: that combination installs the
# runtime only, so there is no separated common half to cache.
if [ ${want_common} -eq 1 ] && [ ${want_runtime} -eq 0 ]; then
  # AOTRITON_COMMON_LIBRARY_ONLY_MODE installs the archive; a gpu build leaves
  # it only in the build tree. Staged rather than tarred in place so the gpu
  # build's .so and images do not end up in a "common" tarball.
  common_a="${AOTRITON_INSTALL_PATH}/lib/libaotriton_common.a"
  if [ ! -f "${common_a}" ]; then
    common_a="${AOTRITON_BUILD_PATH}/v3src/common/libaotriton_common.a"
  fi
  common_pkg="${AOTRITON_BUILD_PATH}/common-pkg"
  rm -rf "${common_pkg}" && mkdir -p "${common_pkg}/aotriton/lib"
  cp -a "${AOTRITON_INSTALL_PATH}/include" "${common_pkg}/aotriton/include"
  cp "${common_a}" "${common_pkg}/aotriton/lib/"
  mkdir -p "${COMMON_CACHE_DIR}"
  cd "${common_pkg}" && tar cz aotriton > "${COMMON_CACHE_DIR}/${COMMON_TARBALL}"
fi

# Debug: drop into interactive shell after everything is done so the full
# build and install tree can be inspected. Requires -t from the caller.
if [[ "${SUITE_DEBUG:-0}" == "1" ]]; then
  if [ -t 0 ]; then
    echo "=== DEBUG MODE: build complete. Dropping into interactive shell. ===" >&2
    bash -i </dev/tty >/dev/tty 2>&1 || true
  else
    echo "=== DEBUG MODE: build complete, but no TTY available. Skipping interactive shell. ===" >&2
  fi
fi
