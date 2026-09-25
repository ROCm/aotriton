#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Host half of the FlyDSL wheel build: option parsing, the wheel cache check,
# the git mirror, the LLVM tarball it needs, and one `docker run` of
# runc-build-flydsl-wheel.sh. Same shape as build_triton_wheels.sh.
#
# The product is a path handed to `.ci/build-test.sh --flydsl_wheel` (or to
# cmake as -DAOTRITON_USE_LOCAL_FLYDSL_WHEEL). build-test.sh gains nothing from
# this script and still just builds; orchestration lives in the caller, which
# for a release is .ci/releasesuite-git-head.sh.
#
# The ONLY thing written to stdout is the absolute path of the wheel.

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

usage() {
  cat <<'EOF' >&2
Usage: build_flydsl_wheel.sh --wheel_output_dir <dir> --flydsl_commit <ref> [options]
Options:
  --wheel_output_dir <dir>  Required. Wheel cache; a matching
                            flydsl-*+git<sha8>*llvm<sha12>.p<n>-cp<XY>-*.whl here is
                            a hit. The LLVM identity is part of the key: the pin
                            names a moving branch, and a wheel built against the
                            wrong LLVM miscompiles rather than fails.
    --flydsl_commit <ref>   Required. Branch, tag or SHA in the FlyDSL
                            compiler repo. Resolved to a SHA before caching.
    --flydsl_origin <url>   Default https://github.com/ROCm/FlyDSL (public).
     --llvm_tarball <path>  Required. The LLVM/MLIR tarball to build against,
                            named llvm-<sha12>-<...>.tar.gz so the wheel cache
                            can be keyed on it. Produce one with
                            .ci/build_llvm_tarball.sh, which reads
                            third_party/flydsl-llvm.txt.
    --version_suffix <s>    Appended inside the wheel's local version segment,
                            after the git hash and before the LLVM tag:
                            <base>+git<sha8><s>.llvm<sha12>.p<patches>.
     --pat_environ <VAR>    Name (not value) of an environment variable
                            holding a GitHub PAT, for a private origin.
         --python <X.Y>     CPython to build for. Default 3.11. flydsl wheels
                            are ABI specific, so this is part of the cache key.
           --rocm <ver>     TheRock version for the build image. Default
                            10.2.0a20260918, the newest in the release suite's
                            list. FlyJitRuntime links HIP but AOTriton never
                            launches through it, so this does not affect the
                            kernels the wheel produces.
           --jobs <N>       CMAKE_BUILD_PARALLEL_LEVEL for the FlyDSL build.
                            Advisory only: FlyDSL's own build.sh passes an
                            explicit -j$(nproc), which beats it. Nothing here
                            builds LLVM.
EOF
  exit "${1:-1}"
}

WHEEL_OUTPUT_DIR=""
FLYDSL_COMMIT=""
FLYDSL_ORIGIN=""
LLVM_TARBALL=""
VERSION_SUFFIX=""
PAT_ENVIRON=""
PYVER="3.11"
ROCMVER="10.2.0a20260918"
JOBS=""
while [[ "$1" == --* ]]; do
  # `shift 2` with one argument left FAILS WITHOUT SHIFTING, so $1 never changes
  # and a trailing valued flag spins this loop forever at 100% CPU with no
  # output. .ci/run-test.sh guards its own -k the same way and for the same
  # reason. -h/--help is the only flag here that takes no value.
  case "$1" in
    -h|--help) ;;
    *) [[ "$#" -ge 2 ]] || { echo "Error: $1 requires a value." >&2; usage ;} ;;
  esac
  case "$1" in
    --wheel_output_dir) WHEEL_OUTPUT_DIR="$2"; shift 2 ;;
    --flydsl_commit)    FLYDSL_COMMIT="$2"; shift 2 ;;
    --flydsl_origin)    FLYDSL_ORIGIN="$2"; shift 2 ;;
    --llvm_tarball)     LLVM_TARBALL="$2"; shift 2 ;;
    --version_suffix)   VERSION_SUFFIX="$2"; shift 2 ;;
    --pat_environ)      PAT_ENVIRON="$2"; shift 2 ;;
    --python)           PYVER="$2"; shift 2 ;;
    --rocm)             ROCMVER="$2"; shift 2 ;;
    --jobs)             JOBS="$2"; shift 2 ;;
    -h|--help)          usage 0 ;;
    *) echo "Unknown option: $1" >&2; usage ;;
  esac
done
if [[ -z "${WHEEL_OUTPUT_DIR}" ]]; then
  echo "Error: --wheel_output_dir is required." >&2; usage
fi
if [[ -z "${FLYDSL_COMMIT}" ]]; then
  echo "Error: --flydsl_commit is required." >&2; usage
fi
if [[ -z "${LLVM_TARBALL}" ]]; then
  echo "Error: --llvm_tarball is required. Build one with" >&2
  echo "  .ci/build_llvm_tarball.sh --tarball_output_dir <dir>" >&2
  echo "which reads third_party/flydsl-llvm.txt for the pin." >&2
  usage
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/common-git-cache.sh"
. "${SCRIPT_DIR}/common-altwheel.sh"
. "${SCRIPT_DIR}/common-therock.sh"

# The *compiler* repo, and it is public, so nothing here needs a credential by
# default. The kernel-development fork is a different repository and is not a
# build input at all: its output reaches AOTriton vendored under
# modules/<family>/flyc/, never by clone.
FLYDSL_DEFAULT_ORIGIN="https://github.com/ROCm/FlyDSL"
FLYDSL_ORIGIN="${FLYDSL_ORIGIN:-${FLYDSL_DEFAULT_ORIGIN}}"

mkdir -p "${WHEEL_OUTPUT_DIR}"
WHEEL_OUTPUT_DIR="$(realpath "${WHEEL_OUTPUT_DIR}")"

# --- The LLVM half is the CALLER's ---
# This script does not build LLVM. Its caller resolves the pin and hands the
# tarball down, so the LLVM is a sibling input of the wheel rather than a
# private dependency of it -- which is what lets a future Triton build be
# pointed at the same tarball. .ci/releasesuite-git-head.sh calls
# build_llvm_tarball.sh then this, in that order.
#
# The tarball is still part of the wheel's cache KEY, and that has not moved:
# a wheel built against the wrong LLVM miscompiles register spills and returns
# wrong numbers rather than failing, so "same FlyDSL commit, same Python" does
# not identify a wheel.
if [[ ! -f "${LLVM_TARBALL}" ]]; then
  echo "Error: LLVM tarball not found: ${LLVM_TARBALL}" >&2
  exit 1
fi
LLVM_TARBALL="$(realpath "${LLVM_TARBALL}")"

# The LLVM identity that goes into the wheel's local version segment and into
# the cache glob. build_llvm_tarball.sh names its output
# llvm-<sha12>-<distro>-x64.tar.gz, so the SHA is already in the filename.
#
# A tarball whose name does not carry one is REFUSED rather than keyed on
# something invented. A digest of the basename would make `custom.tar.gz` mean
# whatever it meant last time -- replace its contents and the cache serves a
# wheel built against the old LLVM, which is exactly the miscompile
# third_party/flydsl-llvm.txt exists to keep out. Rename the tarball to
# llvm-<sha12>-<anything>.tar.gz and the key is honest again.
LLVM_TARBALL_BASE="$(basename "${LLVM_TARBALL}")"
if [[ "${LLVM_TARBALL_BASE}" =~ ^llvm-([0-9a-fA-F]{12}) ]]; then
  LLVM_SHA="${BASH_REMATCH[1]}"
else
  echo "Error: ${LLVM_TARBALL_BASE} does not name its LLVM commit." >&2
  echo "--llvm_tarball must be llvm-<sha12>-<...>.tar.gz: the wheel cache is keyed" >&2
  echo "on that SHA, and a name carrying no identity cannot distinguish two" >&2
  echo "different LLVMs." >&2
  exit 1
fi
# The patches in .ci/flydsl-patch/ are part of what the wheel IS -- one of them
# moves a build requirement -- so their count goes in the name. Adding a patch
# must not return the wheel built before it. (A count, not a digest: editing a
# patch in place still collides. Bump nothing and rebuild by hand if you do
# that.)
FLYDSL_PATCH_COUNT="$(ls "${SCRIPT_DIR}"/flydsl-patch/*.patch 2>/dev/null | wc -l)"
# Appended INSIDE the local version segment, after the git hash and after any
# caller --version_suffix:
# <base>+git<flydsl sha8><suffix>.llvm<llvm sha12>.p<patches>.
# PEP 440 allows [a-z0-9.] there, and the wheel filename is what both cache
# probes below glob against.
WHEEL_VERSION_SUFFIX="${VERSION_SUFFIX}.llvm${LLVM_SHA}.p${FLYDSL_PATCH_COUNT}"

# flydsl wheels are CPython-ABI specific (flydsl-...-cp313-cp313-linux_x86_64.whl)
# and AOTriton's CMake already fails a build whose --flydsl_wheel cp tag does
# not match the build venv. So the ABI tag is part of the cache key: a wheel
# cached for another Python is not a hit for this one. Same reasoning, and the
# same helper, as build_triton_wheels.sh.
ABI_TAG="$(altwheel_abi_glob "${PYVER}")"
# Every component of the version this run would produce, in order, so a wheel
# built from the same FlyDSL and LLVM but under a different --version_suffix is
# a MISS rather than a silently mislabelled hit.
cached_wheel() {
  ls "${WHEEL_OUTPUT_DIR}"/flydsl-*+git"${1:0:8}${WHEEL_VERSION_SUFFIX}"-*"${ABI_TAG}"*.whl 2>/dev/null | head -n1
}

# Cache before the FlyDSL network round-trip: a SHA needs no resolution, so a
# hit here skips the fetch and the container. The LLVM half above has already
# run -- it has to, the key depends on it -- but that is seconds of resolution
# against an existing tarball cache, not an LLVM build.
if [[ "${FLYDSL_COMMIT}" =~ ^[0-9a-fA-F]{40}$ ]]; then
  HIT="$(cached_wheel "${FLYDSL_COMMIT}")"
  if [[ -n "${HIT}" ]]; then
    echo "FlyDSL wheel for ${FLYDSL_COMMIT:0:8} (python ${PYVER}) already cached, skipping." >&2
    realpath "${HIT}"
    exit 0
  fi
fi

BASE_DOCKER_IMAGE="aotriton:base-py${PYVER}"
if [ -z "$(docker images -q "${BASE_DOCKER_IMAGE}" 2>/dev/null)" ]; then
  (cd "${SCRIPT_DIR}" && docker build --network=host -t "${BASE_DOCKER_IMAGE}" \
    --build-arg "PYVER=${PYVER}" \
    -f base.Dockerfile .) >&2
fi
# --flydsl_commit is normally a tag (v0.3.1), which is exactly the moving-ref
# problem the LLVM pin has: make it available in the git mirror and
# resolve it against the origin before the wheel name -- and therefore the
# cache key -- depends on it. CHECKED, and this script has no `set -e` to
# check it for us: same hazard build_llvm_tarball.sh records, a swallowed
# failure caches the resulting wheel under a name that claims otherwise.
MIRROR_VOLUME="flydsl-mirror"
RESOLVED=$(sync_mirror "${MIRROR_VOLUME}" "${FLYDSL_ORIGIN}" "${BASE_DOCKER_IMAGE}" "${PAT_ENVIRON}" "${FLYDSL_COMMIT}") || exit 1
if [[ "${RESOLVED}" != "${FLYDSL_COMMIT}" ]]; then
  echo "Resolved ${FLYDSL_COMMIT} -> ${RESOLVED}" >&2
fi

HIT="$(cached_wheel "${RESOLVED}")"
if [[ -n "${HIT}" ]]; then
  echo "FlyDSL wheel for ${RESOLVED:0:8} (python ${PYVER}) already cached, skipping." >&2
  realpath "${HIT}"
  exit 0
fi

# Built HERE, below the cache check, not before it. Only the wheel build below
# uses this image -- sync_mirror above runs in
# BASE_DOCKER_IMAGE -- and it is by far the most expensive step in the script,
# a whole ROCm install via pip. Building it first meant a run whose wheel was
# already cached still paid for it, which is what made a cached FlyDSL look
# like a rebuild next to a cached LLVM tarball that returns immediately.
#
# FlyDSL needs ROCm to configure at all -- lib/Runtime/ROCm/CMakeLists.txt does
# find_package(hip REQUIRED) under its only backend -- so the wheel is built in
# a ROCm-bearing derivative of the base image. theRock.Dockerfile, the same one
# the release suite uses, with BASE_TAG pointing at this Python: a wheel's cp
# tag has to match the venv that will install it.
#
# ROCm comes from TheRock rather than dnf packages because that is what the
# rest of AOTriton targets, and .ci/flydsl-patch/ teaches FlyDSL's CMake to
# find it -- a TheRock root is a site-packages directory, not /opt/rocm.
FLYDSL_DOCKER_IMAGE="aotriton:buildenv-rocm${ROCMVER}-py${PYVER}"
if [ -z "$(docker images -q "${FLYDSL_DOCKER_IMAGE}" 2>/dev/null)" ]; then
  # The index has to follow the version: theRock.Dockerfile's own default is a
  # RELEASE index, so a nightly --rocm left to that default resolves no wheel.
  (cd "${SCRIPT_DIR}" && docker build --network=host -t "${FLYDSL_DOCKER_IMAGE}" \
    --build-arg "BASE_TAG=base-py${PYVER}" \
    --build-arg "THEROCK_VERSION=${ROCMVER}" \
    --build-arg "THEROCK_PIP_INDEX_URL=$(therock_pip_index_url "${ROCMVER}")" \
    -f theRock.Dockerfile .) >&2
fi

PAT_ENV_ARG=()
if [[ -n "${PAT_ENVIRON}" ]]; then
  # By NAME only: docker resolves the value from this shell's environment, so
  # the token never enters an argv. `-e NAME` only sees exported variables.
  export "${PAT_ENVIRON}"
  PAT_ENV_ARG=(-e "${PAT_ENVIRON}")
fi

# Container stdout goes to stderr; this script's stdout is the wheel path.
docker run --network=host -i --rm \
  -v "${MIRROR_VOLUME}:/mirror:ro" \
  --tmpfs "/scratch:exec" \
  --mount "type=bind,source=${WHEEL_OUTPUT_DIR},target=/cache/wheels" \
  --mount "type=bind,source=${LLVM_TARBALL},target=/cache/llvm/$(basename "${LLVM_TARBALL}"),readonly" \
  --mount "type=bind,source=$(realpath "${SCRIPT_DIR}/runc-build-flydsl-wheel.sh"),target=/tmp/runc-build-flydsl-wheel.sh,readonly" \
  --mount "type=bind,source=$(realpath "${SCRIPT_DIR}/flydsl-patch"),target=/tmp/flydsl-patch,readonly" \
  -e AOTRITON_FLYDSL_PATCH_DIR=/tmp/flydsl-patch \
  "${PAT_ENV_ARG[@]}" \
  "${FLYDSL_DOCKER_IMAGE}" \
  bash -s "${RESOLVED}" "$(basename "${LLVM_TARBALL}")" "${WHEEL_VERSION_SUFFIX}" "${JOBS}" >&2 << 'EOF'
set -ex
COMMIT="$1"
LLVM_TARBALL_NAME="$2"
VERSION_SUFFIX="$3"
JOBS="$4"
scl enable gcc-toolset-13 -- bash /tmp/runc-build-flydsl-wheel.sh \
  file:///mirror "$COMMIT" "/cache/llvm/${LLVM_TARBALL_NAME}" /cache/wheels \
  "$VERSION_SUFFIX" /scratch/build "$JOBS"
EOF

HIT="$(cached_wheel "${RESOLVED}")"
if [[ -z "${HIT}" ]]; then
  echo "Error: build reported success but no flydsl-*+git${RESOLVED:0:8}${WHEEL_VERSION_SUFFIX}-*${ABI_TAG}*.whl appeared in ${WHEEL_OUTPUT_DIR}." >&2
  exit 1
fi
realpath "${HIT}"
