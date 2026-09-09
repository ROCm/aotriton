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
                            flydsl-*+git<sha8>*llvm<sha8>*-cp<XY>-*.whl here is
                            a hit. The LLVM identity is part of the key: the pin
                            names a moving branch, and a wheel built against the
                            wrong LLVM miscompiles rather than fails.
    --flydsl_commit <ref>   Required. Branch, tag or SHA in the FlyDSL
                            compiler repo. Resolved to a SHA before caching.
    --flydsl_origin <url>   Default https://github.com/ROCm/FlyDSL (public).
     --llvm_tarball <path>  Use this prebuilt LLVM/MLIR tarball as-is. Without
                            it, .ci/build_llvm_tarball.sh is invoked (reading
                            third_party/flydsl-llvm.txt) into a sibling cache
                            directory, and its output is used.
      --llvm_origin <url>   Forwarded to build_llvm_tarball.sh.
      --llvm_commit <ref>   Forwarded to build_llvm_tarball.sh.
    --version_suffix <s>    Appended inside the wheel's local version segment,
                            after the git hash and before the LLVM tag:
                            <base>+git<sha8><s>.llvm<sha8>.
     --pat_environ <VAR>    Name (not value) of an environment variable
                            holding a GitHub PAT, for a private origin.
         --python <X.Y>     CPython to build for. Default 3.11. flydsl wheels
                            are ABI specific, so this is part of the cache key.
           --rocm <ver>     ROCm version for the build image. Default 7.2.4.
           --jobs <N>       Parallel build jobs for the LLVM build. FlyDSL's
                            own build.sh hardcodes -j$(nproc) and ignores it.
EOF
  exit "${1:-1}"
}

WHEEL_OUTPUT_DIR=""
FLYDSL_COMMIT=""
FLYDSL_ORIGIN=""
LLVM_TARBALL=""
LLVM_ORIGIN=""
LLVM_COMMIT=""
VERSION_SUFFIX=""
PAT_ENVIRON=""
PYVER="3.11"
ROCMVER="7.2.4"
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
    --llvm_origin)      LLVM_ORIGIN="$2"; shift 2 ;;
    --llvm_commit)      LLVM_COMMIT="$2"; shift 2 ;;
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/common-git-cache.sh"
. "${SCRIPT_DIR}/common-altwheel.sh"

# The *compiler* repo, and it is public, so nothing here needs a credential by
# default. The kernel-development fork is a different repository and is not a
# build input at all: its output reaches AOTriton vendored under
# modules/<family>/flyc/, never by clone.
FLYDSL_DEFAULT_ORIGIN="https://github.com/ROCm/FlyDSL"
FLYDSL_ORIGIN="${FLYDSL_ORIGIN:-${FLYDSL_DEFAULT_ORIGIN}}"

mkdir -p "${WHEEL_OUTPUT_DIR}"
WHEEL_OUTPUT_DIR="$(realpath "${WHEEL_OUTPUT_DIR}")"

# --- The LLVM half, BEFORE the wheel cache key ---
# One flag, two modes, the same shape as build-test.sh's --flydsl_wheel versus
# the pinned flydsl-compiler.txt: an explicit tarball is used as-is, and
# without one we build the tarball third_party/flydsl-llvm.txt names. The
# sibling directory keeps two caches with different lifetimes apart -- wheels
# are per FlyDSL commit, per Python AND per LLVM; tarballs are per LLVM commit
# only.
#
# THIS RUNS BEFORE EITHER WHEEL CACHE PROBE, and that ordering is the point.
# The LLVM identity is part of what a flydsl wheel IS -- this file exists
# because a wheel built against the wrong LLVM miscompiles register spills and
# returns wrong numbers rather than failing -- so it has to be part of the
# cache key, and a key cannot be computed after it has been used. The pin
# normally names a MOVING branch (aotriton/0.14b/rc0 advances as the RC does),
# so "same FlyDSL commit, same Python" genuinely does describe two different
# wheels either side of an RC bump, and the old key served the older one
# forever.
#
# It costs a cache hit the LLVM *resolution* (a mirror sync and a rev-parse,
# seconds) that it used to skip. It does NOT cost an LLVM build: that has its
# own cache, keyed on the resolved SHA, checked inside build_llvm_tarball.sh.
if [[ -z "${LLVM_TARBALL}" ]]; then
  LLVM_TARBALL_DIR="${LLVM_TARBALL_DIR:-$(dirname "${WHEEL_OUTPUT_DIR}")/llvm-tarballs}"
  LLVM_ARGS=(--tarball_output_dir "${LLVM_TARBALL_DIR}" --python "${PYVER}")
  [[ -n "${LLVM_ORIGIN}" ]] && LLVM_ARGS+=(--llvm_origin "${LLVM_ORIGIN}")
  [[ -n "${LLVM_COMMIT}" ]] && LLVM_ARGS+=(--llvm_commit "${LLVM_COMMIT}")
  [[ -n "${JOBS}" ]] && LLVM_ARGS+=(--jobs "${JOBS}")
  # Forwarded like the rest: --pat_environ exists precisely for a private
  # --llvm_origin, and the origin IS forwarded, so dropping the credential name
  # leaves the inner sync_mirror to fetch unauthenticated and die -- an hour
  # into a run the caller believed was credentialed.
  [[ -n "${PAT_ENVIRON}" ]] && LLVM_ARGS+=(--pat_environ "${PAT_ENVIRON}")
  LLVM_TARBALL="$(bash "${SCRIPT_DIR}/build_llvm_tarball.sh" "${LLVM_ARGS[@]}")" || exit 1
fi
if [[ ! -f "${LLVM_TARBALL}" ]]; then
  echo "Error: LLVM tarball not found: ${LLVM_TARBALL}" >&2
  exit 1
fi
LLVM_TARBALL="$(realpath "${LLVM_TARBALL}")"

# The LLVM identity, as eight characters that go into the wheel's local version
# segment and into the cache glob. build_llvm_tarball.sh names its output
# llvm-<sha8>-<distro>-x64.tar.gz, so the SHA is already in the filename and
# there is nothing to recompute. A tarball handed in with --llvm_tarball may be
# named anything, so fall back to a digest of its basename: not the LLVM SHA,
# but still a stable identity that distinguishes two different hand-supplied
# tarballs, which is all the key needs to do.
LLVM_TARBALL_BASE="$(basename "${LLVM_TARBALL}")"
if [[ "${LLVM_TARBALL_BASE}" =~ ^llvm-([0-9a-fA-F]{8}) ]]; then
  LLVM_SHA8="${BASH_REMATCH[1]}"
else
  LLVM_SHA8="$(printf '%s' "${LLVM_TARBALL_BASE}" | md5sum | cut -c1-8)"
  echo "Note: ${LLVM_TARBALL_BASE} does not follow llvm-<sha8>-*; keying the wheel cache on ${LLVM_SHA8} (digest of the name)." >&2
fi
# Appended INSIDE the local version segment, after the git hash and after any
# caller --version_suffix: <base>+git<flydsl sha8><suffix>.llvm<llvm sha8>.
# PEP 440 allows [a-z0-9.] there, and the wheel filename is what both cache
# probes below glob against.
WHEEL_VERSION_SUFFIX="${VERSION_SUFFIX}.llvm${LLVM_SHA8}"

# flydsl wheels are CPython-ABI specific (flydsl-...-cp313-cp313-linux_x86_64.whl)
# and AOTriton's CMake already fails a build whose --flydsl_wheel cp tag does
# not match the build venv. So the ABI tag is part of the cache key: a wheel
# cached for another Python is not a hit for this one. Same reasoning, and the
# same helper, as build_triton_wheels.sh.
ABI_TAG="$(altwheel_abi_glob "${PYVER}")"
cached_wheel() {
  ls "${WHEEL_OUTPUT_DIR}"/flydsl-*+*"${1:0:8}"*"llvm${LLVM_SHA8}"*"${ABI_TAG}"*.whl 2>/dev/null | head -n1
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
# FlyDSL needs ROCm to configure at all (see buildenv-flydsl.Dockerfile), so
# the wheel is built in a ROCm-bearing derivative of the base image rather
# than in the base image the Triton wheel uses. Same "build if missing" idiom.
FLYDSL_DOCKER_IMAGE="aotriton:buildenv-flydsl-py${PYVER}-rocm${ROCMVER}"
if [ -z "$(docker images -q "${FLYDSL_DOCKER_IMAGE}" 2>/dev/null)" ]; then
  (cd "${SCRIPT_DIR}" && docker build --network=host -t "${FLYDSL_DOCKER_IMAGE}" \
    --build-arg "PYVER=${PYVER}" \
    --build-arg "ROCM_VERSION_IN_URL=${ROCMVER}" \
    -f buildenv-flydsl.Dockerfile .) >&2
fi

# One mirror volume per distinct origin, mirror_volume_for_origin()'s shape.
if [[ "${FLYDSL_ORIGIN}" == "${FLYDSL_DEFAULT_ORIGIN}" ]]; then
  MIRROR_VOLUME="flydsl-mirror"
else
  MIRROR_VOLUME="flydsl-mirror-$(printf '%s' "${FLYDSL_ORIGIN}" | md5sum | cut -c1-12)"
fi
# CHECKED, and this script has no `set -e` to check it for us. Same hazard
# build_llvm_tarball.sh records: the mirror volume is never deleted, so a warm
# one plus a swallowed fetch failure resolves the tag below against the STALE
# tip and caches the resulting wheel under a name that claims otherwise.
if ! sync_mirror "${MIRROR_VOLUME}" "${FLYDSL_ORIGIN}" "${BASE_DOCKER_IMAGE}" "${PAT_ENVIRON}" >&2; then
  echo "Error: could not sync the git mirror ${MIRROR_VOLUME} from ${FLYDSL_ORIGIN}." >&2
  echo "Refusing to resolve '${FLYDSL_COMMIT}' against a possibly stale mirror." >&2
  exit 1
fi

# --flydsl_commit is normally a tag (v0.3.1), which is exactly the moving-ref
# problem the LLVM pin has: resolve it against the mirror before the wheel
# name -- and therefore the cache key -- depends on it.
RESOLVED=$(docker run --rm -i \
  -v "${MIRROR_VOLUME}:/mirror:ro" \
  "${BASE_DOCKER_IMAGE}" \
  bash -s "${FLYDSL_COMMIT}" <<'EOF'
set -e
git config --global --add safe.directory '*'
git -C /mirror rev-parse --verify "$1^{commit}"
EOF
)
if [[ ! "${RESOLVED}" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "Error: '${FLYDSL_COMMIT}' did not resolve to a commit in ${MIRROR_VOLUME} (origin ${FLYDSL_ORIGIN})." >&2
  exit 1
fi
if [[ "${RESOLVED}" != "${FLYDSL_COMMIT}" ]]; then
  echo "Resolved ${FLYDSL_COMMIT} -> ${RESOLVED}" >&2
fi

HIT="$(cached_wheel "${RESOLVED}")"
if [[ -n "${HIT}" ]]; then
  echo "FlyDSL wheel for ${RESOLVED:0:8} (python ${PYVER}) already cached, skipping." >&2
  realpath "${HIT}"
  exit 0
fi

# Persistent, not tmpfs: this volume holds the extracted LLVM prefix (several
# GB) as well as the FlyDSL checkout and build tree, and keeping it means the
# next wheel skips the extraction and rebuilds incrementally.
FLYDSL_BUILD_VOLUME="${FLYDSL_BUILD_VOLUME:-aotriton-flydsl-build}"
docker volume create --name "${FLYDSL_BUILD_VOLUME}" >/dev/null

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
  -v "${FLYDSL_BUILD_VOLUME}:/build" \
  --mount "type=bind,source=${WHEEL_OUTPUT_DIR},target=/cache/wheels" \
  --mount "type=bind,source=${LLVM_TARBALL},target=/cache/llvm/$(basename "${LLVM_TARBALL}"),readonly" \
  --mount "type=bind,source=$(realpath "${SCRIPT_DIR}/runc-build-flydsl-wheel.sh"),target=/tmp/runc-build-flydsl-wheel.sh,readonly" \
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
  "$VERSION_SUFFIX" /build "$JOBS"
EOF

HIT="$(cached_wheel "${RESOLVED}")"
if [[ -z "${HIT}" ]]; then
  echo "Error: build reported success but no flydsl-*${RESOLVED:0:8}*llvm${LLVM_SHA8}*${ABI_TAG}*.whl appeared in ${WHEEL_OUTPUT_DIR}." >&2
  exit 1
fi
realpath "${HIT}"
