#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Host half of the LLVM/MLIR tarball build: option parsing, the pin, the cache
# check, the git mirror and one `docker run` of runc-build-llvm-tarball.sh.
# Same split as build_triton_wheels.sh + runc-build-triton-wheel.sh, and for
# the same reason -- the in-container half must stay usable without Docker.
#
# The product is a tarball that .ci/build_flydsl_wheel.sh feeds to FlyDSL's
# setup.py as MLIR_PATH. Nothing else in the tree consumes it, and no default
# AOTriton build ever runs this script.
#
# The ONLY thing written to stdout is the absolute path of the tarball, so a
# caller can do `tarball=$(bash .ci/build_llvm_tarball.sh ...)`. Progress,
# docker output and errors all go to stderr.

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

usage() {
  cat <<'EOF' >&2
Usage: build_llvm_tarball.sh --tarball_output_dir <dir> [options]
Options:
  --tarball_output_dir <dir>  Required. Tarball cache; a matching
                              llvm-<sha8>-<distro>-x64.tar.gz here is a hit.
       --llvm_origin <url>    Override the git origin from
                              third_party/flydsl-llvm.txt.
       --llvm_commit <ref>    Override the ref from that file. Branch, tag or
                              SHA; it is resolved to a SHA before caching.
      --pat_environ <VAR>     Name (not value) of an environment variable
                              holding a GitHub PAT, for a private origin.
           --python <X.Y>     CPython for the build container. Default 3.11.
              --jobs <N>      Parallel build jobs. Default: nproc/2 inside the
                              container.
EOF
  exit "${1:-1}"
}

TARBALL_OUTPUT_DIR=""
LLVM_ORIGIN=""
LLVM_COMMIT=""
PAT_ENVIRON=""
PYVER="3.11"
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
    --tarball_output_dir) TARBALL_OUTPUT_DIR="$2"; shift 2 ;;
    --llvm_origin)        LLVM_ORIGIN="$2"; shift 2 ;;
    --llvm_commit)        LLVM_COMMIT="$2"; shift 2 ;;
    --pat_environ)        PAT_ENVIRON="$2"; shift 2 ;;
    --python)             PYVER="$2"; shift 2 ;;
    --jobs)               JOBS="$2"; shift 2 ;;
    -h|--help)            usage 0 ;;
    *) echo "Unknown option: $1" >&2; usage ;;
  esac
done
if [[ -z "${TARBALL_OUTPUT_DIR}" ]]; then
  echo "Error: --tarball_output_dir is required." >&2
  usage
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/common-git-cache.sh"

# The pin AOTriton keeps for LLVM, independent of the FlyDSL compiler and
# kernel pins. This script is only its consumer: the file arrives with the
# third_party/ pin policy, alongside the configure-time check that refuses a
# build when the pin is non-empty and no local FlyDSL wheel was supplied.
#
# It exists because upstream LLVM miscompiles register spills and 0.14b needs
# a fixed one, so the steady state is a file with nothing in it but its own
# comment. "Empty" means no non-comment, non-blank line -- the comment survives
# an emptying so the next person still learns what the file is for.
LLVM_PIN_FILE="${SCRIPT_DIR}/../third_party/flydsl-llvm.txt"

# One PEP 508 direct reference: git+<scheme>://<url>@<ref>. Split on the LAST
# '@', which is unambiguous for that form and stays unambiguous for
# git+ssh://git@host/org/repo@ref. The scp-style git@github.com:org/repo is the
# one spelling the rule cannot disambiguate, so it is rejected rather than
# guessed at.
parse_llvm_pin() {
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
  [[ "${count}" -eq 0 ]] && return 0
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

if [[ -z "${LLVM_ORIGIN}" || -z "${LLVM_COMMIT}" ]]; then
  if [[ ! -f "${LLVM_PIN_FILE}" ]]; then
    echo "Error: ${LLVM_PIN_FILE} does not exist and --llvm_origin/--llvm_commit were not both given." >&2
    exit 1
  fi
  # Command substitution, not process substitution: a parse failure has to
  # propagate here, and `readarray < <(...)` reports the status of readarray.
  PIN_TEXT="$(parse_llvm_pin "${LLVM_PIN_FILE}")" || exit 1
  PIN=()
  [[ -n "${PIN_TEXT}" ]] && readarray -t PIN <<< "${PIN_TEXT}"
  if [[ "${#PIN[@]}" -eq 0 ]]; then
    # Not a failure of this script so much as a question about why it was
    # called: an empty pin is the steady state, and it means the released
    # FlyDSL wheel is fine and nobody needs an hour of LLVM.
    echo "Error: ${LLVM_PIN_FILE} is empty -- upstream LLVM is believed good," >&2
    echo "so there is nothing to build here. Install the wheel pinned by" >&2
    echo "third_party/flydsl-compiler.txt instead, or pass --llvm_origin and" >&2
    echo "--llvm_commit explicitly to build a specific LLVM anyway." >&2
    exit 1
  fi
  LLVM_ORIGIN="${LLVM_ORIGIN:-${PIN[0]#git+}}"
  LLVM_COMMIT="${LLVM_COMMIT:-${PIN[1]}}"
fi

mkdir -p "${TARBALL_OUTPUT_DIR}"
TARBALL_OUTPUT_DIR="$(realpath "${TARBALL_OUTPUT_DIR}")"

# The tarball name carries a distro slug (the aotriton:base-py* family is
# AlmaLinux 8) because .ci/triton-patch/docker-script-build.sh's LLVM
# filenames do, and one naming scheme beats two. It is not something this
# script chooses, so match on a glob rather than reconstructing it.
cached_tarball() {
  ls "${TARBALL_OUTPUT_DIR}"/llvm-"${1:0:8}"-*-x64.tar.gz 2>/dev/null | head -n1
}

# Cache before network, part 1: a pin that is already a SHA needs no
# resolution, so a hit here costs nothing at all -- not even a git fetch.
if [[ "${LLVM_COMMIT}" =~ ^[0-9a-fA-F]{40}$ ]]; then
  HIT="$(cached_tarball "${LLVM_COMMIT}")"
  if [[ -n "${HIT}" ]]; then
    echo "LLVM tarball for ${LLVM_COMMIT:0:8} already cached, skipping." >&2
    realpath "${HIT}"
    exit 0
  fi
fi

BASE_DOCKER_IMAGE="aotriton:base-py${PYVER}"
# Build on demand, same "build if missing" idiom as aotriton:base itself.
if [ -z "$(docker images -q "${BASE_DOCKER_IMAGE}" 2>/dev/null)" ]; then
  (cd "${SCRIPT_DIR}" && docker build --network=host -t "${BASE_DOCKER_IMAGE}" \
    --build-arg "PYVER=${PYVER}" \
    -f base.Dockerfile .) >&2
fi

# One mirror volume per distinct origin: "llvm-mirror" for the origin the pin
# normally names, a stable per-origin slug otherwise. Lifted from
# build_triton_wheels.sh's mirror_volume_for_origin(), which solves exactly
# this and whose volumes are equally harmless local caches.
LLVM_DEFAULT_ORIGIN="https://github.com/ROCm/llvm-project"
if [[ "${LLVM_ORIGIN}" == "${LLVM_DEFAULT_ORIGIN}" ]]; then
  MIRROR_VOLUME="llvm-mirror"
else
  MIRROR_VOLUME="llvm-mirror-$(printf '%s' "${LLVM_ORIGIN}" | md5sum | cut -c1-12)"
fi

# CHECKED, and this script has no `set -e` to check it for us. The mirror
# volume is never deleted, so a warm one plus a swallowed fetch failure
# resolves the moving branch below against the STALE tip -- and then builds,
# names and caches the previous RC under an authoritative-looking
# llvm-<sha8>-<distro> filename that nothing will ever invalidate. Failing here
# costs a re-run; not failing costs a wrong tarball, forever.
if ! sync_mirror "${MIRROR_VOLUME}" "${LLVM_ORIGIN}" "${BASE_DOCKER_IMAGE}" "${PAT_ENVIRON}" >&2; then
  echo "Error: could not sync the git mirror ${MIRROR_VOLUME} from ${LLVM_ORIGIN}." >&2
  echo "Refusing to resolve '${LLVM_COMMIT}' against a possibly stale mirror." >&2
  exit 1
fi

# Resolve the ref against the freshly synced mirror. The pin names a moving
# branch (aotriton/0.14b/rc0 advances as the RC does), and the cache key must
# not: two different builds under one filename is a wrong tarball served
# forever. The mirror fetches +refs/*:refs/*, so a branch or tag name resolves
# here exactly as it would at the origin.
RESOLVED=$(docker run --rm -i \
  -v "${MIRROR_VOLUME}:/mirror:ro" \
  "${BASE_DOCKER_IMAGE}" \
  bash -s "${LLVM_COMMIT}" <<'EOF'
set -e
git config --global --add safe.directory '*'
git -C /mirror rev-parse --verify "$1^{commit}"
EOF
)
if [[ ! "${RESOLVED}" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "Error: '${LLVM_COMMIT}' did not resolve to a commit in ${MIRROR_VOLUME} (origin ${LLVM_ORIGIN})." >&2
  exit 1
fi
if [[ "${RESOLVED}" != "${LLVM_COMMIT}" ]]; then
  echo "Resolved ${LLVM_COMMIT} -> ${RESOLVED}" >&2
fi

# Cache before network, part 2: for a branch pin this is the first point at
# which the key is known. The fetch above cost seconds; the build below costs
# an hour, and that is the thing a cache hit has to prevent.
HIT="$(cached_tarball "${RESOLVED}")"
if [[ -n "${HIT}" ]]; then
  echo "LLVM tarball for ${RESOLVED:0:8} already cached, skipping." >&2
  realpath "${HIT}"
  exit 0
fi

# Not a tmpfs. build_triton_wheels.sh builds Triton in `--tmpfs /scratch`, but
# an LLVM tree with assertions is tens of gigabytes and RAM-backed scratch of
# that size is not something a build host can be assumed to have. A named
# volume also survives the run, so a re-spin of the same RC rebuilds
# incrementally instead of from scratch. runc-build-llvm-tarball.sh keys its
# subdirectories by SHA, so two commits never share a CMakeCache.
LLVM_BUILD_VOLUME="${LLVM_BUILD_VOLUME:-aotriton-llvm-build}"
docker volume create --name "${LLVM_BUILD_VOLUME}" >/dev/null

PAT_ENV_ARG=()
if [[ -n "${PAT_ENVIRON}" ]]; then
  # Forward by NAME only, never `-e NAME=value`: docker resolves the value from
  # this shell's environment, so the token never reaches an argv a `set -x` or
  # a `ps` could show. `-e NAME` only sees EXPORTED variables, so export it
  # ourselves rather than silently forwarding nothing.
  export "${PAT_ENVIRON}"
  PAT_ENV_ARG=(-e "${PAT_ENVIRON}")
fi

# stdout of the container goes to stderr: this script's stdout is the tarball
# path and nothing else.
docker run --network=host -i --rm \
  -v "${MIRROR_VOLUME}:/mirror:ro" \
  -v "${LLVM_BUILD_VOLUME}:/build" \
  --mount "type=bind,source=${TARBALL_OUTPUT_DIR},target=/cache/llvm" \
  --mount "type=bind,source=$(realpath "${SCRIPT_DIR}/runc-build-llvm-tarball.sh"),target=/tmp/runc-build-llvm-tarball.sh,readonly" \
  "${PAT_ENV_ARG[@]}" \
  "${BASE_DOCKER_IMAGE}" \
  bash -s "${RESOLVED}" "${JOBS}" >&2 << 'EOF'
set -ex
COMMIT="$1"
JOBS="$2"
scl enable gcc-toolset-13 -- bash /tmp/runc-build-llvm-tarball.sh \
  file:///mirror "$COMMIT" /cache/llvm /build "$JOBS"
EOF

HIT="$(cached_tarball "${RESOLVED}")"
if [[ -z "${HIT}" ]]; then
  echo "Error: build reported success but no llvm-${RESOLVED:0:8}-*-x64.tar.gz appeared in ${TARBALL_OUTPUT_DIR}." >&2
  exit 1
fi
realpath "${HIT}"
