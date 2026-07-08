#!/bin/bash

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

if [[ "$#" -eq 0 ]]; then
  echo "Usage: build_triton_wheels.sh --wheel_output_dir <dir> --version_suffix <suffix> <hash1> [<hash2> ...]" >&2
  exit 1
fi

WHEEL_OUTPUT_DIR=""
TRITON_WHEEL_VERSION_SUFFIX=""
while [[ "$1" == --* ]]; do
  case "$1" in
    --wheel_output_dir) WHEEL_OUTPUT_DIR="$2"; shift 2 ;;
    --version_suffix)   TRITON_WHEEL_VERSION_SUFFIX="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done
if [[ -z "${WHEEL_OUTPUT_DIR}" ]]; then
  echo "Error: --wheel_output_dir is required." >&2; exit 1
fi
if [[ -z "${TRITON_WHEEL_VERSION_SUFFIX}" ]]; then
  echo "Error: --version_suffix is required." >&2; exit 1
fi

TRITON_HASHES=("$@")
if [[ "${#TRITON_HASHES[@]}" -eq 0 ]]; then
  echo "Error: at least one Triton commit hash is required." >&2; exit 1
fi

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
. "${SCRIPT_DIR}/common-git-cache.sh"
BASE_DOCKER_IMAGE="aotriton:base"
TRITON_MIRROR_VOLUME="triton-mirror"
# TRITON_GIT_ORIGIN may be overridden from the environment (e.g. by
# releasesuite-git-head.sh --triton_origin) to fetch Triton from a fork or a
# local checkout via the file:// protocol.
TRITON_GIT_ORIGIN="${TRITON_GIT_ORIGIN:-https://github.com/ROCm/triton}"

mkdir -p "${WHEEL_OUTPUT_DIR}"

# GitHub -> local mirror volume. The per-hash loop below then clones the exact
# commit from the local mirror (file:///mirror).
sync_mirror "${TRITON_MIRROR_VOLUME}" "${TRITON_GIT_ORIGIN}" "${BASE_DOCKER_IMAGE}"

# Build wheel for each hash, skipping if already cached
for HASH in "${TRITON_HASHES[@]}"; do
  SHORT="${HASH:0:8}"
  if ls "${WHEEL_OUTPUT_DIR}"/triton-*+*"${SHORT}"*.whl &>/dev/null; then
    echo "Wheel for ${SHORT} already cached, skipping."
    continue
  fi

  docker run --network=host -i --rm \
    -v "${TRITON_MIRROR_VOLUME}:/mirror:ro" \
    --mount "type=bind,source=$(realpath ${WHEEL_OUTPUT_DIR}),target=/cache/wheels" \
    --tmpfs "/scratch:exec" \
    -e TRITON_WHEEL_VERSION_SUFFIX="${TRITON_WHEEL_VERSION_SUFFIX}" \
    "${BASE_DOCKER_IMAGE}" \
    bash -s "${HASH}" << 'EOF'
set -ex
git config --global --add safe.directory '*'
HASH="$1"
SHORT="${HASH:0:8}"
rm -rf /scratch/build
git init /scratch/build
git -C /scratch/build remote add origin file:///mirror
git -C /scratch/build fetch --depth=1 origin "${HASH}"
git -C /scratch/build checkout FETCH_HEAD
scl enable gcc-toolset-13 -- python -m pip wheel /scratch/build -w /cache/wheels/
ls /cache/wheels/triton-*+*${SHORT}*.whl
EOF
done
