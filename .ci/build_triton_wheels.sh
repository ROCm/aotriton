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
BASE_DOCKER_IMAGE="aotriton:base"
TRITON_MIRROR_VOLUME="triton-mirror"
TRITON_GIT_ORIGIN="https://github.com/ROCm/triton"

mkdir -p "${WHEEL_OUTPUT_DIR}"

# Ensure triton-mirror bare volume exists and is up to date
docker volume create --name "${TRITON_MIRROR_VOLUME}"
if docker run --rm \
     -v "${TRITON_MIRROR_VOLUME}:/mirror" \
     "${BASE_DOCKER_IMAGE}" \
     bash -c "git -C /mirror rev-parse --git-dir" &>/dev/null; then
  # fetch --all: required because we don't know which branch contains the target hash
  docker run --network=host --rm \
    -v "${TRITON_MIRROR_VOLUME}:/mirror" \
    "${BASE_DOCKER_IMAGE}" \
    bash -c "git -C /mirror fetch --all"
else
  docker run --network=host --rm \
    -v "${TRITON_MIRROR_VOLUME}:/mirror" \
    "${BASE_DOCKER_IMAGE}" \
    bash -c "set -ex
git clone --bare ${TRITON_GIT_ORIGIN} /mirror
git -C /mirror config uploadpack.allowReachableSHA1InWant true"
fi

# Build wheel for each hash, skipping if already cached
for HASH in "${TRITON_HASHES[@]}"; do
  SHORT="${HASH:0:8}"
  if ls "${WHEEL_OUTPUT_DIR}"/triton-*+*"${SHORT}"*.whl &>/dev/null; then
    echo "Wheel for ${SHORT} already cached, skipping."
    continue
  fi

  docker run --network=host --rm \
    -v "${TRITON_MIRROR_VOLUME}:/mirror:ro" \
    --mount "type=bind,source=$(realpath ${WHEEL_OUTPUT_DIR}),target=/cache/wheels" \
    --tmpfs "/scratch:exec" \
    -e TRITON_WHEEL_VERSION_SUFFIX="${TRITON_WHEEL_VERSION_SUFFIX}" \
    "${BASE_DOCKER_IMAGE}" \
    bash -s "${HASH}" << 'EOF'
set -ex
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
