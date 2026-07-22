#!/bin/bash

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

if [[ "$#" -eq 0 ]]; then
  echo "Usage: build_triton_wheels.sh --wheel_output_dir <dir> --version_suffix <suffix> [--python <X.Y>] [--altwheel_yaml <yaml>] <hash1> [<hash2> ...]" >&2
  exit 1
fi

WHEEL_OUTPUT_DIR=""
TRITON_WHEEL_VERSION_SUFFIX=""
PYVER=""
ALTWHEEL_YAML=""
while [[ "$1" == --* ]]; do
  case "$1" in
    --wheel_output_dir) WHEEL_OUTPUT_DIR="$2"; shift 2 ;;
    --version_suffix)   TRITON_WHEEL_VERSION_SUFFIX="$2"; shift 2 ;;
    --python)           PYVER="$2"; shift 2 ;;
    --altwheel_yaml)    ALTWHEEL_YAML="$2"; shift 2 ;;
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
. "${SCRIPT_DIR}/common-altwheel.sh"
BASE_DOCKER_IMAGE="aotriton:base"
if [[ -n "${PYVER}" ]]; then
  BASE_DOCKER_IMAGE="aotriton:base-py${PYVER}"
  # Build on demand, same "build if missing" idiom as aotriton:base itself.
  if [ -z "$(docker images -q ${BASE_DOCKER_IMAGE} 2>/dev/null)" ]; then
    (cd "${SCRIPT_DIR}" && docker build --network=host -t ${BASE_DOCKER_IMAGE} \
      --build-arg "PYVER=${PYVER}" \
      -f buildenv-triton-py.Dockerfile .)
  fi
fi
# TRITON_GIT_ORIGIN may be overridden from the environment (e.g. by
# releasesuite-git-head.sh --triton_origin) to fetch Triton from a fork or a
# local checkout via the file:// protocol.
TRITON_GIT_ORIGIN="${TRITON_GIT_ORIGIN:-https://github.com/ROCm/triton}"

mkdir -p "${WHEEL_OUTPUT_DIR}"

# A hash may live in a different origin than TRITON_GIT_ORIGIN (altwheel's
# {hash, origin} map form) -- look it up, defaulting to TRITON_GIT_ORIGIN.
hash_origin() {
  local hash="$1"
  if [[ -z "${ALTWHEEL_YAML}" ]]; then
    echo "${TRITON_GIT_ORIGIN}"
    return
  fi
  local key origin
  for key in $(yq -r '.venvs | keys | .[]' "${ALTWHEEL_YAML}"); do
    if [[ "$(altwheel_venv_hash "${ALTWHEEL_YAML}" ".venvs.${key}")" == "${hash}" ]]; then
      origin=$(altwheel_venv_origin "${ALTWHEEL_YAML}" ".venvs.${key}")
      echo "${origin:-${TRITON_GIT_ORIGIN}}"
      return
    fi
  done
  echo "${TRITON_GIT_ORIGIN}"
}

# Optional: the env var (in THIS shell) holding a GitHub PAT for hash's venv
# entry, e.g. `pat_environ: GITHUB_TOKEN` in the altwheel yaml. Empty if the
# entry doesn't declare one -- no PAT is used for that hash.
hash_pat_environ() {
  local hash="$1"
  if [[ -z "${ALTWHEEL_YAML}" ]]; then
    echo ""
    return
  fi
  local key
  for key in $(yq -r '.venvs | keys | .[]' "${ALTWHEEL_YAML}"); do
    if [[ "$(altwheel_venv_hash "${ALTWHEEL_YAML}" ".venvs.${key}")" == "${hash}" ]]; then
      altwheel_venv_pat_environ "${ALTWHEEL_YAML}" ".venvs.${key}"
      return
    fi
  done
  echo ""
}

# One mirror volume per distinct origin, named "triton-mirror" for the
# default origin (unchanged from before) and a stable per-origin slug
# otherwise -- a harmless local cache like the default mirror.
mirror_volume_for_origin() {
  local origin="$1"
  if [[ "${origin}" == "${TRITON_GIT_ORIGIN}" ]]; then
    echo "triton-mirror"
  else
    echo "triton-mirror-$(printf '%s' "${origin}" | md5sum | cut -c1-12)"
  fi
}

# Check the wheel cache first -- only hashes that still need building
# require their origin's mirror synced at all, avoiding a needless
# git fetch (network round-trip) when everything is already cached.
NEEDED_HASHES=()
for HASH in "${TRITON_HASHES[@]}"; do
  SHORT="${HASH:0:8}"
  if ls "${WHEEL_OUTPUT_DIR}"/triton-*+*"${SHORT}"*.whl &>/dev/null; then
    echo "Wheel for ${SHORT} already cached, skipping."
    continue
  fi
  NEEDED_HASHES+=("${HASH}")
done

# Fail fast, before any docker work, if a needed hash names a pat_environ
# whose environment variable isn't actually set: the same PAT authenticates
# both the mirror fetch below and (if the Triton build downloads its own
# artifacts from GitHub) the build itself, so check it once up front.
for HASH in "${NEEDED_HASHES[@]}"; do
  pat_environ="$(hash_pat_environ "${HASH}")"
  if [[ -n "${pat_environ}" && -z "${!pat_environ:-}" ]]; then
    echo "Error: pat_environ '${pat_environ}' is set in the yaml for ${HASH:0:8} but that environment variable is empty/unset." >&2
    exit 1
  fi
done

declare -A SYNCED_VOLUMES
for HASH in "${NEEDED_HASHES[@]}"; do
  origin="$(hash_origin "${HASH}")"
  volume="$(mirror_volume_for_origin "${origin}")"
  if [[ -z "${SYNCED_VOLUMES[${volume}]:-}" ]]; then
    sync_mirror "${volume}" "${origin}" "${BASE_DOCKER_IMAGE}" "$(hash_pat_environ "${HASH}")"
    SYNCED_VOLUMES[${volume}]=1
  fi
done

# Build each hash that's still needed.
for HASH in "${NEEDED_HASHES[@]}"; do
  volume="$(mirror_volume_for_origin "$(hash_origin "${HASH}")")"
  pat_environ="$(hash_pat_environ "${HASH}")"

  # Forward the PAT by NAME only: Triton's own build (setup.py) may read it
  # directly under this name to download artifacts (e.g. a prebuilt
  # toolchain) from a private GitHub instance, sharing the same token used
  # to authenticate the mirror fetch above.
  PAT_ENV_ARG=()
  if [[ -n "${pat_environ}" ]]; then
    PAT_ENV_ARG=(-e "${pat_environ}")
  fi

  docker run --network=host -i --rm \
    -v "${volume}:/mirror:ro" \
    --mount "type=bind,source=$(realpath ${WHEEL_OUTPUT_DIR}),target=/cache/wheels" \
    --mount "type=bind,source=$(realpath ${SCRIPT_DIR}/runc-build-triton-wheel.sh),target=/tmp/runc-build-triton-wheel.sh,readonly" \
    --tmpfs "/scratch:exec" \
    -e TRITON_WHEEL_VERSION_SUFFIX="${TRITON_WHEEL_VERSION_SUFFIX}" \
    "${PAT_ENV_ARG[@]}" \
    "${BASE_DOCKER_IMAGE}" \
    bash -s "${HASH}" << 'EOF'
set -ex
HASH="$1"
scl enable gcc-toolset-13 -- bash /tmp/runc-build-triton-wheel.sh \
  file:///mirror "$HASH" /cache/wheels "$TRITON_WHEEL_VERSION_SUFFIX" /scratch/build
EOF
done
