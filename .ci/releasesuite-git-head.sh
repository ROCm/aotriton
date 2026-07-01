#!/bin/bash

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

if ! command -v yq &> /dev/null; then
  cat <<EOF >&2
Command 'yq' could not be found. Install it with
dnf install yq
or
snap install yq
EOF
  exit 1
fi

function help() {
  cat <<EOF >&2
Usage: releasesuite-git-head.sh [-h] [options..] <output directory>
Options:
                    -h: show help and exit.
         -r <ROCM ver>: build ROCM runtime image
               --image: build GPU images.
             --runtime: build all C++ runtimes.
         --yaml <.yml>: Use yml config file to build the release
        --origin <url>: Override the git HTTPS origin URL (default: auto-detected from remote)
By default both GPU images and runtimes are built.
If either --image or --runtime is specified, the missing one will not be built.

The YAML configuration file follows the format shown in docs/AltWheelExample.yaml.
However it accepts GIT SHA1 for Triton wheels instead.
The build process will
1. Build Triton wheels from the SHA1
2. Replace SHA1 with actual wheel path and use the replaced yaml file to build
   AOTriton
EOF
  exit $1
}

TEMP=$(getopt -o hr: --longoptions image,runtime,yaml:,origin: -- "$@")

if [ $? -ne 0 ]; then
  echo "Error: Invalid option." >&2
  help 1
fi

eval set -- "$TEMP"

SUITE_SELECT_IMAGE=-1
SUITE_SELECT_RUNTIME=-1
SUITE_RUNTIME_LIST=(6.4.4 7.0.3 7.1.1 7.2.3)
CMDLIST=()
SUITE_DEFAULT_SELECTION=1
SUITE_YAML=""
SUITE_ORIGIN=""

while true; do
  case "$1" in
    -h)
      help 0
      ;;
    --image)
      SUITE_SELECT_IMAGE=1
      SUITE_DEFAULT_SELECTION=0
      ;;
    --runtime)
      SUITE_SELECT_RUNTIME=1
      SUITE_DEFAULT_SELECTION=0
      ;;
    -r)
      SUITE_SELECT_RUNTIME=1
      SUITE_DEFAULT_SELECTION=0
      shift
      CMDLIST+=("$1")
      ;;
    --yaml)
      shift
      SUITE_YAML="$1"
      ;;
    --origin)
      shift
      SUITE_ORIGIN="$1"
      ;;
    '--')
      shift
      break
      ;;
  esac
  shift
done

if [[ ${#CMDLIST[@]} -ne 0 ]]; then
  SUITE_RUNTIME_LIST=("${CMDLIST[@]}")
fi

if [ "$#" -ne 1 ]; then
  echo "$@"
  echo 'Missing argument <output directory>.' >&2
  help 1
fi

if [ ${SUITE_SELECT_IMAGE} -lt 0 ]; then
  SUITE_SELECT_IMAGE=${SUITE_DEFAULT_SELECTION}
fi

if [ ${SUITE_SELECT_RUNTIME} -lt 0 ]; then
  SUITE_SELECT_RUNTIME=${SUITE_DEFAULT_SELECTION}
fi

echo "SUITE_RUNTIME_LIST ${SUITE_RUNTIME_LIST[@]}"

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
. "${SCRIPT_DIR}/common-vars.sh"
. "${SCRIPT_DIR}/common-setup-volume.sh"
. "${SCRIPT_DIR}/common-git-https-origin.sh"
if [[ -n "${SUITE_ORIGIN}" ]]; then
  GIT_HTTPS_ORIGIN="${SUITE_ORIGIN}"
fi
. "${SCRIPT_DIR}/include-altwheel.sh"

GIT_COMMIT=$(git rev-parse HEAD)

BASE_DOCKER_IMAGE="aotriton:base"

# build base docker image
if [ -z "$(docker images -q ${BASE_DOCKER_IMAGE} 2>/dev/null)" ]; then
  (cd "${SCRIPT_DIR}" && docker build --network=host -t ${BASE_DOCKER_IMAGE} -f base.Dockerfile .)
fi

SOURCE_VOLUME="aotriton-src-shared"
LOCAL_DIR="aotriton"
setup_source_volume ${SOURCE_VOLUME} ${GIT_HTTPS_ORIGIN} ${LOCAL_DIR} ${GIT_COMMIT}

OUTPUT_DIR="$1"
CACHE_DIR="${OUTPUT_DIR}/.cache"
WHEEL_CACHE_DIR="${CACHE_DIR}/wheels"
mkdir -p "${WHEEL_CACHE_DIR}" "${CACHE_DIR}/pip"

# Determine Triton hashes to build.
# .venvs.default in SUITE_YAML replaces the embedded submodule hash;
# otherwise the submodule is the mandatory default.
DEFAULT_HASH=""
if [[ -n "${SUITE_YAML}" ]]; then
  DEFAULT_HASH=$(yq -r '.venvs.default // empty' "${SUITE_YAML}")
fi
if [[ -z "${DEFAULT_HASH}" ]]; then
  DEFAULT_HASH=$(git rev-parse HEAD:third_party/triton)
fi
TRITON_HASHES=("${DEFAULT_HASH}")
if [[ -n "${SUITE_YAML}" ]]; then
  readarray -t YAML_HASHES < <(yq -r '.venvs | to_entries | .[] | select(.key != "default") | .value' "${SUITE_YAML}")
  TRITON_HASHES+=("${YAML_HASHES[@]}")
fi

# Triton wheels are only needed for image builds (GPU kernel images embed the wheel).
# Runtime builds consume pre-built wheels from /cache/wheels via WHEEL_CFG.
if [[ ${SUITE_SELECT_IMAGE} -gt 0 ]]; then
  TRITON_WHEEL_VERSION_SUFFIX="+aotriton${aotriton_major}.${aotriton_minor}"
  bash "${SCRIPT_DIR}/build_triton_wheels.sh" \
    --wheel_output_dir "${WHEEL_CACHE_DIR}" \
    --version_suffix "${TRITON_WHEEL_VERSION_SUFFIX}" \
    "${TRITON_HASHES[@]}"
fi

# Resolve wheel configuration: yaml altwheel config, or path to the default pre-built wheel.
if [[ -n "${SUITE_YAML}" ]]; then
  cp "${SUITE_YAML}" "${CACHE_DIR}/tmpconfig.yaml"
  replace_hash \
    "${CACHE_DIR}/tmpconfig.yaml" \
    "${WHEEL_CACHE_DIR}" \
    "/cache/wheels" \
    "${TRITON_HASHES[@]}"
  WHEEL_CFG="/cache/tmpconfig.yaml"
else
  DEFAULT_SHORT="${DEFAULT_HASH:0:8}"
  WHEEL_CFG=$(ls "${WHEEL_CACHE_DIR}"/triton-*+*${DEFAULT_SHORT}*.whl 2>/dev/null | head -1)
  if [[ -z "${WHEEL_CFG}" ]]; then
    echo "Error: no pre-built triton wheel found for ${DEFAULT_SHORT} in ${WHEEL_CACHE_DIR}" >&2
    exit 1
  fi
  # Map host path to in-container path under /cache/wheels
  WHEEL_CFG="/cache/wheels/$(basename "${WHEEL_CFG}")"
fi

function build_inside() {
  rocmver="$1"
  NOIMAGE_MODE="$2"
  DOCKER_IMAGE="aotriton:buildenv-rocm${rocmver}"
  if [ -z "$(docker images -q ${DOCKER_IMAGE} 2>/dev/null)" ]; then
    # Use theRock.Dockerfile for ROCm >= 7.10
    if printf '%s\n%s\n' "7.10" "${rocmver}" | sort -V -C; then
      DOCKERFILE="theRock.Dockerfile"
    else
      DOCKERFILE="rocm.Dockerfile"
    fi
    (cd "${SCRIPT_DIR}" && docker build --network=host -t ${DOCKER_IMAGE} \
      --build-arg ROCM_VERSION_IN_URL=${rocmver} \
      -f ${DOCKERFILE} .)
  fi
  set -x
  docker run --network=host -i --rm \
    -v ${SOURCE_VOLUME}:/src:ro \
    --mount "type=bind,source=$(realpath ${OUTPUT_DIR}),target=/output" \
    --mount "type=bind,source=$(realpath ${CACHE_DIR}),target=/cache" \
    --tmpfs "/scratch:exec" \
    -e AOTRITON_BUILD_PATH=/scratch/build/aotriton \
    -e AOTRITON_INSTALL_PREFIX=/scratch/install \
    -e PIP_CACHE_DIR=/cache/pip \
    -w / \
    ${DOCKER_IMAGE} \
    bash -l -s "${NOIMAGE_MODE}" "${WHEEL_CFG}" \
    < "${SCRIPT_DIR}/runc-manylinux-build-tar.sh"
}

if [ ${SUITE_SELECT_RUNTIME} -gt 0 ]; then
  # build ROCM runtime image
  for rocmver in "${SUITE_RUNTIME_LIST[@]}"
  do
    build_inside ${rocmver} ON
  done
fi

if [ ${SUITE_SELECT_IMAGE} -gt 0 ]; then
  rocmver=7.2.3
  build_inside ${rocmver} OFF
fi
