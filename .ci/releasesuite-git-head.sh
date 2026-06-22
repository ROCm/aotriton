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
       --arch <list>: ';'-separated GPU arch list (e.g. 'gfx942;gfx950'),
                        forwarded to cmake as AOTRITON_TARGET_ARCH. Defaults
                        to ALL (every arch in the CMakeLists default list).
                        Useful to shorten debug builds.
              --debug: Debug mode: require exactly one runtime version (-r),
                        and leave the build container running after the build
                        so the contents can be inspected interactively.
         --yaml <.yml>: Use yml config file to build the release
        --origin <url>: Override the git HTTPS origin URL (default: auto-detected from remote)
 --triton_origin <url>: Override the Triton git origin for wheel builds.
                        Accepts a fork URL or a local checkout via file:///abs/path
                        (default: https://github.com/ROCm/triton)
Either --image, --runtime, or both must be specified explicitly.

The YAML configuration file follows the format shown in docs/AltWheelExample.yaml.
However it accepts GIT SHA1 for Triton wheels instead.
The build process will
1. Build Triton wheels from the SHA1
2. Replace SHA1 with actual wheel path and use the replaced yaml file to build
   AOTriton
EOF
  exit $1
}

TEMP=$(getopt -o hr: --longoptions image,runtime,debug,arch:,yaml:,origin:,triton_origin: -- "$@")

if [ $? -ne 0 ]; then
  echo "Error: Invalid option." >&2
  help 1
fi

eval set -- "$TEMP"

SUITE_SELECT_IMAGE=0
SUITE_SELECT_RUNTIME=0
SUITE_RUNTIME_LIST=(6.4.4 7.0.3 7.1.1 7.2.3)
CMDLIST=()
SUITE_YAML=""
SUITE_ORIGIN=""
SUITE_TRITON_ORIGIN=""
SUITE_ARCH="ALL"
SUITE_DEBUG=0

while true; do
  case "$1" in
    -h)
      help 0
      ;;
    --image)
      SUITE_SELECT_IMAGE=1
      ;;
    --runtime)
      SUITE_SELECT_RUNTIME=1
      ;;
    --debug)
      SUITE_DEBUG=1
      ;;
    --arch)
      shift
      SUITE_ARCH="$1"
      ;;
    -r)
      shift
      CMDLIST+=("$1")
      # -r specifies the ROCm version list only; component selection (runtime
      # vs image) is governed solely by --runtime / --image.  When neither is
      # given, both components build (default behaviour unchanged).
      ;;
    --yaml)
      shift
      SUITE_YAML="$1"
      ;;
    --origin)
      shift
      SUITE_ORIGIN="$1"
      ;;
    --triton_origin)
      shift
      SUITE_TRITON_ORIGIN="$1"
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

if [ ${SUITE_SELECT_IMAGE} -eq 0 ] && [ ${SUITE_SELECT_RUNTIME} -eq 0 ]; then
  echo "Error: specify --runtime, --image, or both." >&2
  help 1
fi

if [ ${SUITE_DEBUG} -gt 0 ] && [ ${#SUITE_RUNTIME_LIST[@]} -ne 1 ]; then
  echo "Error: --debug requires exactly one runtime version via -r <ROCM ver>." >&2
  help 1
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
mkdir -p "${WHEEL_CACHE_DIR}"

# Determine Triton hashes to build.
# .venvs.default in SUITE_YAML replaces the embedded submodule hash;
# otherwise the submodule is the mandatory default.
DEFAULT_HASH=""
if [[ -n "${SUITE_YAML}" ]]; then
  DEFAULT_HASH=$(yq -r '.venvs.default // ""' "${SUITE_YAML}")
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
  TRITON_ORIGIN_ENV=()
  if [[ -n "${SUITE_TRITON_ORIGIN}" ]]; then
    TRITON_ORIGIN_ENV=(TRITON_GIT_ORIGIN="${SUITE_TRITON_ORIGIN}")
  fi
  env "${TRITON_ORIGIN_ENV[@]}" bash "${SCRIPT_DIR}/build_triton_wheels.sh" \
    --wheel_output_dir "${WHEEL_CACHE_DIR}" \
    --version_suffix "${TRITON_WHEEL_VERSION_SUFFIX}" \
    "${TRITON_HASHES[@]}"
fi

# Resolve wheel configuration: yaml altwheel config, or path to the default pre-built wheel.
# Only image builds embed a Triton wheel. Runtime builds run with
# AOTRITON_NOIMAGE_MODE=ON and never touch Triton, so skip wheel resolution
# entirely (and avoid failing when no wheel was pre-built).
WHEEL_CFG="NONE"
if [[ ${SUITE_SELECT_IMAGE} -gt 0 ]]; then
  if [[ -n "${SUITE_YAML}" ]]; then
    cp "${SUITE_YAML}" "${CACHE_DIR}/tmpconfig.yaml"
    # The user yaml may omit venvs.default. Arches unmatched by any rule fall
    # back to 'default' in CMakeLists.txt; without it the default venv builds
    # Triton from the read-only third_party/triton source and fails in CI.
    # Inject the resolved DEFAULT_HASH (whose wheel is already built and part
    # of TRITON_HASHES) so replace_hash maps it to the pre-built wheel path.
    if [[ -z "$(yq -r '.venvs.default // ""' "${CACHE_DIR}/tmpconfig.yaml")" ]]; then
      yq -i ".venvs.default = \"${DEFAULT_HASH}\"" "${CACHE_DIR}/tmpconfig.yaml"
    fi
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
fi

function build_inside() {
  rocmver="$1"
  NOIMAGE_MODE="$2"
  ARCH_LIST="${3:-ALL}"
  DOCKER_IMAGE="aotriton:buildenv-rocm${rocmver}"
  if [ -z "$(docker images -q ${DOCKER_IMAGE} 2>/dev/null)" ]; then
    # Use theRock.Dockerfile for ROCm >= 7.10
    if printf '%s\n%s\n' "7.10" "${rocmver}" | sort -V -C; then
      DOCKERFILE="theRock.Dockerfile"
      BUILD_ARG=(--build-arg "THEROCK_VERSION=${rocmver}")
    else
      DOCKERFILE="rocm.Dockerfile"
      BUILD_ARG=(--build-arg "ROCM_VERSION_IN_URL=${rocmver}")
    fi
    (cd "${SCRIPT_DIR}" && docker build --network=host -t ${DOCKER_IMAGE} \
      "${BUILD_ARG[@]}" \
      -f ${DOCKERFILE} .)
  fi
  # Cache pip downloads under <output>/.cache/pip on the host.
  mkdir -p "${CACHE_DIR}/pip"
  # In debug mode allocate a tty so runc-manylinux-build-tar.sh can drop into
  # an interactive shell after the build. Container is always removed on exit.
  DEBUG_FLAGS=()
  if [ ${SUITE_DEBUG} -gt 0 ]; then
    DEBUG_FLAGS=(-t -e SUITE_DEBUG=1)
  fi
  set -x
  # Always bind-mount the build script from the host so the working-tree
  # version is used without requiring a commit. In debug mode also allocate a
  # TTY so the interactive shell at the end of the script works; the script is
  # run by path (not piped via stdin) so stdin stays attached to the terminal.
  TTY_FLAGS=()
  [ ${SUITE_DEBUG} -gt 0 ] && TTY_FLAGS=(-t -e SUITE_DEBUG=1)
  docker run --network=host -i --rm \
    -v ${SOURCE_VOLUME}:/src:ro \
    -v "$(realpath ${SCRIPT_DIR}/runc-manylinux-build-tar.sh)":/tmp/runc-manylinux-build-tar.sh:ro \
    --mount "type=bind,source=$(realpath ${OUTPUT_DIR}),target=/output" \
    --mount "type=bind,source=$(realpath ${CACHE_DIR}),target=/cache" \
    --tmpfs "/scratch:exec" \
    -e AOTRITON_BUILD_PATH=/scratch/build/aotriton \
    -e AOTRITON_INSTALL_PREFIX=/scratch/install \
    -e PIP_CACHE_DIR=/cache/pip \
    "${TTY_FLAGS[@]}" \
    -w / \
    ${DOCKER_IMAGE} \
    bash -l /tmp/runc-manylinux-build-tar.sh \
    "${NOIMAGE_MODE}" "${WHEEL_CFG}" "${ARCH_LIST}"
}

if [ ${SUITE_SELECT_RUNTIME} -gt 0 ]; then
  # build ROCM runtime image
  for rocmver in "${SUITE_RUNTIME_LIST[@]}"
  do
    build_inside ${rocmver} ON "${SUITE_ARCH}"
  done
fi

if [ ${SUITE_SELECT_IMAGE} -gt 0 ]; then
  # Build the GPU image with the last ROCm version in the runtime list.
  # Newer archs (e.g. gfx1250) require a matching newer ROCm (e.g. 7.14.0)
  # that older versions like 7.2.3 cannot compile. With `-r 7.14.0` the
  # runtime list becomes (7.14.0); without -r it defaults to the embedded
  # list whose last entry (7.2.3) preserves prior behavior.
  rocmver="${SUITE_RUNTIME_LIST[-1]}"
  build_inside ${rocmver} OFF "${SUITE_ARCH}"
fi
