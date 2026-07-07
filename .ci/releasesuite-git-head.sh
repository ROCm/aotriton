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
         -r <ROCM ver>: override the ROCm version list (does not select a
                        component; use with --runtime, --image, or both)
               --image: build GPU images.
             --runtime: build all C++ runtimes.
                --asan: build with AddressSanitizer (clang). Requires TheRock
                        clang, which ships only with ROCm >= 7.10, so every -r
                        must be a TheRock version given as a long pre-release
                        string (e.g. 7.14.0a20260624). With no -r it defaults
                        to THEROCK_ASAN_VERSION. Tarball gets a +asan suffix.
       --arch <list>: ';'-separated GPU arch list (e.g. 'gfx942;gfx950'),
                        forwarded to cmake as AOTRITON_TARGET_ARCH. Defaults
                        to ALL (every arch in the CMakeLists default list).
              --debug: Debug mode: require exactly one runtime version (-r),
                        and leave the build container running after the build
                        so the contents can be inspected interactively.
         --yaml <.yml>: Use yml config file to build the release
        --origin <url>: Override the git HTTPS origin URL. Pass "auto" to use the
                        tracked remote URL (ssh/git URLs are rewritten to https).
                        (default: auto-detected from remote)
 --triton_origin <url>: Override the Triton git origin for wheel builds.
                        Accepts a fork URL or a local checkout via file:///abs/path
                        (default: https://github.com/ROCm/triton)
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

TEMP=$(getopt -o hr: --longoptions image,runtime,asan,debug,arch:,yaml:,origin:,triton_origin: -- "$@")

if [ $? -ne 0 ]; then
  echo "Error: Invalid option." >&2
  help 1
fi

eval set -- "$TEMP"

SUITE_SELECT_IMAGE=-1
SUITE_SELECT_RUNTIME=-1
# TheRock runtimes are pre-release/nightlies and must use the long version
# string (e.g. 7.15.0a20260707). The last entry is also the default GPU image
# ROCm (IMAGE_ROCMVER), so keep a gfx1250-capable TheRock build last.
SUITE_RUNTIME_LIST=(6.4.4 7.0.3 7.1.1 7.2.4 7.14.0a20260624 7.15.0a20260707)
CMDLIST=()
SUITE_DEFAULT_SELECTION=1
SUITE_YAML=""
SUITE_ORIGIN=""
SUITE_TRITON_ORIGIN=""
SUITE_DEBUG=0
SUITE_ASAN=0
SUITE_ARCH="ALL"
# ASAN support is pinned to a single TheRock 7.14 pre-release.
# Pass -r 7.14.0 as a symbolic selector; the build always uses this version.
THEROCK_ASAN_VERSION="7.14.0a20260624"

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
    --asan)
      SUITE_ASAN=1
      ;;
    --arch)
      shift
      SUITE_ARCH="$1"
      ;;
    --debug)
      SUITE_DEBUG=1
      ;;
    -r)
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

if [ ${SUITE_SELECT_IMAGE} -lt 0 ]; then
  SUITE_SELECT_IMAGE=${SUITE_DEFAULT_SELECTION}
fi

if [ ${SUITE_SELECT_RUNTIME} -lt 0 ]; then
  SUITE_SELECT_RUNTIME=${SUITE_DEFAULT_SELECTION}
fi

if [ ${SUITE_DEBUG} -gt 0 ] && [ ${#CMDLIST[@]} -ne 1 ]; then
  echo "Error: --debug requires exactly one runtime version specified via -r <ROCM ver>." >&2
  help 1
fi

# --asan: AddressSanitizer needs TheRock's clang, which ships only with ROCm
# >= 7.10 (the theRock.Dockerfile path). Every -r must therefore be a TheRock
# (pre-release) version, given as a long nightly string. With no -r, default
# to THEROCK_ASAN_VERSION.
if [ ${SUITE_ASAN} -gt 0 ]; then
  if [[ ${#CMDLIST[@]} -gt 0 ]]; then
    for _ver in "${SUITE_RUNTIME_LIST[@]}"; do
      if ! printf '%s\n%s\n' "7.10" "${_ver}" | sort -V -C; then
        echo "Error: --asan requires TheRock ROCm >= 7.10 (e.g. ${THEROCK_ASAN_VERSION}); got -r ${_ver}." >&2
        exit 1
      fi
    done
  else
    SUITE_RUNTIME_LIST=("${THEROCK_ASAN_VERSION}")
  fi
  IMAGE_ROCMVER="${SUITE_RUNTIME_LIST[-1]}"
  ASAN_MODE="ON"
else
  # Build the GPU image with the last ROCm version in the runtime list.
  IMAGE_ROCMVER="${SUITE_RUNTIME_LIST[-1]}"
  ASAN_MODE="OFF"
fi

echo "SUITE_RUNTIME_LIST ${SUITE_RUNTIME_LIST[@]}"

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
. "${SCRIPT_DIR}/common-vars.sh"
. "${SCRIPT_DIR}/common-setup-volume.sh"
. "${SCRIPT_DIR}/common-git-https-origin.sh"
# --origin auto: keep the GIT_HTTPS_ORIGIN auto-derived from the tracked
# remote by common-git-https-origin.sh (ssh/git URLs already rewritten to
# https). Any other non-empty value overrides it explicitly.
if [[ -n "${SUITE_ORIGIN}" && "${SUITE_ORIGIN}" != "auto" ]]; then
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
# Runtime-only ASAN builds skip wheel resolution entirely (WHEEL_CFG=NONE).
WHEEL_CFG="NONE"
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

  # Resolve wheel configuration for image builds.
  if [[ -n "${SUITE_YAML}" ]]; then
    cp "${SUITE_YAML}" "${CACHE_DIR}/tmpconfig.yaml"
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
    WHEEL_CFG="/cache/wheels/$(basename "${WHEEL_CFG}")"
  fi
fi

function build_inside() {
  rocmver="$1"
  NOIMAGE_MODE="$2"
  ASAN_MODE="${3:-OFF}"
  ARCH_LIST="${4:-ALL}"
  DOCKER_IMAGE="aotriton:buildenv-rocm${rocmver}"
  if [ -z "$(docker images -q ${DOCKER_IMAGE} 2>/dev/null)" ]; then
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
  EXTRA_ENV=()
  if [[ "${ASAN_MODE}" == "ON" ]]; then
    EXTRA_ENV+=(-e "TRITON_ENABLE_ASAN=1")
  fi
  TTY_FLAGS=()
  [ ${SUITE_DEBUG} -gt 0 ] && TTY_FLAGS=(-t -e SUITE_DEBUG=1)
  set -x
  docker run --network=host -i --rm \
    -v ${SOURCE_VOLUME}:/src:ro \
    --mount "type=bind,source=$(realpath ${SCRIPT_DIR}/runc-manylinux-build-tar.sh),target=/tmp/runc-manylinux-build-tar.sh,readonly" \
    --mount "type=bind,source=$(realpath ${OUTPUT_DIR}),target=/output" \
    --mount "type=bind,source=$(realpath ${CACHE_DIR}),target=/cache" \
    --tmpfs "/scratch:exec" \
    -e AOTRITON_BUILD_PATH=/scratch/build/aotriton \
    -e AOTRITON_INSTALL_PREFIX=/scratch/install \
    -e PIP_CACHE_DIR=/cache/pip \
    "${EXTRA_ENV[@]}" \
    "${TTY_FLAGS[@]}" \
    -w / \
    ${DOCKER_IMAGE} \
    bash -l /tmp/runc-manylinux-build-tar.sh \
    "${NOIMAGE_MODE}" "${WHEEL_CFG}" "${ASAN_MODE}" "${ARCH_LIST}"
}

if [ ${SUITE_SELECT_RUNTIME} -gt 0 ]; then
  for rocmver in "${SUITE_RUNTIME_LIST[@]}"; do
    build_inside "${rocmver}" ON "${ASAN_MODE}" "${SUITE_ARCH}"
  done
fi

if [ ${SUITE_SELECT_IMAGE} -gt 0 ]; then
  build_inside "${IMAGE_ROCMVER}" OFF "${ASAN_MODE}" "${SUITE_ARCH}"
fi
