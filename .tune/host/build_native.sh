#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Build the tuning version of AOTriton for ONE arch on a build host, with no
# docker: the active (managed) venv supplies the compiler. The native
# counterpart of .tune/container/build_arch.sh, and what .tune/bin/nativebld
# runs over ssh.
#
# Usage: build_native.sh [options] <host workdir> <arch>
#   --flydsl_wheel <whl>        a local FlyDSL wheel (cp tag of the venv). Required
#                               while third_party/flydsl-llvm.txt is non-empty (the
#                               configure tripwire); the wheel is only installed
#                               into the build venv, no arch but gfx950 uses it.
#   --jobs <n>                  ninja -j; the build runs under nice. A build host
#                               shared with a measurement should not get all its
#                               cores.
#   --external_tuning_database <dir>   see .ci/common-build.sh
#
# The build goes to <host workdir>/build/<arch>, the install to
# <host workdir>/installed/<arch>, where fetchbuild/relay_build expect it. As
# .ci/build-tune.sh, the profile it mirrors, it is Release, NAME_SUFFIX 123,
# AOTRITON_BUILD_FOR_TUNING=ON.
#
# Triton is not built: the build venv is created from the base interpreter, so
# the active venv's site-packages (which hold triton) are put on PYTHONPATH and
# AOTRITON_INHERIT_SYSTEM_SITE_TRITON=ON verifies the import. The compiler is
# therefore the venv's Triton, not third_party/triton's pin: say so when a
# result is published.

set -euo pipefail

FLYDSL_WHEEL=""
JOBS=""
EXTERNAL_DB=""
POSITIONAL=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --flydsl_wheel) FLYDSL_WHEEL="$2"; shift 2 ;;
        --external_tuning_database) EXTERNAL_DB="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        -*) echo "Error: unknown option: $1" >&2; exit 1 ;;
        *) POSITIONAL+=("$1"); shift ;;
    esac
done
if [ "${#POSITIONAL[@]}" -ne 2 ]; then
    sed -n '/^# Usage:/,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
    exit 1
fi
WORKDIR="${POSITIONAL[0]}"
ARCH="${POSITIONAL[1]}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AOTRITON_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

. "$AOTRITON_ROOT/.ci/common-pin.sh"
. "$AOTRITON_ROOT/.tune/lib/find_flydsl.sh"
if flydsl_required "$AOTRITON_ROOT" && [ -z "$FLYDSL_WHEEL" ]; then
    echo "Error: third_party/flydsl-llvm.txt is non-empty, so configure needs --flydsl_wheel <whl>" >&2
    exit 1
fi

for tool in cmake ninja python; do
    command -v "$tool" >/dev/null || { echo "Error: $tool is not on PATH" >&2; exit 1; }
done
VENV_SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
python -c 'import triton, triton.language' || { echo "Error: the active venv has no triton" >&2; exit 1; }
echo "Triton: $(python -c 'import triton; print(triton.__version__)') from $VENV_SITE"
export PYTHONPATH="$VENV_SITE${PYTHONPATH:+:$PYTHONPATH}"

BUILD_DIR="$WORKDIR/build/$ARCH"
INSTALL_DIR="$WORKDIR/installed/$ARCH"
mkdir -p "$BUILD_DIR" "$INSTALL_DIR"

CMAKE_ARGS=(-DCMAKE_INSTALL_PREFIX="$INSTALL_DIR"
            -DCMAKE_BUILD_TYPE=Release
            -DAOTRITON_TARGET_ARCH="$ARCH"
            -DAOTRITON_NAME_SUFFIX=123
            -DAOTRITON_BUILD_FOR_TUNING=ON
            -DAOTRITON_INHERIT_SYSTEM_SITE_TRITON=ON)
[ -z "$FLYDSL_WHEEL" ] || CMAKE_ARGS+=(-DAOTRITON_USE_LOCAL_FLYDSL_WHEEL="$(realpath "$FLYDSL_WHEEL")")
[ -z "$EXTERNAL_DB" ] || CMAKE_ARGS+=(-DAOTRITON_EXTERNAL_TUNING_DATABASE="$(realpath "$EXTERNAL_DB")")
set -x
cd "$BUILD_DIR"
cmake "$AOTRITON_ROOT" "${CMAKE_ARGS[@]}" -G Ninja
nice -n 10 ninja ${JOBS:+-j "$JOBS"} install/strip
