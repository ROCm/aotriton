#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
build_dir="$(realpath -m "${AOTRITON_BUILD_DIR:-$repo_dir/build-gfx908}")"
install_dir="$build_dir/install"

# The Python binding is built against the development interpreter's Torch.
# CMake manages its own isolated compiler venv and pinned Triton submodule.
cmake_args=(
    -S "$repo_dir" -B "$build_dir" -G Ninja
    "-DPython3_EXECUTABLE=$(command -v python)"
    "-DCMAKE_INSTALL_PREFIX=$install_dir"
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
    "-DAOTRITON_TARGET_ARCH=${AOTRITON_TARGET_ARCH:-gfx908}"
    -DAOTRITON_NAME_SUFFIX=123
    -DAOTRITON_NO_PYTHON=OFF
    -DAOTRITON_USE_TORCH=ON
    -DAOTRITON_COMMON_LIBRARY_ONLY_MODE=OFF
    -DAOTRITON_NOIMAGE_MODE=OFF
    -DAOTRITON_DEBUG_SKIP_TRITON_KERNELS=OFF
    -DAOTRITON_BUILD_FOR_TUNING=OFF
    -DAOTRITON_INHERIT_SYSTEM_SITE_TRITON=OFF
    -DAOTRITON_GPU_BUILD_TIMEOUT=0
    -DAOTRITON_TERMINATE_WHEN_GPU_BUILD_TIMEOUT=ON
)
if [[ -n "${AOTRITON_FLYDSL_WHEEL:-}" ]]; then
    cmake_args+=("-DAOTRITON_USE_LOCAL_FLYDSL_WHEEL=$(realpath "$AOTRITON_FLYDSL_WHEEL")")
fi
if [[ -n "${AOTRITON_TRITON_WHEEL:-}" ]]; then
    cmake_args+=("-DAOTRITON_USE_LOCAL_TRITON_WHEEL=$(realpath "$AOTRITON_TRITON_WHEEL")")
fi

# Additional -D options can override this development profile.
cmake "${cmake_args[@]}" "$@"
# Build and install in a single invocation; retain symbols for debugging.
cmake --build "$build_dir" --target install --parallel "${MAX_JOBS:-8}"

echo "AOTriton installed in $install_dir"
printf 'export PYTHONPATH="%s${PYTHONPATH:+:$PYTHONPATH}"\n' "$install_dir/lib"
