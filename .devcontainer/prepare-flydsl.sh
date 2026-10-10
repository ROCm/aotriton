#!/usr/bin/env bash
set -euo pipefail

# Explicit, potentially long compiler preparation. Reuse the native halves of
# the repository's CI scripts; no nested Docker/Podman daemon is required.
# Build logs go to stderr; stdout contains only the resulting wheel path.
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
# shellcheck source=.ci/common-pin.sh
source "$repo_dir/.ci/common-pin.sh"

llvm_pin="$(parse_llvm_pin third_party/flydsl-llvm.txt)"
flydsl_pin="$(parse_flydsl_pin third_party/flydsl-compiler.txt)"
if [[ -z "$llvm_pin" ]]; then
    echo "No LLVM override is pinned. Let CMake install the pinned FlyDSL compiler instead." >&2
    exit 1
fi
mapfile -t llvm_parts <<< "$llvm_pin"
mapfile -t flydsl_parts <<< "$flydsl_pin"
llvm_origin="${llvm_parts[0]#git+}"
llvm_sha="${llvm_parts[1]}"
flydsl_origin="${flydsl_parts[0]#git+}"
flydsl_sha="${flydsl_parts[1]}"
if [[ ! "$llvm_sha" =~ ^[0-9a-fA-F]{40}$ || ! "$flydsl_sha" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "Compiler preparation requires exact commit SHAs in both FlyDSL pin files." >&2
    exit 1
fi

compiler_dir="$(realpath -m "${AOTRITON_COMPILER_DIR:-$repo_dir/build-compilers}")"
abi_tag="$(python -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
export LLVM_TARBALL_DISTRO="${LLVM_TARBALL_DISTRO:-ubuntu}"
export AOTRITON_FLYDSL_PATCH_DIR="$repo_dir/.ci/flydsl-patch"
shopt -s nullglob
patches=("$AOTRITON_FLYDSL_PATCH_DIR"/*.patch)
patch_hash="$( { if (( ${#patches[@]} )); then sha256sum "${patches[@]}"; fi; } | sha256sum)"
wheel_dir="$compiler_dir/flydsl-${flydsl_sha:0:12}-llvm${llvm_sha:0:12}-${LLVM_TARBALL_DISTRO}-${abi_tag}-${patch_hash:0:12}"
wheels=("$wheel_dir"/flydsl-*.whl)
if (( ${#wheels[@]} == 1 )); then
    printf '%s\n' "${wheels[0]}"
    exit 0
fi
if (( ${#wheels[@]} > 1 )); then
    echo "Multiple cached FlyDSL wheels in $wheel_dir; select one with AOTRITON_FLYDSL_WHEEL." >&2
    exit 1
fi

mkdir -p "$compiler_dir/llvm" "$wheel_dir"
llvm_tarball="$compiler_dir/llvm/llvm-${llvm_sha:0:12}-${LLVM_TARBALL_DISTRO}-x64.tar.gz"
if [[ ! -f "$llvm_tarball" ]]; then
    llvm_build_dir="$(mktemp -d "$compiler_dir/llvm-build.XXXXXX")"
    bash .ci/runc-build-llvm-tarball.sh "$llvm_origin" "$llvm_sha" \
        "$compiler_dir/llvm" "$llvm_build_dir" "${AOTRITON_COMPILER_JOBS:-2}" >&2
fi
flydsl_build_dir="$(mktemp -d "$compiler_dir/flydsl-build.XXXXXX")"
bash .ci/runc-build-flydsl-wheel.sh "$flydsl_origin" "$flydsl_sha" \
    "$llvm_tarball" "$wheel_dir" ".llvm${llvm_sha:0:12}" \
    "$flydsl_build_dir" "${AOTRITON_COMPILER_JOBS:-2}" >&2

wheels=("$wheel_dir"/flydsl-*.whl)
if (( ${#wheels[@]} != 1 )); then
    echo "Expected one FlyDSL wheel in $wheel_dir, found ${#wheels[@]}." >&2
    exit 1
fi
printf '%s\n' "${wheels[0]}"
