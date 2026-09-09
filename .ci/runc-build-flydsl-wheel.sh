#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Build a FlyDSL wheel from a fresh git checkout, against a prebuilt LLVM/MLIR
# tarball. Runs inside a container -- .ci/build_flydsl_wheel.sh's, or .tune's
# worker container as its no-Docker fallback, same split (and same reason) as
# runc-build-triton-wheel.sh.
#
# Usage: runc-build-flydsl-wheel.sh <flydsl_source> <commit> <llvm_tarball> <output_dir> [<version_suffix>] [<build_dir>] [<jobs>]
#
#   <flydsl_source>  git URL, normally file:///mirror.
#   <commit>         a resolved 40-hex SHA -- it goes into the wheel's version
#                    and therefore into the cache key, so a moving ref will not do.
#   <llvm_tarball>   llvm-<sha8>-<distro>-x64.tar.gz from runc-build-llvm-tarball.sh.
#                    Extracted here and handed to FlyDSL as MLIR_PATH.
#   <output_dir>     where the .whl is written.
#   <version_suffix> appended inside the local version segment, after the git
#                    hash: <base>+git<sha8><version_suffix>, e.g. ".aotriton0.14"
#                    giving 0.3.1+git421935cc.aotriton0.14. Same idea as
#                    TRITON_WHEEL_VERSION_SUFFIX, spelled for FlyDSL's own
#                    FLYDSL_PACKAGE_VERSION_OVERRIDE.
#   <build_dir>      scratch root; the checkout and the extracted LLVM live
#                    here and are reused. Defaults to a fresh mktemp -d.
#   <jobs>           advisory only, see CMAKE_BUILD_PARALLEL_LEVEL below.
#
# The environment variable that points FlyDSL's setup.py at a prebuilt LLVM is
# MLIR_PATH -- not LLVM_SYSPATH, which is Triton's. setup.py itself only reads
# it to help auditwheel; the load-bearing consumer is scripts/build.sh, which
# setup.py shells out to, and which passes it on as
# -DMLIR_DIR="${MLIR_PATH}/lib/cmake/mlir" for FlyDSL's
# find_package(MLIR REQUIRED CONFIG). So the prefix must contain lib/cmake/mlir,
# and that is checked below.

set -ex

FLYDSL_SOURCE="$1"
COMMIT="$2"
LLVM_TARBALL="$3"
OUTPUT_DIR="$4"
VERSION_SUFFIX="${5:-}"
BUILD_DIR="${6:-}"
JOBS="${7:-}"

if [ -z "$FLYDSL_SOURCE" ] || [ -z "$COMMIT" ] || [ -z "$LLVM_TARBALL" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: $0 <flydsl_source> <commit> <llvm_tarball> <output_dir> [<version_suffix>] [<build_dir>] [<jobs>]" >&2
  exit 1
fi

if [[ ! "$COMMIT" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "Error: <commit> must be a resolved 40-hex SHA, got '${COMMIT}'." >&2
  exit 1
fi

if [ ! -f "$LLVM_TARBALL" ]; then
  echo "Error: LLVM tarball not found: ${LLVM_TARBALL}" >&2
  exit 1
fi

if [ -z "$BUILD_DIR" ]; then
  BUILD_DIR="$(mktemp -d)"
fi
if [ -z "$JOBS" ]; then
  JOBS=$(nproc)
fi

SHORT="${COMMIT:0:8}"
git config --global --add safe.directory '*'

# --- Unpack the LLVM/MLIR install prefix ---
# The tarball holds a single top-level directory named after itself (see
# runc-build-llvm-tarball.sh), so the extracted prefix is predictable without
# listing the archive. Extraction of a multi-gigabyte tree is skipped when it
# is already there: BUILD_DIR is normally a persistent volume.
LLVM_NAME="$(basename "${LLVM_TARBALL}")"
LLVM_NAME="${LLVM_NAME%.tar.gz}"
LLVM_ROOT="${BUILD_DIR}/llvm"
export MLIR_PATH="${LLVM_ROOT}/${LLVM_NAME}"
if [ ! -d "${MLIR_PATH}/lib/cmake/mlir" ]; then
  mkdir -p "${LLVM_ROOT}"
  tar -xzf "${LLVM_TARBALL}" -C "${LLVM_ROOT}"
fi
if [ ! -d "${MLIR_PATH}/lib/cmake/mlir" ]; then
  echo "Error: ${LLVM_TARBALL} did not unpack to a prefix with lib/cmake/mlir at ${MLIR_PATH}." >&2
  exit 1
fi

# --- Check out FlyDSL ---
# In-tree, on the persistent build dir: FlyDSL's setup.py refuses a
# FLY_BUILD_DIR outside its own repo root, so incremental rebuilds require the
# checkout itself to persist, not just a scratch build directory.
SRC_DIR="${BUILD_DIR}/flydsl"
if [ ! -d "${SRC_DIR}/.git" ]; then
  git init "${SRC_DIR}"
  git -C "${SRC_DIR}" remote add origin "${FLYDSL_SOURCE}"
else
  git -C "${SRC_DIR}" remote set-url origin "${FLYDSL_SOURCE}"
fi
if ! git -C "${SRC_DIR}" cat-file -e "${COMMIT}^{commit}" 2>/dev/null; then
  git -C "${SRC_DIR}" fetch --depth=1 origin "${COMMIT}"
fi
git -C "${SRC_DIR}" checkout -f "${COMMIT}"
# dlpack and tvm-ffi are submodules FlyDSL's CMake include()s headers from.
# scripts/build.sh initialises them itself, but only by probing for one header;
# do it here so the failure, if any, is a git failure rather than a compile one.
git -C "${SRC_DIR}" submodule update --init --recursive

cd "${SRC_DIR}"

# --- Tripwire: an LLVM the tarball cannot be ---
# FlyDSL gained thirdparty/llvm-rocdl-lld-argv0.patch after v0.3.1, applied by
# its scripts/build_llvm.sh to the LLVM checkout before configuring. Our
# tarball is built by runc-build-llvm-tarball.sh, which deliberately does not
# clone FlyDSL and so cannot apply it. A FlyDSL that wants a patched LLVM and
# silently gets an unpatched one is precisely the failure mode
# third_party/flydsl-llvm.txt exists to prevent, so refuse rather than warn.
if [ -f scripts/build_llvm.sh ]; then
  for p in $(grep -oE 'thirdparty/[A-Za-z0-9._/-]+\.patch' scripts/build_llvm.sh | sort -u); do
    if [ -f "${p}" ]; then
      echo "Error: FlyDSL ${SHORT} applies ${p} to LLVM before building it, but" >&2
      echo "the tarball supplied here was built without it:" >&2
      echo "  ${LLVM_TARBALL}" >&2
      echo "Teach .ci/runc-build-llvm-tarball.sh to apply this patch (and bump" >&2
      echo "the tarball name so the cache does not serve the unpatched one)," >&2
      echo "in the same commit that moves third_party/flydsl-compiler.txt." >&2
      exit 1
    fi
  done
fi

# Informational, not a check: AOTriton pins LLVM itself, in
# third_party/flydsl-llvm.txt, and for 0.14b that pin is deliberately not the
# hash FlyDSL names -- it is that hash plus the spill fix. Printing both is
# what makes a surprising divergence visible in the build log.
if [ -f thirdparty/llvm-build-info.json ]; then
  python3 -c 'import json;print("FlyDSL expects upstream LLVM:", json.load(open("thirdparty/llvm-build-info.json"))["upstream"]["llvm_hash"])' || true
fi
echo "Building against LLVM: ${LLVM_NAME}"

# --- Python build deps for the outer interpreter ---
# setup.py shells out to scripts/build.sh, which configures CMake with
# Python3_EXECUTABLE=$(which python3) and locates nanobind by importing it in
# that same interpreter. pip's build isolation gives those packages to the
# build backend, not to the interpreter CMake ends up using, so they have to be
# here too. nanobind is pinned to the version the LLVM tarball's MLIR bindings
# were built with -- they share the "mlir" nanobind domain, so the two must
# agree; bump both together or not at all.
NANOBIND_VERSION="${NANOBIND_VERSION:-2.12.0}"
python -m pip install "nanobind==${NANOBIND_VERSION}" numpy pybind11

# --- Version ---
# FlyDSL's default version for a non-release build is <base>.dev<commit count>,
# which a --depth=1 checkout renders as ".dev1" for every commit -- useless as
# a cache key. FLYDSL_PACKAGE_VERSION_OVERRIDE is FlyDSL's own supported hook
# for release automation; use it to embed the commit the way Triton's
# +git<hash8> does, so the wheel filename identifies what is in it.
BASE_VERSION=$(python3 - <<'PY'
import pathlib, re
text = pathlib.Path("python/flydsl/__init__.py").read_text(encoding="utf-8")
m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
# A base that already carries a local segment cannot take ours; drop theirs
# rather than emit a PEP 440-invalid double '+'.
print((m.group(1) if m else "0.0.0").split("+")[0])
PY
)
if [ -z "${BASE_VERSION}" ]; then
  echo "Error: could not read __version__ from python/flydsl/__init__.py." >&2
  exit 1
fi
export FLYDSL_PACKAGE_VERSION_OVERRIDE="${BASE_VERSION}+git${SHORT}${VERSION_SUFFIX}"

# --- Build ---
# FLY_BUILD_DIR is keyed by ABI tag and LLVM tarball so that a rebuild for a
# different Python, or against a different LLVM, never inherits the previous
# CMakeCache. It must stay relative to the repo root (setup.py rejects a path
# outside it), which is also why the checkout above is the persistent thing.
ABI_TAG=$(python3 -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
export FLY_BUILD_DIR="build-fly/build_${ABI_TAG}_${LLVM_NAME}"
# "auto" would reuse a stale embedded _mlir from a previous LLVM; always
# reconfigure and let CMake decide what is actually out of date. This is also
# FlyDSL's own default in scripts/build_wheels.sh.
export FLY_REBUILD=1
# Advisory. FlyDSL's scripts/build.sh -- which setup.py invokes with no
# arguments -- passes an explicit `-j$(nproc)` to `cmake --build`, and an
# explicit -j beats CMAKE_BUILD_PARALLEL_LEVEL, so this only takes effect for
# any cmake step that does not set its own. FlyDSL's build is small enough
# next to LLVM's that this has not been worth working around; <jobs> is here
# for the LLVM half, which honours it.
export CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}"

mkdir -p "${OUTPUT_DIR}"
# --no-deps: FlyDSL declares no runtime requirements today, and if it gains
# some we still want exactly one wheel in the output directory, because that
# directory is a cache the host half globs.
python -m pip wheel . --no-deps -w "${OUTPUT_DIR}"

# The cp tag is not cosmetic: AOTriton's CMake refuses a --flydsl_wheel whose
# ABI does not match the build venv, so a wheel that came out tagged for a
# different interpreter is a failure here, not three steps later.
ls "${OUTPUT_DIR}"/flydsl-*+*"${SHORT}"*"${ABI_TAG}"*.whl
