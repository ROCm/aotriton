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
#   <llvm_tarball>   llvm-<sha12>-<distro>-x64.tar.gz from runc-build-llvm-tarball.sh.
#                    Extracted here and handed to FlyDSL as MLIR_PATH.
#   <output_dir>     where the .whl is written.
#   <version_suffix> appended inside the local version segment, after the git
#                    hash: <base>+git<sha8><version_suffix>.p<patches>, e.g.
#                    ".aotriton0.14" giving 0.3.1+git421935cc.aotriton0.14.p2.
#                    Same idea as TRITON_WHEEL_VERSION_SUFFIX, spelled for
#                    FlyDSL's own FLYDSL_PACKAGE_VERSION_OVERRIDE. .p<patches>
#                    is the number of patches applied below, always present.
#   <build_dir>      scratch root for this run; the checkout and the extracted
#                    LLVM live here. Defaults to a fresh mktemp -d.
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
# listing the archive. Extracted unconditionally: BUILD_DIR is per-run scratch,
# so "is it already there" can only ever answer no, and a check that can only
# answer one way hides the case where it answers the other.
LLVM_NAME="$(basename "${LLVM_TARBALL}")"
LLVM_NAME="${LLVM_NAME%.tar.gz}"
LLVM_ROOT="${BUILD_DIR}/llvm"
export MLIR_PATH="${LLVM_ROOT}/${LLVM_NAME}"
mkdir -p "${LLVM_ROOT}"
tar -xzf "${LLVM_TARBALL}" -C "${LLVM_ROOT}"
if [ ! -d "${MLIR_PATH}/lib/cmake/mlir" ]; then
  echo "Error: ${LLVM_TARBALL} did not unpack to a prefix with lib/cmake/mlir at ${MLIR_PATH}." >&2
  exit 1
fi

# --- Check out FlyDSL ---
# Under BUILD_DIR, because FlyDSL's setup.py refuses a FLY_BUILD_DIR outside its
# own repo root, so the build tree has to live inside the checkout.
SRC_DIR="${BUILD_DIR}/flydsl"
git init "${SRC_DIR}"
git -C "${SRC_DIR}" remote add origin "${FLYDSL_SOURCE}"
git -C "${SRC_DIR}" fetch --depth=1 origin "${COMMIT}"
git -C "${SRC_DIR}" checkout -f "${COMMIT}"

# --- Version ---
# FlyDSL's default version for a non-release build is <base>.dev<commit count>,
# which a --depth=1 checkout renders as ".dev1" for every commit -- useless as
# a cache key. FLYDSL_PACKAGE_VERSION_OVERRIDE is FlyDSL's own supported hook
# for release automation; use it to embed the commit the way Triton's
# +git<hash8> does, so the wheel filename identifies what is in it.
#
# Read before patching: the base version is also what selects the patch set
# below.
BASE_VERSION=$(python3 - "${SRC_DIR}/python/flydsl/__init__.py" <<'PY'
import pathlib, re, sys
text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
# Nothing printed when the declaration is missing or has changed shape: the
# caller turns an empty value into an error. A stand-in like "0.0.0" would
# publish a wheel whose version names neither what is in it nor the fact that
# the version could not be read.
if m:
    # A base that already carries a local segment cannot take ours; drop theirs
    # rather than emit a PEP 440-invalid double '+'.
    print(m.group(1).split("+")[0])
PY
)
if [ -z "${BASE_VERSION}" ]; then
  echo "Error: could not read __version__ from python/flydsl/__init__.py" >&2
  echo "in FlyDSL ${SHORT}. The declaration is gone or has changed shape." >&2
  exit 1
fi

# --- Patch: per FlyDSL version ---
# AOTRITON_FLYDSL_PATCH_DIR is mounted by the host half and holds one
# subdirectory per FlyDSL base version (e.g. 0.3.1/), matched exactly against
# BASE_VERSION above. The version is read here, from the checkout, because the
# host half only has a SHA and cannot tell which release line it belongs to.
# A version without a subdirectory needs no patches: 0.4.2 already finds HIP
# through ROCM_PATH / rocm-sdk and bounds nanobind below 3 itself. A run
# without AOTRITON_FLYDSL_PATCH_DIR (a caller invoking this script directly
# against a classical /opt/rocm) skips patching altogether.
#
# `git apply` and NOT `|| true`: a patch that stopped applying means FlyDSL
# moved the code it edits, and silently building without it gives a
# configure-time "hip not found" three minutes later, or worse a wheel linked
# against the wrong ROCm.
FLYDSL_PATCH_DIR=""
PATCH_COUNT=0
if [ -n "${AOTRITON_FLYDSL_PATCH_DIR:-}" ]; then
  FLYDSL_PATCH_DIR="${AOTRITON_FLYDSL_PATCH_DIR}/${BASE_VERSION}"
  if [ ! -d "${FLYDSL_PATCH_DIR}" ]; then
    echo "No AOTriton patches for FlyDSL ${BASE_VERSION}."
    FLYDSL_PATCH_DIR=""
  fi
fi
if [ -n "${FLYDSL_PATCH_DIR}" ]; then
  for patch in "${FLYDSL_PATCH_DIR}"/*.patch; do
    [ -e "${patch}" ] || continue
    echo "Applying $(basename "${patch}")"
    if ! git -C "${SRC_DIR}" apply "${patch}"; then
      echo "Error: $(basename "${patch}") does not apply to FlyDSL ${SHORT}." >&2
      echo "Rebase it onto that commit, or drop it if FlyDSL has taken the change." >&2
      exit 1
    fi
    PATCH_COUNT=$((PATCH_COUNT + 1))
  done
fi
# .p<count> is the last component of the local version, and the wheel cache's
# readers (.ci/common-flydsl.sh) check it against .ci/flydsl-patch/<version>/.
export FLYDSL_PACKAGE_VERSION_OVERRIDE="${BASE_VERSION}+git${SHORT}${VERSION_SUFFIX}.p${PATCH_COUNT}"

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
# The wheel is built with --no-build-isolation (see `pip wheel` below), so
# this interpreter is the build environment: everything FlyDSL's
# pyproject.toml [build-system] requires is installed here -- setuptools>=64,
# wheel, pybind11, nanobind, numpy, cmake -- and nothing else installs it.
# scripts/build.sh also configures CMake with Python3_EXECUTABLE=$(which
# python3) and locates nanobind by importing it in that same interpreter.
# nanobind is pinned to the version the LLVM tarball's MLIR bindings were
# built with -- they share the "mlir" nanobind domain, so the two must agree;
# bump both together or not at all.
#
# patchelf is a BINARY the build shells out to by name -- FlyDSL's CMake sets
# the RPATH of the MLIR runtime libraries it copies into the wheel -- so it has
# to be on PATH, not importable. The PyPI package ships the executable, which
# is how auditwheel gets it too, and keeps the base image free of an EPEL
# dependency for one tool.
#
# auditwheel is not optional either, though FlyDSL's setup.py treats it as
# such: its bdist_wheel runs `auditwheel repair` only if the executable is on
# PATH, and otherwise prints a warning and ships the raw linux_x86_64 wheel.
# That wheel is broken. MLIR's libmlir_c_runner_utils.so, which FlyDSL copies
# into flydsl/_mlir/_mlir_libs, carries a DT_NEEDED on
# libmlir_apfloat_wrappers.so.<ver> that nothing else puts in the wheel;
# repair is what grafts it into flydsl.libs/, as in FlyDSL's own releases.
NANOBIND_VERSION="${NANOBIND_VERSION:-2.12.0}"
python -m pip install "setuptools>=64" wheel cmake \
  "nanobind==${NANOBIND_VERSION}" numpy pybind11 patchelf auditwheel
command -v patchelf >/dev/null || { echo "Error: patchelf is not on PATH after install." >&2; exit 1; }
command -v auditwheel >/dev/null || { echo "Error: auditwheel is not on PATH after install." >&2; exit 1; }
# rsync publishes the finished wheel (end of this script). Unlike the two above
# it cannot come from PyPI, so it has to be in the image -- base.Dockerfile
# installs it -- and a missing one should fail here, not after the build.
command -v rsync >/dev/null || { echo "Error: rsync is not on PATH; it is needed to publish the wheel." >&2; exit 1; }

# --- Build ---
# FLY_BUILD_DIR must stay relative to the repo root: setup.py rejects a path
# outside it. Named by ABI tag and LLVM so the directory says what configured
# it, which a build log needs even when nothing is reused.
ABI_TAG=$(python3 -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
export FLY_BUILD_DIR="build-fly/build_${ABI_TAG}_${LLVM_NAME}"
# "auto" would reuse a stale embedded _mlir from a previous LLVM; always
# reconfigure and let CMake decide what is actually out of date. This is also
# FlyDSL's own default in scripts/build_wheels.sh.
export FLY_REBUILD=1
# Advisory. FlyDSL's scripts/build.sh -- which setup.py invokes with no
# arguments -- passes an explicit `-j$(nproc)` to `cmake --build`, and an
# explicit -j beats CMAKE_BUILD_PARALLEL_LEVEL, so this only takes effect for
# a cmake step that does not set its own.
export CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}"

# Built into scratch, NOT into OUTPUT_DIR: OUTPUT_DIR is the host's wheel
# cache, and anything that lands there is a hit for every later run. A wheel
# that fails the checks below must never reach it.
#
# --no-deps: FlyDSL declares no runtime requirements today, and if it gains
# some we still want exactly one wheel to publish.
#
# --no-build-isolation, as FlyDSL's own scripts/build_wheels.sh effectively
# does by running `setup.py bdist_wheel` directly. setup.py runs `auditwheel
# repair` as a child process, and pip's build isolation puts a sitecustomize
# on PYTHONPATH that drops this venv's site-packages from every Python child:
# /opt/therock/bin/auditwheel is found on PATH and then cannot import
# auditwheel. setup.py catches that and keeps the unrepaired wheel.
#
# -v because setup.py only warns when repair fails, and pip hides a
# successful build's output; without it the manylinux check below could say
# that repair failed but never why.
DIST_DIR="${BUILD_DIR}/dist"
python -m pip wheel . --no-deps --no-build-isolation -v -w "${DIST_DIR}"

# The cp tag is not cosmetic: AOTriton's CMake refuses a --flydsl_wheel whose
# ABI does not match the build venv, so a wheel that came out tagged for a
# different interpreter is a failure here, not three steps later.
#
# Neither is the manylinux tag. setup.py swallows a failed `auditwheel repair`
# and leaves the unrepaired wheel in place, so the platform tag is the only
# sign that repair ran. Same check as FlyDSL's scripts/build_wheels.sh.
WHEEL=$(ls "${DIST_DIR}"/flydsl-*+*"${SHORT}"*-"${ABI_TAG}"-"${ABI_TAG}"-manylinux_*.whl 2>/dev/null | head -n1)
if [ -z "${WHEEL}" ]; then
  echo "Error: no manylinux FlyDSL wheel in ${DIST_DIR}; auditwheel repair did not run or failed." >&2
  echo "An unrepaired wheel lacks libmlir_apfloat_wrappers and fails to load." >&2
  ls -1 "${DIST_DIR}" >&2 || true
  exit 1
fi

# rsync for atomic copy: more concise and reliable than cp+mv ourselves.
mkdir -p "${OUTPUT_DIR}"
rsync "${WHEEL}" "${OUTPUT_DIR}/"
ls "${OUTPUT_DIR}/$(basename "${WHEEL}")"
