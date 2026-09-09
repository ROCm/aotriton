#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Build the LLVM/MLIR install tarball that FlyDSL's setup.py links against,
# from a fresh git checkout. Runs inside a container -- .ci/build_llvm_tarball.sh's
# ephemeral one, or .tune's worker container as its no-Docker fallback. Keeping
# the in-container half a separate file is the whole reason
# runc-build-triton-wheel.sh exists as its own file; this mirrors it.
#
# Usage: runc-build-llvm-tarball.sh <llvm_source> <commit> <output_dir> [<build_dir>] [<jobs>]
#
#   <llvm_source>  git URL to fetch from. Normally file:///mirror -- the bare
#                  mirror volume the host half syncs, so this fetch is local.
#   <commit>       a resolved 40-hex SHA. NOT a branch name: the tarball is
#                  named after this value and the name is a cache key, so the
#                  host half rev-parses the pin before calling us (see D7 in
#                  the FlyDSL integration plan, and build_llvm_tarball.sh).
#   <output_dir>   where llvm-<sha8>-<distro>-x64.tar.gz is written.
#   <build_dir>    scratch root; the checkout and build tree live here and are
#                  reused across runs. Defaults to a fresh mktemp -d, which
#                  throws away an hour of work on every invocation -- pass a
#                  persistent path (a docker volume) in anger.
#   <jobs>         parallel build jobs. Default nproc/2, see below.
#
# This does NOT read third_party/flydsl-llvm.txt. It is handed an origin and a
# SHA and builds exactly those; deciding what to build is the host half's job.

set -ex

LLVM_SOURCE="$1"
COMMIT="$2"
OUTPUT_DIR="$3"
BUILD_DIR="${4:-}"
JOBS="${5:-}"

if [ -z "$LLVM_SOURCE" ] || [ -z "$COMMIT" ] || [ -z "$OUTPUT_DIR" ]; then
  echo "Usage: $0 <llvm_source> <commit> <output_dir> [<build_dir>] [<jobs>]" >&2
  exit 1
fi

# A branch or tag here would silently produce a tarball whose name does not
# identify its contents, and the cache would then serve a stale build forever.
if [[ ! "$COMMIT" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "Error: <commit> must be a resolved 40-hex SHA, got '${COMMIT}'." >&2
  echo "The tarball name is the cache key; a moving ref cannot be one." >&2
  exit 1
fi

if [ -z "$BUILD_DIR" ]; then
  BUILD_DIR="$(mktemp -d)"
fi

# An LLVM build with assertions needs ~5 GB of RAM per link job at peak, so the
# usual $(nproc) oversubscribes memory and the linker gets OOM-killed near the
# end of an hour-long build. FlyDSL's own scripts/build_llvm.sh defaults to
# nproc/2 for this reason; keep the same default rather than discovering it
# again the hard way.
if [ -z "$JOBS" ]; then
  JOBS=$(( $(nproc) / 2 ))
  [ "$JOBS" -lt 1 ] && JOBS=1
fi

# Distro slug in the tarball name. "almalinux" for the aotriton:base-py* image
# family (AlmaLinux 8, glibc 2.28 -- the manylinux_2_28 baseline every AOTriton
# artifact targets), matching the llvm-${HASH}-almalinux-x64.tar.gz filenames
# .ci/triton-patch/docker-script-build.sh already downloads and unpacks. One
# naming scheme, so a tarball built here drops into $HOME/.triton/llvm unchanged.
DISTRO="${LLVM_TARBALL_DISTRO:-}"
if [ -z "$DISTRO" ]; then
  # `|| true`, because this script runs under `set -e` and the `.` fails when
  # /etc/os-release is absent -- which is exactly the case the next line's
  # fallback exists for. Without it the assignment's non-zero status kills the
  # script and the fallback is unreachable in its own scenario.
  DISTRO=$(. /etc/os-release 2>/dev/null && echo "${ID}") || true
  DISTRO="${DISTRO:-almalinux}"
fi

SHORT="${COMMIT:0:8}"
# The install prefix is named after the tarball, not "mlir_install" (FlyDSL's
# own name for it), and the tarball holds that one directory at top level --
# same shape as Triton's llvm-<hash>-almalinux-x64.tar.gz. FlyDSL only ever
# sees the extracted path through MLIR_PATH, so the directory name is free;
# a second naming scheme would not be. runc-build-flydsl-wheel.sh knows to
# point MLIR_PATH at <extracted>/${TARBALL_NAME}.
TARBALL_NAME="llvm-${SHORT}-${DISTRO}-x64"
TARBALL="${OUTPUT_DIR}/${TARBALL_NAME}.tar.gz"

SRC_DIR="${BUILD_DIR}/llvm-project"
# Key the build and install trees by SHA so a second commit never inherits the
# first one's CMakeCache. Nothing is ever deleted to make room: a stale tree
# for some other SHA is a cache, and .ci/CLAUDE.md's rule about not wiping
# caches applies to build trees for the same reason it applies to mirrors.
CMAKE_BUILD_DIR="${BUILD_DIR}/build-${SHORT}"
INSTALL_ROOT="${BUILD_DIR}/install-${SHORT}"
INSTALL_DIR="${INSTALL_ROOT}/${TARBALL_NAME}"

git config --global --add safe.directory '*'

# --- Fetch ---
# Plain shallow fetch of the SHA. Do NOT add --filter=blob:none: a blob-filtered
# fetch of an arbitrary SHA makes the server build an uncached pack, and the
# checkout that follows then lazily re-fetches every file in the tree one batch
# at a time. FlyDSL measured this on their CI -- `--depth 1` alone downloads
# llvm-project in ~100s, `--depth 1 --filter=blob:none` did not finish in 100
# minutes. (Moot against file:///mirror, which is local, but this script also
# runs against a real origin from .tune's worker container.)
if [ ! -d "${SRC_DIR}/.git" ]; then
  git init "${SRC_DIR}"
  git -C "${SRC_DIR}" remote add origin "${LLVM_SOURCE}"
else
  git -C "${SRC_DIR}" remote set-url origin "${LLVM_SOURCE}"
fi
if ! git -C "${SRC_DIR}" cat-file -e "${COMMIT}^{commit}" 2>/dev/null; then
  git -C "${SRC_DIR}" fetch --depth=1 origin "${COMMIT}"
fi
git -C "${SRC_DIR}" checkout -f "${COMMIT}"

# --- Python build deps for the MLIR bindings ---
# MLIR's Python bindings are nanobind-based, and the version used to build them
# has to agree with the one FlyDSL later builds its own bindings with -- they
# share the "mlir" nanobind domain (MLIR_BINDINGS_PYTHON_NB_DOMAIN below).
# Pinned to the same default runc-build-flydsl-wheel.sh uses, for that reason
# alone; bump both together or not at all.
NANOBIND_VERSION="${NANOBIND_VERSION:-2.12.0}"
python -m pip install "nanobind==${NANOBIND_VERSION}" numpy pybind11
NANOBIND_DIR=$(python -c "import nanobind, os; print(os.path.dirname(nanobind.__file__) + '/cmake')")

# --- Configure ---
# These flags are FlyDSL's, not ours. They are transcribed from FlyDSL's
# scripts/build_llvm.sh at the tag third_party/flydsl-compiler.txt pins
# (v0.3.1, commit 421935cc), because FlyDSL's CMakeLists.txt does a plain
# `find_package(MLIR REQUIRED CONFIG)` against MLIR_PATH/lib/cmake/mlir and
# inherits whatever that build decided:
#
#   LLVM_ENABLE_PROJECTS=mlir;clang;lld    mlir is the point; lld provides the
#                                          ld.lld the AMDGPU path links objects
#                                          with; clang is in FlyDSL's list and
#                                          dropping it has not been tested.
#   LLVM_TARGETS_TO_BUILD=X86;NVPTX;AMDGPU AMDGPU is the one we need. X86 is the
#                                          host. NVPTX is FlyDSL's, kept so the
#                                          tarball is the one their setup.py
#                                          expects rather than a subset of it.
#   LLVM_ENABLE_ASSERTIONS=ON              FlyDSL builds and tests against an
#                                          asserting MLIR; a NDEBUG build turns
#                                          their invariant failures into
#                                          miscompiles instead of aborts.
#   MLIR_ENABLE_BINDINGS_PYTHON=ON,        FlyDSL's wheel embeds `flydsl._mlir`,
#   MLIR_BINDINGS_PYTHON_NB_DOMAIN=mlir    built from the sources this install
#                                          tree carries. Without these the
#                                          install has no bindings to build from.
#   BUILD_SHARED_LIBS / *_DYLIB=OFF        FlyDSL links MLIR statically into its
#                                          own extension module.
#
# Do not "improve" this list. If FlyDSL's build_llvm.sh changes -- it gained an
# amd-minimal profile and an ROCDL lld patch after v0.3.1 -- the change belongs
# here, in the same commit that moves flydsl-compiler.txt.
#
# NOT set here on purpose: Python3_EXECUTABLE picks up whatever python is on
# PATH, and that is fine. The tarball is *not* Python-ABI specific and is not
# cached per Python version: FlyDSL rebuilds MLIR's bindings from this install
# tree's sources for each target interpreter, which is why their own CI builds
# LLVM once and then loops build_wheels.sh over Python 3.10 through 3.14
# against the single MLIR_PATH.
cmake -G Ninja \
  -S "${SRC_DIR}/llvm" \
  -B "${CMAKE_BUILD_DIR}" \
  -DLLVM_ENABLE_PROJECTS="${LLVM_ENABLE_PROJECTS:-mlir;clang;lld}" \
  -DLLVM_TARGETS_TO_BUILD="${LLVM_TARGETS_TO_BUILD:-X86;NVPTX;AMDGPU}" \
  -DLLVM_ENABLE_RUNTIMES="${LLVM_ENABLE_RUNTIMES-compiler-rt}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=17 \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_INSTALL_UTILS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=mlir \
  -DPython3_EXECUTABLE="$(which python3)" \
  -Dnanobind_DIR="${NANOBIND_DIR}" \
  -DBUILD_SHARED_LIBS=OFF \
  -DLLVM_BUILD_LLVM_DYLIB=OFF \
  -DLLVM_LINK_LLVM_DYLIB=OFF \
  -DMLIR_INCLUDE_TESTS=OFF \
  -DHIP_PLATFORM=amd \
  -DCMAKE_INSTALL_RPATH='$ORIGIN'

# --- Build and install ---
cmake --build "${CMAKE_BUILD_DIR}" -j"${JOBS}"

rm -rf "${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}"
cmake --install "${CMAKE_BUILD_DIR}" --prefix "${INSTALL_DIR}"

# The one thing FlyDSL actually looks for. Check it here rather than letting
# find_package(MLIR) fail an hour later in a different container.
if [ ! -d "${INSTALL_DIR}/lib/cmake/mlir" ]; then
  echo "Error: install prefix has no lib/cmake/mlir: ${INSTALL_DIR}" >&2
  exit 1
fi

# The install tree is mostly bin/, carrying a symbol table nothing downstream
# reads; stripping takes roughly a fifth off a tarball that gets copied around
# on every cache miss. Static archives are left alone -- stripping an archive
# can drop symbols the link still needs. Set LLVM_STRIP_INSTALL=0 to keep
# symbols for crash backtraces.
if [ "${LLVM_STRIP_INSTALL:-1}" == "1" ] && command -v strip >/dev/null 2>&1; then
  find "${INSTALL_DIR}/bin" "${INSTALL_DIR}/lib" "${INSTALL_DIR}/python_packages" \
       -type f ! -name '*.a' -print0 2>/dev/null |
    while IFS= read -r -d '' f; do
      strip --strip-unneeded "${f}" 2>/dev/null || true
    done
fi

# --- Package ---
mkdir -p "${OUTPUT_DIR}"
# Write to a temporary name and rename into place: the caller's cache check is
# "does this filename exist", so a half-written tarball from an interrupted run
# must never answer to it.
#
# --warning=no-file-changed: the install tree can still gain __pycache__ entries
# while tar reads it, which GNU tar reports as a fatal "file changed as we read
# it". That is not a corrupt archive, just a race with bytecode caching.
tar --warning=no-file-changed --warning=no-file-removed --ignore-failed-read \
    -C "${INSTALL_ROOT}" -czf "${TARBALL}.tmp" "${TARBALL_NAME}"
# --ignore-failed-read is what makes the pycache race non-fatal, and it is also
# what makes a REAL read failure non-fatal: tar exits 0 either way, so a
# truncated archive would be renamed into place and answer the caller's
# "does this filename exist" cache check forever. Read the archive back before
# the rename -- it decompresses the whole stream and checks every member
# header, which is the cheapest thing that can tell the two apart.
if ! tar -tzf "${TARBALL}.tmp" >/dev/null; then
  echo "Error: ${TARBALL}.tmp is not a readable archive; refusing to publish it." >&2
  rm -f "${TARBALL}.tmp"
  exit 1
fi
mv "${TARBALL}.tmp" "${TARBALL}"

ls -l "${TARBALL}"
