# Convenience wrappers around releasesuite-git-head.sh.
#
# Usage (from anywhere):
#   make -C .ci check-runtime
#   make -C .ci check-gpu
#   make -C .ci release-gpu     OUTDIR=/path/to/out
#   make -C .ci release-runtime ORIGIN=https://github.com/me/aotriton.git
#
# Every knob below is overridable on the command line.

# Directory of this Makefile, so the suite is found regardless of CWD.
CI_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
SUITE  := bash $(CI_DIR)releasesuite-git-head.sh
CMAKELISTS := $(CI_DIR)../CMakeLists.txt

# Release version directory, derived from CMakeLists.txt. Legacy naming: a
# x.y.0 release drops the ".0" and is named "x.y"; a x.y.z (z>0) release keeps
# it as "x.y.z". A "b" (beta) suffix is appended while the major version is
# still 0 (i.e. until x >= 1). So 0.13.0 -> "0.13b", 0.12.1 -> "0.12.1b".
VERSION_DIR := $(shell \
  M=$$(sed -n 's/^set.AOTRITON_VERSION_MAJOR_INT \([0-9]*\).*/\1/p' $(CMAKELISTS)); \
  N=$$(sed -n 's/^set.AOTRITON_VERSION_MINOR_INT \([0-9]*\).*/\1/p' $(CMAKELISTS)); \
  P=$$(sed -n 's/^set.AOTRITON_VERSION_PATCH_INT \([0-9]*\).*/\1/p' $(CMAKELISTS)); \
  V="$$M.$$N"; \
  if [ "$$P" != "0" ]; then V="$$V.$$P"; fi; \
  if [ "$$M" -lt 1 ]; then V="$$V"b; fi; \
  echo "$$V")

# Output directory for release artifacts.
OUTDIR ?= $(HOME)/aotriton-gh-release/release/$(VERSION_DIR)/hashed

# Git origin forwarded to the suite. "auto" derives it from the tracked
# remote (ssh/git URLs are rewritten to https).
ORIGIN ?= auto

# Latest stable ROCm release.
STABLE_ROCM ?= 7.2.4
# Latest TheRock build. These are pre-release/nightlies, so use the long
# version string. gfx1250 (Tech Preview) needs a gfx1250-capable TheRock ROCm.
THEROCK_ROCM ?= 7.15.0a20260707

# ASAN is supported only on gfx942 and gfx950, and only on TheRock ROCm
# (needs TheRock clang). Its runtime ships for both 7.14 and 7.15.
ASAN_ARCH ?= gfx942;gfx950
ASAN_ROCM ?= 7.14.0a20260624 7.15.0a20260707
ASAN_RFLAGS := $(addprefix -r ,$(ASAN_ROCM))

.PHONY: check-runtime check-gpu release-gpu release-runtime \
        check-runtime-asan check-gpu-asan release-gpu-asan release-runtime-asan \
        check-all release-all

# Smoke check: only the C++ runtime, on both the stable and TheRock ROCm.
check-runtime:
	$(SUITE) --runtime -r $(STABLE_ROCM) -r $(THEROCK_ROCM) --origin $(ORIGIN) $(OUTDIR)

# Smoke check: only the gfx1250 GPU image.
check-gpu:
	$(SUITE) --image --arch gfx1250 -r $(THEROCK_ROCM) --origin $(ORIGIN) $(OUTDIR)

# Release: all GPU images (every arch in the default AOTRITON_TARGET_ARCH).
release-gpu:
	$(SUITE) --image --origin $(ORIGIN) $(OUTDIR)

# Release: the C++ runtime on every ROCm in the default runtime list.
release-runtime:
	$(SUITE) --runtime --origin $(ORIGIN) $(OUTDIR)

# Smoke check: only the ASAN C++ runtime, on the latest TheRock ROCm.
check-runtime-asan:
	$(SUITE) --asan --runtime -r $(THEROCK_ROCM) --origin $(ORIGIN) $(OUTDIR)

# Smoke check: only one ASAN GPU image (gfx942).
check-gpu-asan:
	$(SUITE) --asan --image --arch gfx942 -r $(THEROCK_ROCM) --origin $(ORIGIN) $(OUTDIR)

# Release: ASAN GPU images for the supported arches on the latest TheRock ROCm.
release-gpu-asan:
	$(SUITE) --asan --image --arch '$(ASAN_ARCH)' -r $(THEROCK_ROCM) --origin $(ORIGIN) $(OUTDIR)

# Release: the ASAN C++ runtime on every supported TheRock ROCm (7.14 and 7.15).
release-runtime-asan:
	$(SUITE) --asan --runtime $(ASAN_RFLAGS) --origin $(ORIGIN) $(OUTDIR)

# Run every smoke check: non-ASAN and ASAN, GPU images and runtimes.
check-all: check-gpu check-runtime check-gpu-asan check-runtime-asan

# Build every release target: non-ASAN and ASAN, GPU images and runtimes.
release-all: release-gpu release-runtime release-gpu-asan release-runtime-asan
