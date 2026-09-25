#!/bin/bash

# The shared half of every `.ci/build-*.sh` usage message.
#
# Its own file, and deliberately a *quiet* one: no `set -x`, no GPU probe, no
# version parsing. A profile sources this and answers an empty command line
# BEFORE it sources common-build.sh, because that one pulls in common-vars.sh,
# which turns on tracing and warns about a missing llvm-hash.txt -- forty lines
# of `++ rocm_agent_enumerator` in front of a help message.

# Notes that do NOT belong in `--help` -- a maintainer wants them once, a
# caller listing the flags never does:
#
# --flydsl_kernel_root: left unset, CMake shallow-clones the ref named by
#   third_party/flydsl-kernel.txt into <build dir>/flydsl and points
#   AOTRITON_FLYDSL_KERNEL_ROOT at that, so no build needs this flag to
#   configure. Pass it to build against local FlyDSL kernel changes. No
#   head-dim range needs it: the gfx950 kernels above head_dim 128 want an
#   LDS-transpose emitter the pinned FlyDSL release lacks, but those five
#   emitters are vendored in modules/flash/flyc/fmha_dualwave_gfx950.py, so
#   the default clone builds them correctly. See docs/FlyDSL.md.
#
# --flydsl_wheel: for an unreleased build, one carrying a compiler patch say.
#   Once that build is published, move the pin instead. The wheel must match
#   the build venv Python -- flydsl wheels are CPython-ABI specific. The
#   compiler wheel and the kernel source tree are independent pins, so a
#   patched wheel may or may not want a matching --flydsl_kernel_root.
#
# --name_suffix: the suffix exists so a build can coexist with the AOTriton
#   PyTorch ships (CMakeLists.txt:293). A build meant to REPLACE it passes ''.
function common_build_usage_options() {
  cat <<'EOF' >&2

Options (shared by every .ci/build-*.sh):
  --database_root <dir>       -DAOTRITON_TUNING_DATABASE_ROOT
  --flydsl_kernel_root <dir>  -DAOTRITON_FLYDSL_KERNEL_ROOT
  --flydsl_wheel <whl>        -DAOTRITON_USE_LOCAL_FLYDSL_WHEEL
  --altwheel_config <yaml>    -DAOTRITON_ALT_TRITON_WHEEL_CONFIG_FILE
  --name_suffix <suffix>      -DAOTRITON_NAME_SUFFIX; '' for no suffix
  --mold / --no_mold          Force the mold linker on/off
  --help                      This message

Environment: AOTRITON_BUILD_PATH, AOTRITON_INSTALL_PATH,
             AOTRITON_NAME_SUFFIX_OVERRIDE
EOF
}

# Answer a request for usage before the caller sources anything noisy.
#
# An empty command line is one: every profile needs at least a target arch, and
# guessing one from the local GPU is how a build lands in a directory the caller
# did not mean. `--help` is the other, and it has to be answered here rather
# than in the option parser for the same reason -- by the time getopt sees it,
# common-vars.sh has already printed thirty lines of trace.
function common_build_usage_if_requested() {
  if [ "$#" -eq 0 ]; then
    usage
    exit 1
  fi
  local arg
  for arg in "$@"; do
    case "$arg" in
      --help|-h) usage; exit 0 ;;
      --) break ;;
    esac
  done
}
