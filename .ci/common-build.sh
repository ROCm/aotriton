#!/bin/bash

# The shared implementation behind every `.ci/build-*.sh`.
#
# Each front end is a **profile**: a build type, a name suffix, a build-dir
# tag and a handful of cmake flags. Everything else -- option parsing, the
# local-artifact plumbing, the cmake invocation -- lives here, so a feature
# added for one build is available to all of them. That was not true before:
# `--flydsl_wheel`, `--altwheel_config` and the optional triton-wheel
# positional had been copy-pasted between build-test.sh and build-tune.sh down
# to the comments, while `--database_root` and `--flydsl_kernel_root` existed
# only in build-test.sh despite nothing about them being test-specific.
#
# A profile looks like this:
#
#     . "${SCRIPT_DIR}/common-usage.sh"
#     usage() { echo 'Usage: build-x.sh [options] <target arch>' >&2
#               common_build_usage_options; }
#     common_build_usage_if_requested "$@"
#     . "${SCRIPT_DIR}/common-build.sh"
#     CB_PROG=build-x.sh
#     common_build_parse "$@"
#     common_build_take_arch
#     common_build_take_triton_wheel
#     common_build_run Release 123 x -DSOMETHING=ON
#
# The order is load-bearing: `usage` has to be defined and an empty or `--help`
# command line answered BEFORE this file is sourced, because sourcing it pulls
# in common-vars.sh, which probes the GPU and warns about a missing
# llvm-hash.txt. A help message should not arrive behind either. `usage` is
# also called from the parser for a bad option or a missing arch, so every
# profile must define it.

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
# common-vars.sh turns on `set -x` in its first line, so it traces its own
# body as it is sourced -- a `set +x` after the fact is too late. Send xtrace
# to /dev/null for the duration instead, which leaves `set -e` and
# common-vars.sh's own diagnostics (plain stderr, not xtrace) alone.
#
# Tracing then stays off for the option phase: it is short, already described
# by the usage text, and forty lines of `+ local opt` are the first thing
# between a caller and an error message. `common_build_run` restores it, so
# build logs are unchanged from the cmake invocation onwards.
exec {_cb_xtrace_fd}>/dev/null
BASH_XTRACEFD=${_cb_xtrace_fd}
. "${SCRIPT_DIR}/common-vars.sh"
. "${SCRIPT_DIR}/common-usage.sh"
set +x
BASH_XTRACEFD=2
exec {_cb_xtrace_fd}>&-
unset _cb_xtrace_fd

# Long options every profile accepts. A profile adds its own through
# `CB_EXTRA_LONGOPTS` (getopt syntax: a trailing `:` means "takes a value").
_CB_LONGOPTS='database_root:,flydsl_kernel_root:,flydsl_wheel:,altwheel_config:,name_suffix:,mold,no_mold,help'

# common_build_parse "$@"
#
# Reads CB_PROG (required), CB_EXTRA_LONGOPTS and CB_MOLD_DEFAULT from the
# profile. Sets:
#   CB_CMAKE_ARGS[]  cmake -D flags derived from the shared options
#   CB_POSITIONAL[]  what was left after the options
#   CB_OPT[<name>]   values of the profile's own long options
#   CB_ALTWHEEL      non-empty when --altwheel_config was given
function common_build_parse() {
    local prog="${CB_PROG:-$(basename "${BASH_SOURCE[1]}")}"
    local extra="${CB_EXTRA_LONGOPTS:-}"
    local mold="${CB_MOLD_DEFAULT:-false}"

    common_build_usage_if_requested "$@"

    # Which of the profile's own options take a value, read off the same
    # getopt syntax the profile declared them in.
    declare -gA CB_OPT=()
    local -A _cb_extra_has_arg=()
    local opt
    for opt in ${extra//,/ }; do
        if [ "${opt: -1}" = ':' ]; then
            _cb_extra_has_arg["--${opt%:}"]=1
        else
            _cb_extra_has_arg["--${opt}"]=0
        fi
    done

    local TEMP
    if ! TEMP=$(getopt -o '' --long "${_CB_LONGOPTS}${extra:+,$extra}" -n "${prog}" -- "$@"); then
        usage
        exit 1
    fi
    eval set -- "$TEMP"

    local database_root='' flydsl_kernel_root='' flydsl_wheel='' altwheel_config=''
    CB_ALTWHEEL=''
    # Set-ness, not non-emptiness: `--name_suffix ''` is a meaningful request
    # (no suffix at all) and has to survive to _common_build.
    local name_suffix_given=false name_suffix=''
    while true; do
        case "$1" in
            --database_root)       database_root="$2"; shift 2 ;;
            --flydsl_kernel_root)  flydsl_kernel_root="$2"; shift 2 ;;
            --flydsl_wheel)        flydsl_wheel="$2"; shift 2 ;;
            --altwheel_config)     altwheel_config="$2"; shift 2 ;;
            --name_suffix)         name_suffix_given=true; name_suffix="$2"; shift 2 ;;
            --mold)                mold=true; shift ;;
            --no_mold)             mold=false; shift ;;
            --help)                usage; exit 0 ;;
            --) shift; break ;;
            *)
                if [ "${_cb_extra_has_arg[$1]:-}" = '1' ]; then
                    CB_OPT["${1#--}"]="$2"; shift 2
                elif [ "${_cb_extra_has_arg[$1]:-}" = '0' ]; then
                    CB_OPT["${1#--}"]=true; shift
                else
                    echo "Internal error: unhandled option $1" >&2
                    exit 1
                fi
                ;;
        esac
    done

    CB_POSITIONAL=("$@")
    CB_CMAKE_ARGS=()

    if [ "$mold" = true ]; then
        CB_CMAKE_ARGS+=(-DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=mold"
                        -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=mold")
    fi
    if [ -n "$database_root" ]; then
        CB_CMAKE_ARGS+=("-DAOTRITON_TUNING_DATABASE_ROOT=${database_root}")
    fi
    # realpath is what satisfies cmake's absolute-path requirement on these.
    if [ -n "$flydsl_kernel_root" ]; then
        CB_CMAKE_ARGS+=("-DAOTRITON_FLYDSL_KERNEL_ROOT=$(realpath "$flydsl_kernel_root")")
    fi
    if [ -n "$flydsl_wheel" ]; then
        CB_CMAKE_ARGS+=("-DAOTRITON_USE_LOCAL_FLYDSL_WHEEL=$(realpath "$flydsl_wheel")")
    fi
    if [ -n "$altwheel_config" ]; then
        CB_ALTWHEEL=1
        CB_CMAKE_ARGS+=("-DAOTRITON_ALT_TRITON_WHEEL_CONFIG_FILE=$(realpath "$altwheel_config")")
    fi
    if [ "$name_suffix_given" = true ]; then
        export AOTRITON_NAME_SUFFIX_OVERRIDE="$name_suffix"
    fi
    return 0
}

# common_build_take_arch [default]
#
# Pops <target arch> off CB_POSITIONAL into CB_ARCH. With no default, a missing
# arch is a usage error.
function common_build_take_arch() {
    if [ "${#CB_POSITIONAL[@]}" -ge 1 ]; then
        CB_ARCH="${CB_POSITIONAL[0]}"
        CB_POSITIONAL=("${CB_POSITIONAL[@]:1}")
    elif [ "$#" -ge 1 ]; then
        CB_ARCH="$1"
    else
        echo "Missing <target arch>." >&2
        usage
        exit 1
    fi
    return 0
}

# common_build_take_triton_wheel
#
# Consumes the optional pre-compiled triton wheel positional.
#
# Not when an altwheel config is set. The two are mutually exclusive at the
# cmake level (fatal error if both given): the altwheel config's own
# .venvs.default supplies the main venv's wheel too, so no separate flag is
# needed here.
function common_build_take_triton_wheel() {
    if [ "${#CB_POSITIONAL[@]}" -ge 1 ] && [ -z "${CB_ALTWHEEL}" ]; then
        CB_CMAKE_ARGS+=("-DAOTRITON_USE_LOCAL_TRITON_WHEEL=$(realpath "${CB_POSITIONAL[0]}")")
    fi
    if [ "${#CB_POSITIONAL[@]}" -ge 1 ]; then
        CB_POSITIONAL=("${CB_POSITIONAL[@]:1}")
    fi
    return 0
}

# common_build_run <build type> <default suffix> <build for> [cmake args...]
#
# The profile's own flags come last so they land after the shared ones, which
# is the order these scripts have always emitted.
function common_build_run() {
    if [ "$#" -lt 3 ]; then
        echo 'common_build_run expects <build type> <default suffix> <build for> [cmake options...]' >&2
        exit 1
    fi
    local build_type="$1" default_suffix="$2" build_for="$3"
    shift 3
    set -x  # tracing was held off for the option phase; see the top
    _common_build "${build_type}" "${default_suffix}" "${CB_ARCH}" "${build_for}" \
        "${CB_CMAKE_ARGS[@]}" "$@"
}

function _common_build() {
  build_type="$1"
  suffix="$2"
  target_arch="$3"
  build_for="$4"
  shift 4
  # `+x`, not the bare value: an **empty** override is a real request -- it is
  # how a build that must REPLACE PyTorch's AOTriton rather than coexist with
  # it asks for an unsuffixed library and namespace. Testing the value instead
  # silently handed such a caller the default suffix.
  if [ -n "${AOTRITON_NAME_SUFFIX_OVERRIDE+x}" ]; then
    suffix="${AOTRITON_NAME_SUFFIX_OVERRIDE}"
  fi
  # Explicit, absolute source dir -- `cmake ..` only works when bdir is a
  # subdirectory of the source tree, which isn't true once AOTRITON_BUILD_PATH
  # points at an external workdir. Same convention as build-release.sh.
  source_dir="$(realpath "${SCRIPT_DIR}/..")"
  if [ -n "${AOTRITON_BUILD_PATH}" ]; then
    bdir="${AOTRITON_BUILD_PATH}"
  else
    bdir="${source_dir}/$(aotriton_build_dir "${build_for}" "${target_arch}")"
  fi
  if [ -n "${AOTRITON_INSTALL_PATH}" ]; then
    install_prefix="${AOTRITON_INSTALL_PATH}"
  else
    install_prefix="./install_dir"
  fi
  mkdir -p ${bdir}
  (
    cd ${bdir};
    cmake "${source_dir}" -DCMAKE_INSTALL_PREFIX=${install_prefix} \
      -DCMAKE_BUILD_TYPE=${build_type} \
      -DAOTRITON_TARGET_ARCH=${target_arch} \
      -DAOTRITON_NAME_SUFFIX=${suffix} \
      "$@" \
      -G Ninja;
    ninja install/strip
  )
}

function common_build() {
  if [ "$#" -lt 2 ]; then
    echo 'common_build expects at least 2 arguments: <target arch> <build for> [cmake options...]' >&2
    exit 1
  fi
  _common_build Release "123" "$@"
}

function debug_build() {
  if [ "$#" -lt 2 ]; then
    echo 'common_build expects at least 2 arguments: <target arch> <build for> [cmake options..]' >&2
    exit 1
  fi
  _common_build Debug "123" "$@"
}
