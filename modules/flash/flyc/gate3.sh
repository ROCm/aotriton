#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Cross-compile vendored FlyDSL kernels to .hsaco, for architectures this host
# is not, with no GPU present and no AOTriton build.
#
# TEMPORARY, and it lives here rather than in .ci/ for that reason: .ci/ is
# where permanent build entry points live, and this is neither permanent nor a
# build entry point. It sits next to the kernels it compiles and is deleted
# once the image build compiles every flyc kernel for every functional, at
# which point that build subsumes this check completely.
#
#   bash modules/flash/flyc/gate3.sh                    # the default matrix
#   bash modules/flash/flyc/gate3.sh --kernel flyc_bwd_dq --arch gfx950 \
#        --head_dim 192 --dtype bf16
#
# Requirements, none of which is a GPU:
#   * a Python interpreter with `flydsl` and `numpy` installed, matching the
#     wheel's CPython ABI tag. Point at it with --python or $PYTHON.
#   * ROCM_PATH, or anything python/flyc_bootstrap.py's resolve_rocm_path()
#     can find: what it needs is <ROCM_PATH>/llvm/bin/ld.lld at exactly that
#     relative path.
#   * network, once, to clone the FlyDSL source tree at the kernel pin --
#     unless $AOTRITON_FLYDSL_KERNEL_ROOT already points at one.
#
# Pass condition: one .hsaco per configuration whose ELF flags name the
# architecture that was asked for. NOT a size: the size varies with the dtype
# and with the toolchain and is not a contract.

set -u -o pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

PYTHON="${PYTHON:-python3}"
OUTDIR=""
KERNELS=()
ARCHES=()
HEAD_DIMS=()
DTYPES=()
CAUSAL_TYPE=0
BIAS_TYPE=0
ENABLE_DROPOUT=False
PADDED_HEAD=False
KEEP=0

usage() {
    sed -n '5,29p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options (each repeatable; omitting all of the first four runs the default
matrix, which is what the gate means unqualified):
  --kernel <flyc_attn_fwd|flyc_bwd_dkdv|flyc_bwd_dq>
  --arch <gfx1201|gfx950>
  --head_dim <N>
  --dtype <fp16|bf16>
  --causal_type <N>          default 0
  --bias_type <N>            default 0
  --enable_dropout <True|False>
  --padded_head <True|False>
  --out_dir <dir>            default: a temporary directory
  --python <path>            interpreter with flydsl + numpy
  --keep                     do not delete the temporary output directory
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kernel)          KERNELS+=("$2"); shift 2 ;;
        --arch)            ARCHES+=("$2"); shift 2 ;;
        --head_dim)        HEAD_DIMS+=("$2"); shift 2 ;;
        --dtype)           DTYPES+=("$2"); shift 2 ;;
        --causal_type)     CAUSAL_TYPE="$2"; shift 2 ;;
        --bias_type)       BIAS_TYPE="$2"; shift 2 ;;
        --enable_dropout)  ENABLE_DROPOUT="$2"; shift 2 ;;
        --padded_head)     PADDED_HEAD="$2"; shift 2 ;;
        --out_dir)         OUTDIR="$2"; shift 2 ;;
        --python)          PYTHON="$2"; shift 2 ;;
        --keep)            KEEP=1; shift ;;
        -h|--help)         usage; exit 0 ;;
        *) echo "gate3.sh: unknown option $1" >&2; usage >&2; exit 2 ;;
    esac
done

# The default matrix. head_dim 192 on gfx950 is deliberate rather than
# convenient: it is above the 128 threshold where the DS-transpose and
# register-spill problems live, so those rows exercise the vendored emitters
# instead of avoiding them.
#
# Each row is "kernel arch head_dim dtype".
DEFAULT_MATRIX=(
    "flyc_attn_fwd  gfx1201  64   fp16"
    "flyc_attn_fwd  gfx1201  128  bf16"
    "flyc_bwd_dkdv  gfx1201  128  bf16"
    "flyc_bwd_dq    gfx1201  128  bf16"
    "flyc_attn_fwd  gfx950   192  bf16"
    "flyc_bwd_dkdv  gfx950   192  bf16"
    "flyc_bwd_dq    gfx950   192  bf16"
)

MATRIX=()
if [[ ${#KERNELS[@]} -eq 0 && ${#ARCHES[@]} -eq 0 && ${#HEAD_DIMS[@]} -eq 0 && ${#DTYPES[@]} -eq 0 ]]; then
    MATRIX=("${DEFAULT_MATRIX[@]}")
else
    [[ ${#KERNELS[@]}   -eq 0 ]] && KERNELS=(flyc_attn_fwd flyc_bwd_dkdv flyc_bwd_dq)
    [[ ${#ARCHES[@]}    -eq 0 ]] && ARCHES=(gfx1201 gfx950)
    [[ ${#HEAD_DIMS[@]} -eq 0 ]] && HEAD_DIMS=(128)
    [[ ${#DTYPES[@]}    -eq 0 ]] && DTYPES=(bf16)
    for k in "${KERNELS[@]}"; do
        for a in "${ARCHES[@]}"; do
            for h in "${HEAD_DIMS[@]}"; do
                for d in "${DTYPES[@]}"; do
                    MATRIX+=("$k $a $h $d")
                done
            done
        done
    done
fi

TMP="$(mktemp -d)"
cleanup() {
    [[ $KEEP -eq 1 ]] && return
    rm -rf "$TMP"
}
trap cleanup EXIT

[[ -n "$OUTDIR" ]] || OUTDIR="$TMP/out"
mkdir -p "$OUTDIR"

# `python -m aotriton.X` needs python/ visible under the package name
# `aotriton`. A symlink is the whole of it -- no cmake configure, no build
# venv, which is most of this gate's value.
ln -s "$REPO/python" "$TMP/aotriton"
export PYTHONPATH="$TMP${PYTHONPATH:+:$PYTHONPATH}"

# No device is used, and asserting that is part of the point: the targets are
# gfx1201 and gfx950 and the build host is neither.
export HIP_VISIBLE_DEVICES=

# kernels/common and kernels/attention, from the pin. Reuse an existing
# checkout when the caller has one.
if [[ -z "${AOTRITON_FLYDSL_KERNEL_ROOT:-}" ]]; then
    FLYDSL_TAG="$(grep -v '^[[:space:]]*#' "$REPO/third_party/flydsl-kernel.txt" | head -n 1 | tr -d '[:space:]')"
    echo "gate3: cloning https://github.com/ROCm/FlyDSL.git at $FLYDSL_TAG"
    git clone --depth 1 --branch "$FLYDSL_TAG" \
        https://github.com/ROCm/FlyDSL.git "$TMP/flydsl" >/dev/null || exit 1
    export AOTRITON_FLYDSL_KERNEL_ROOT="$TMP/flydsl"
fi
echo "gate3: AOTRITON_FLYDSL_KERNEL_ROOT=$AOTRITON_FLYDSL_KERNEL_ROOT"

# llvm-readelf, for the one thing this gate actually checks. Prefer the ROCm
# toolchain's, since resolve_rocm_path() has already found one.
find_readelf() {
    local rocm
    rocm="$("$PYTHON" -c \
        'import sys; sys.path.insert(0, sys.argv[1]); import flyc_bootstrap; print(flyc_bootstrap.resolve_rocm_path())' \
        "$REPO/python" 2>/dev/null)"
    if [[ -n "$rocm" && -x "$rocm/llvm/bin/llvm-readelf" ]]; then
        echo "$rocm/llvm/bin/llvm-readelf"
        return
    fi
    command -v llvm-readelf || command -v readelf
}
READELF="$(find_readelf)"
if [[ -z "$READELF" ]]; then
    echo "gate3: no llvm-readelf or readelf on PATH, and none in ROCM_PATH" >&2
    exit 1
fi
echo "gate3: readelf=$READELF"
echo

pass=0
fail=0

for row in "${MATRIX[@]}"; do
    read -r kernel arch head_dim dtype <<<"$row"
    desc="$REPO/modules/flash/aot/$kernel.py"
    if [[ ! -f "$desc" ]]; then
        echo "FAIL $kernel $arch hd$head_dim $dtype -- no description at $desc"
        fail=$((fail + 1))
        continue
    fi
    # The FULL signature, printed rather than summarised. It has more axes than
    # a (kernel, arch, head_dim) label shows -- the dtype alone moves the
    # emitted code -- and a result recorded without it cannot be reproduced.
    signature="Q='*$dtype:16' BLOCK_DMODEL=$head_dim CAUSAL_TYPE=$CAUSAL_TYPE"
    signature="$signature BIAS_TYPE=$BIAS_TYPE ENABLE_DROPOUT=$ENABLE_DROPOUT"
    signature="$signature PADDED_HEAD=$PADDED_HEAD"
    hints="seqlen_q=0 seqlen_k=0"
    out="$OUTDIR/${kernel}_${arch}_hd${head_dim}_${dtype}"

    echo "=== $kernel $arch head_dim=$head_dim $dtype"
    echo "    --signature \"$signature\""
    echo "    --hints     \"$hints\""

    if ! "$PYTHON" -m aotriton.flyc_compile "$desc" \
            --kernel_name "$kernel" --target "$arch" \
            --signature "$signature" --hints "$hints" \
            --out_path "$out" --verify; then
        echo "    FAIL: flyc_compile returned non-zero"
        fail=$((fail + 1))
        continue
    fi
    if [[ ! -f "$out.hsaco" ]]; then
        echo "    FAIL: no $out.hsaco"
        fail=$((fail + 1))
        continue
    fi
    flags="$("$READELF" -h "$out.hsaco" | sed -n 's/^  Flags: *//p')"
    if [[ "$flags" != *"$arch"* ]]; then
        echo "    FAIL: ELF flags '$flags' do not name $arch"
        fail=$((fail + 1))
        continue
    fi
    # Size is reported, NOT checked. It moves with the dtype and with the
    # toolchain, and two runs of one configuration matching is a property of
    # the driver rather than a contract this gate enforces.
    echo "    PASS: $(stat -c %s "$out.hsaco") bytes, flags [$flags]"
    pass=$((pass + 1))
done

echo
echo "gate3: $pass passed, $fail failed, of $((pass + fail))"
[[ $fail -eq 0 ]]
