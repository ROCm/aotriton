#!/bin/bash

if [ -z "$BASH_VERSION" ]; then
  echo "This script requires Bash. Please run it with 'bash script_name.sh' or ensure /bin/sh points to /bin/bash." >&2
  exit 1
fi

if [ "$#" -ne 3 ]; then
  echo 'Missing arguments. Usage: run-test.sh <pass#> <test_level> <split/fused/aiter/v3>' >&2
  exit 1
fi

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
. "${SCRIPT_DIR}/common-vars.sh"
add_torch_ldconfig
add_rocm_sdk_ldconfig

pass=$1
test_level="$2"
backend="$3"
if [ -n "${AOTRITON_TEST_LIBDIR:-}" ]; then
  bdir=""
else
  mapfile -d '' bdir_cans < <(find . -maxdepth 1 -type d -name "build-${aotriton_major}.${aotriton_minor}-test-*${native_arch}*" -print0)
  if [ ${#bdir_cans[@]} -gt 1 ]; then
    echo "There are multiple build directory candidates matching pattern 'build-${aotriton_major}.${aotriton_minor}-test-*${native_arch}*' for testing: ${bdir_cans[@]}. Please keep one only"
    exit 1
  fi
  bdir="${bdir_cans[0]}"
fi

small_vram=$(amd-smi static -g 0 -v --json|grep -v '^WARNING:'| python -c 'import json, sys; j = json.load(sys.stdin); print(int(j["gpu_data"][0]["vram"]["size"]["value"] / 1024.0 < 60))')

# Output directory: use $OUTPUT_DIR if set, otherwise current directory
outdir="${OUTPUT_DIR:-.}"
mkdir -p "$outdir"

# Partial test mode: if PARTIAL_INFO_DIR is set, fix and use sel files as pytest selectors
SELECT_FROM=""
SELECT_VARLEN_FROM=""
if [ -n "${PARTIAL_INFO_DIR:-}" ]; then
  for kind in "" ".varlen"; do
    src="${PARTIAL_INFO_DIR}/sel${pass}${kind}.txt"
    dst="${outdir}/pytest-select-${pass}${kind}.txt"
    if [ -f "$src" ]; then
      # Remove "path/to/file.py::" prefix (first occurrence per line only)
      sed 's|[^:]*\.py::||' "$src" > "$dst"
      if [ -z "$kind" ]; then
        SELECT_FROM="--select-from-file $dst"
      else
        SELECT_VARLEN_FROM="--select-from-file $dst"
      fi
    fi
  done
fi

if [ -n "${USE_ADIFFS_TXT:-}" ]; then
  if [ -f "$USE_ADIFFS_TXT" ]; then
    echo "USE_ADIFFS_TXT: $USE_ADIFFS_TXT ($(wc -l < "$USE_ADIFFS_TXT") lines)"
  else
    echo "USE_ADIFFS_TXT: $USE_ADIFFS_TXT does not exist, unsetting"
    unset USE_ADIFFS_TXT
  fi
fi

(
  ulimit -c 0
  cd ${SCRIPT_DIR}/..;
  export SMALL_VRAM=${small_vram};
  export COLUMNS=400;
  export FOR_RELEASE=${test_level};
  export AOTRITON_UT_ARCH="${native_arch}";
  if [[ "$backend" == "split" ]]; then
    export BWD_IMPL=0
    fnprefix="ut_pass"
  fi
  if [[ "$backend" == "fused" ]]; then
    export V3_API=1
    export BWD_IMPL=1
    fnprefix="fused_pass"
  fi
  if [[ "$backend" == "aiter" ]]; then
    export V3_API=1
    export BWD_IMPL=2
    fnprefix="aiter_pass"
  fi
  if [[ "$backend" == "v3" ]]; then
    export V3_API=1
    fnprefix="oput_pass"
  fi
  set -v
  export PYTHONPATH="${AOTRITON_TEST_LIBDIR:-${bdir}/install_dir/lib}"
  _sig=$(ls "$PYTHONPATH/aotriton.images/"*"/__signature__" 2>/dev/null | head -n 1)
  {
    [ -n "$_sig" ] && cat "$_sig" \
      || echo "NO __signature__ file at $PYTHONPATH/aotriton.images/"
  } | tee "${outdir}/${fnprefix}${pass}.out" \
          "${outdir}/${fnprefix}${pass}.varlen.out" > /dev/null
  pytest --tb=line -n ${ngpus} --max-worker-restart 9999 -rfEsx \
    --timeout=300 --timeout-method=thread \
    -p no:cacheprovider \
    ${SELECT_FROM} \
    test/test_backward.py \
    -v \
    1>>"${outdir}/${fnprefix}${pass}.out" \
    2>"${outdir}/${fnprefix}${pass}.err" || true
  grep '^FAILED' "${outdir}/${fnprefix}${pass}.out"|sed 's/^FAILED //' | sed 's/].*/]/' > "${outdir}/sel${pass}.txt"
  pytest --tb=line -n ${ngpus} --max-worker-restart 9999 -rfEsx \
    --timeout=300 --timeout-method=thread \
    -p no:cacheprovider \
    ${SELECT_VARLEN_FROM} \
    test/test_varlen.py \
    -v \
    1>>"${outdir}/${fnprefix}${pass}.varlen.out" \
    2>"${outdir}/${fnprefix}${pass}.varlen.err" || true
  grep '^FAILED' "${outdir}/${fnprefix}${pass}.varlen.out"|sed 's/^FAILED //' | sed 's/].*/]/' > "${outdir}/sel${pass}.varlen.txt"
  if [ -n "${RECORD_ADIFFS_TO:-}" ]; then
    SCRIPT_DIR_ABS="$(cd "${SCRIPT_DIR}" && pwd)"
    bash "${SCRIPT_DIR_ABS}/../.tune/bin/append_oom_to_adiffs.sh" "${outdir}/${fnprefix}${pass}.out" >> "${RECORD_ADIFFS_TO}"
    bash "${SCRIPT_DIR_ABS}/../.tune/bin/append_oom_to_adiffs.sh" "${outdir}/${fnprefix}${pass}.varlen.out" >> "${RECORD_ADIFFS_TO}"
  fi
)
