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

# Partial test mode: if PARTIAL_INFO_DIR is set, use the sel file as a pytest selector
SELECT_FROM=""
if [ -n "${PARTIAL_INFO_DIR:-}" ]; then
  src="${PARTIAL_INFO_DIR}/sel${pass}.txt"
  dst="${outdir}/pytest-select-${pass}.txt"
  if [ -f "$src" ]; then
    # Remove "path/to/file.py::" prefix (first occurrence per line only)
    sed 's|[^:]*\.py::||' "$src" > "$dst"
    SELECT_FROM="--select-from-file $dst"
  fi
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
  } > "${outdir}/${fnprefix}${pass}.out"
  # Start watchdog process, use /dev/shm to avoid wearing: container's /tmp
  # may not be tmpfs.
  # Named with both $pass and $$ for uniqueness.
  # Note: this is inside a subshell. Hence parent pid `$$` is the right one to use.
  export GPU_LEASE_LOCKFILE="/dev/shm/gpu_lease.${pass}.$$"
  # One stderr file for the whole pass, so there is a single thing to tail.
  # Truncated once here and appended to from then on: two `2>` on one path
  # would have pytest re-truncate a file the watchdog already holds open, and
  # the watchdog's fd keeps its own offset, so its next line would land past a
  # hole.
  _errfile="${outdir}/${fnprefix}${pass}.err"
  : > "${_errfile}"
  # Start watchdog service, assume pytest_gpu_lease already installed.
  # If not, do it with `pip install -r requirements-dev.txt`
  #
  # --lockfile is redundant with the exported GPU_LEASE_LOCKFILE the watchdog
  # would fall back to, and passed anyway so that `ps aux` says which file each
  # watchdog is watching. That is how a stale one from a SIGKILLed pass is told
  # apart from the live one; an env var is not visible in ps output.
  python -m pytest_gpu_lease.watchdog --lockfile "${GPU_LEASE_LOCKFILE}" \
    --workers "${ngpus}" 2>>"${_errfile}" &
  watchdog_pid=$!
  # Report either way: a silent success reads exactly like a watchdog that
  # never started, and the difference only becomes visible once a pass has
  # already hung -- with --timeout=300 gone, with no hang protection at all.
  sleep 1
  if kill -0 "${watchdog_pid}" 2>/dev/null; then
    echo "run-test.sh: watchdog running, pid ${watchdog_pid}, lockfile ${GPU_LEASE_LOCKFILE}" >> "${_errfile}"
  else
    echo "run-test.sh: WARNING watchdog did not start; this pass has NO hang protection" \
      | tee -a "${_errfile}" >&2
  fi
  # The watchdog unlinks the lock file itself, so this only has to stop it.
  # `wait` is what makes that deterministic: `kill` just posts the signal, and
  # without waiting we would return while the unlink is still in flight and
  # leave a zombie behind until PID 1 reaps it.
  _stop_watchdog() { kill "${watchdog_pid}" 2>/dev/null; wait "${watchdog_pid}" 2>/dev/null; }
  # EXIT alone is not enough: an untrapped SIGTERM or SIGHUP kills the shell
  # without running it (measured; SIGINT does run it).
  # Known Issue: kill -9 CI script (rarely needed) will leave stale lock file
  # under /dev/shm. Users should terminate manually by inspecting ps.
  trap '_stop_watchdog' EXIT
  trap '_stop_watchdog; exit 130' INT
  trap '_stop_watchdog; exit 143' TERM HUP
  # One invocation over the whole suite dir (conftest.py sets up sys.path); pytest
  # collects test_backward / test_varlen together (test_forward.py is excluded via
  # conftest.py's collect_ignore - its coverage is a subset of test_backward.py's).
  pytest --tb=line -n ${ngpus} --max-worker-restart 9999 -rfEsx \
    -p no:cacheprovider \
    ${SELECT_FROM} \
    modules/flash/tests \
    -v \
    1>>"${outdir}/${fnprefix}${pass}.out" \
    2>>"${_errfile}" || true
  grep '^FAILED' "${outdir}/${fnprefix}${pass}.out"|sed 's/^FAILED //' | sed 's/].*/]/' > "${outdir}/sel${pass}.txt"
  if [ -n "${RECORD_ADIFFS_TO:-}" ]; then
    SCRIPT_DIR_ABS="$(cd "${SCRIPT_DIR}" && pwd)"
    bash "${SCRIPT_DIR_ABS}/../.tune/bin/append_oom_to_adiffs.sh" "${outdir}/${fnprefix}${pass}.out" >> "${RECORD_ADIFFS_TO}"
  fi
)
