#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Runs on a fleet peer, over ssh, from .perfmon/bin/setup_session (automation
# doc 5.3, step A1). Idempotent; each check stops with a message at the first
# failure.
#
# Usage: setup_peer.sh <workdir> <scratch> <rocm> [<arch hint>]
#   <arch hint>  used to pick the ROCm device wheel when the peer cannot list
#                its own arch yet (the venv, which brings rocminfo, comes later)

set -o pipefail

PM_WORKDIR="$1"
PM_SCRATCH="$2"
PM_ROCM="$3"
ARCH_HINT="${4:-}"
export PM_WORKDIR PM_SCRATCH PM_ROCM

. "$(dirname "${BASH_SOURCE[0]}")/node_env.sh"

ROCM_INDEX="https://repo.amd.com/rocm/whl-multi-arch/"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}
ok() {
    echo "ok: $*"
}

# Arch names of the GPU agents, one per line, from whichever ROCm tool exists.
peer_arches() {
    if command -v rocminfo >/dev/null 2>&1; then
        rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | grep -v '^gfx000' | sort -u
    elif command -v rocm_agent_enumerator >/dev/null 2>&1; then
        rocm_agent_enumerator 2>/dev/null | grep -v '^gfx000' | sort -u
    fi
}

# 1. The checkout and the scratch are visible, at the same paths as on the
#    control host.
[ -f "$PM_CHECKOUT/.def/container/worker_service.sh" ] \
    || fail "the checkout is not visible on $(hostname) at $PM_CHECKOUT"
ok "checkout visible at $PM_CHECKOUT"
[ -d "$PM_SCRATCH" ] || fail "the scratch $PM_SCRATCH is not visible on $(hostname)"
[ -w "$PM_SCRATCH" ] || fail "the scratch $PM_SCRATCH is not writable on $(hostname)"
ok "scratch visible and writable at $PM_SCRATCH"
[ -f "$PM_WORKDIR/config.rc" ] || fail "the workdir $PM_WORKDIR (config.rc) is not visible on $(hostname)"
ok "workdir visible at $PM_WORKDIR"

# 2. The GPU: /dev/kfd, and (once ROCm is in reach) rocminfo listing an arch.
[ -e /dev/kfd ] || fail "/dev/kfd does not exist on $(hostname): not a GPU container (claudepak --gpu)"
[ -r /dev/kfd ] && [ -w /dev/kfd ] || fail "/dev/kfd is not accessible to $(id -un) (groups: $(id -Gn))"
ok "/dev/kfd accessible"

VENV="$(perfmon_venv_dir)"
mkdir -p "$PM_SCRATCH/perfmon/venv" "$PM_SCRATCH/perfmon/logs" "$PM_SCRATCH/perfmon/handoff"

# 3. The shared venv, created once per ROCm. Two peers may set up at the same
#    time on the shared scratch, so the creation is serialised with a lock
#    directory (mkdir is atomic).
LOCK="$VENV.lock"
LOCKED=0
acquire_lock() {
    local waited=0
    while ! mkdir "$LOCK" 2>/dev/null; do
        if [ "$waited" -ge 1800 ]; then
            fail "venv lock $LOCK held for 30 min; if no other setup is running, remove it"
        fi
        [ "$waited" -eq 0 ] && echo "waiting for another setup holding $LOCK ..."
        sleep 5
        waited=$((waited + 5))
    done
    LOCKED=1
    trap 'rmdir "$LOCK" 2>/dev/null' EXIT
}

ARCH_NOW="$(peer_arches | head -1)"
ARCH="${ARCH_NOW:-$ARCH_HINT}"
[ -n "$ARCH" ] || fail "cannot tell this peer's arch (no rocminfo yet, no --arch hint, host not in workers.db)"

if [ ! -f "$VENV/bin/activate" ]; then
    acquire_lock
    if [ ! -f "$VENV/bin/activate" ]; then
        echo "creating $VENV ..."
        rm -rf "$VENV"
        python3 -m venv "$VENV" || fail "python3 -m venv $VENV failed"
        "$VENV/bin/python" -m pip install -q --upgrade pip setuptools wheel \
            || fail "pip could not reach its index (is pypi.org allowlisted?)"
    fi
fi
. "$VENV/bin/activate"

# ROCm wheels: the SDK once, then the device libraries of every arch a peer
# needs (one venv serves every arch of this ROCm).
if ! command -v rocm-sdk >/dev/null 2>&1 || ! rocm-sdk targets 2>/dev/null | grep -qw "$ARCH"; then
    [ "$LOCKED" -eq 1 ] || acquire_lock
    echo "installing rocm[libraries,devel,device-$ARCH]==$PM_ROCM from $ROCM_INDEX ..."
    python -m pip install -q --extra-index-url "$ROCM_INDEX" \
        "rocm[libraries,devel,device-$ARCH]==$PM_ROCM" \
        || fail "the ROCm $PM_ROCM wheels for $ARCH did not install (is repo.amd.com allowlisted?)"
fi
perfmon_venv_activate || exit 1
ok "venv $VENV (ROCm $(rocm-sdk version 2>/dev/null | head -1))"

# Now rocminfo exists: the real arch check.
ARCHES="$(peer_arches)"
[ -n "$ARCHES" ] || fail "rocminfo lists no GPU arch on $(hostname) (in-venv ROCm, /dev/kfd present)"
ok "GPU arch(es): $(echo $ARCHES)"
case " $ARCHES " in
    *" $ARCH "*) ;;
    *) fail "the device wheel was chosen for $ARCH, but rocminfo lists: $(echo $ARCHES)" ;;
esac

# 4. aotriton from the checkout (rootless, into the shared venv), pytest, amd-smi.
#    install_aotriton_pkg.sh reinstalls the package from the checkout every
#    run, so re-running setup_session after a code change refreshes the venv.
[ "$LOCKED" -eq 1 ] || acquire_lock
AOTRITON_SRC="$PM_CHECKOUT" bash "$PM_CHECKOUT/.def/container/install_aotriton_pkg.sh" \
    || fail "installing aotriton from $PM_CHECKOUT failed"
python -c 'import pytest' 2>/dev/null || python -m pip install -q pytest \
    || fail "pytest did not install"
if ! python -c 'import amdsmi' 2>/dev/null; then
    # The thermal gate needs it; the wheels may already carry it.
    python -m pip install -q amdsmi 2>/dev/null \
        || echo "warning: amdsmi is not importable; AOTRITON_THERMAL_GATE=on will not work here" >&2
fi
python -c 'import aotriton.perfmon.manifest, pytest' \
    || fail "the venv cannot import aotriton.perfmon and pytest"
ok "aotriton, pytest importable from $VENV"

# 5. What a node needs.
echo
echo "Node environment on $(hostname):"
perfmon_env_setup
perfmon_env_print
