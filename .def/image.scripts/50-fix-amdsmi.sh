#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Collapse the two copies of libamd_smi.so, so torch.cuda.device_count() != 0.
#
# On this ROCm SDK layout a worker sees every GPU through the HIP runtime --
# torch.cuda.is_available() is True -- while torch.cuda.device_count() returns
# 0. A tuning worker that believes it has no GPUs has nothing to schedule, so
# the whole node is idle rather than obviously broken.
#
# Two distinct files are involved. torch/cuda/__init__.py's _amdsmi_cdll_hook
# ctypes-loads $ROCM_PATH/lib/libamd_smi.so, but by then
# libgoamdsmi_shim64.so.1 has already pulled in _rocm_sdk_core/lib's own
# libamd_smi.so.<major> via DT_NEEDED. The two are byte-identical and share a
# SONAME, yet they are separate inodes, so glibc maps both. Each mapping
# carries its own global state, and the ctypes one enumerates zero GPUs.
#
# Pointing the devel path at the core file makes both loads resolve to one
# inode, so only one copy is mapped and the state the ctypes hook queries is
# the state the shim initialised.
#
# Deliberately different from the runtime version of this fix in two respects,
# because this runs inside `docker build`:
#
#   * The original is MOVED aside to .orig rather than copied. A copy would
#     duplicate the library in the image layer; the move keeps it available for
#     inspection at no cost, and the file is byte-identical to the core copy it
#     is being replaced by anyway.
#   * There is no LD_PRELOAD fallback. That exists at runtime for a read-only
#     ROCm tree, and it works there because an `export` outlives the snippet
#     that set it. Here the export would die with this script and silently
#     leave the image broken, and a tree that is not writable during its own
#     image build is a build failure worth stopping on -- so a failed symlink
#     is an error, not something to paper over.
#
# Runs before 51-fix_gfx90a_torch.sh: this one decides whether a worker can see
# its GPUs at all, which is the prerequisite for anything the later script's
# kernel-loading fix is about.

set -e

CONFIG_RC="${CONFIG_RC:-/config.rc}"

if [ ! -f "$CONFIG_RC" ]; then
  echo "Error: config.rc not found at $CONFIG_RC" >&2
  exit 1
fi

# shellcheck disable=SC1090
. "$CONFIG_RC"

if [ -z "${CELERY_WORKER_PYTHON:-}" ]; then
  echo "Error: CELERY_WORKER_PYTHON not set in $CONFIG_RC" >&2
  exit 1
fi

# Resolve ROCM_PATH here rather than expecting it in the environment. The
# Dockerfile runs these scripts from a plain RUN, which is not a login shell,
# so /etc/profile.d/aotriton.sh has not been sourced and ROCM_PATH is unset.
# Ask rocm-sdk directly, exactly as that profile does.
ROCM_SDK="$(dirname "$CELERY_WORKER_PYTHON")/rocm-sdk"
if [ ! -x "$ROCM_SDK" ]; then
  echo "50-fix-amdsmi: no rocm-sdk in this image -- skipping"
  exit 0
fi
ROCM_PATH="${ROCM_PATH:-$("$ROCM_SDK" path --root)}"

_core_lib="$(dirname "$ROCM_PATH")/_rocm_sdk_core/lib"
_smi_devel="$ROCM_PATH/lib/libamd_smi.so"

# Find the core copy without pinning a SONAME version. The major differs
# between ROCm releases -- .so.27 is what ROCm 10 ships, where this was first
# diagnosed, and hardcoding it made this script skip silently on 7.14, which
# carries a different one. Whichever name matches, resolve it: every link in
# the chain (libamd_smi.so -> .so.N -> .so.N.M.P) ends at one real file, and
# that inode is the only thing this fix is about.
#
# The pattern requires a digit after .so. so it cannot pick up a .orig or
# .debug sibling.
_smi_core=""
for _cand in "$_core_lib"/libamd_smi.so.[0-9]* "$_core_lib"/libamd_smi.so; do
  [ -e "$_cand" ] || continue
  _smi_core="$(readlink -f "$_cand")"
  break
done

if [ -z "$_smi_core" ]; then
  echo "50-fix-amdsmi: no libamd_smi.so* under $_core_lib -- nothing to collapse, skipping"
  # List what is there: if the layout moved, this is the line that says so,
  # rather than leaving a silent skip to be rediscovered as device_count()==0.
  ls -1 "$_core_lib" 2>/dev/null | sed 's/^/    /' || echo "    (directory does not exist)"
  exit 0
fi

if [ "$(readlink -f "$_smi_devel" 2>/dev/null)" = "$_smi_core" ]; then
  echo "50-fix-amdsmi: $_smi_devel already resolves to the core copy -- nothing to do"
  exit 0
fi

if [ ! -L "$_smi_devel" ] && [ -e "$_smi_devel" ] && [ ! -e "$_smi_devel.orig" ]; then
  mv -- "$_smi_devel" "$_smi_devel.orig"
  echo "50-fix-amdsmi: kept the original devel copy at $_smi_devel.orig"
fi

ln -sfn "$_smi_core" "$_smi_devel"
echo "50-fix-amdsmi: $_smi_devel -> $_smi_core"
