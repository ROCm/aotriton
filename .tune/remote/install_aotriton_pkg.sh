#!/bin/bash
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Install the `aotriton` package into the active venv, from a throwaway copy of
# the checkout. Runs INSIDE the worker container, with the venv already
# activated, immediately before worker_service.sh starts anything.
#
# Why it has to happen at every worker start, and not once at image build time:
# the checkout is bind-mounted at /wkdir/aotriton.src and changes under the
# image. An `aotriton` baked into the image is a snapshot of whatever the tree
# looked like when the image was built, so a deploy that syncs new code to the
# host would leave the workers still importing the old one -- silently, since
# both versions import fine. Installing here means the package always matches
# the checkout the worker is about to run out of.
#
# Why it copies instead of installing from /wkdir/aotriton.src directly:
# pip builds in-tree (the default since 21.3; the out-of-tree option was
# removed in 22.1), so `pip install .` drops build/ and aotriton.egg-info/ into
# the source directory. Two problems with that here. They show up in
# `git status` on every remote host, which they are not ours to leave behind;
# and this container runs as root while the checkout belongs to the invoking
# user, so those directories come back root-owned and the user cannot clean or
# overwrite them without help.
#
# The perfmon image solves the same problem with DIST_EXTRA_CONFIG, pointing
# setuptools' build_base and egg_base at scratch/. That works there because
# that container runs as the invoking uid:gid. This one is still root -- tuning
# has no user-owned venv to protect, works as root today, and moving it would
# put /dev/kfd and /dev/dri access behind untested group membership -- so the
# artifacts have to land somewhere the host cannot see at all, not merely
# somewhere tidier. Container-local /tmp is exactly that.

set -euo pipefail

SRC="${1:-/wkdir/aotriton.src}"
STAGE="$(mktemp -d /tmp/aotriton-pkg.XXXXXX)"
STAGE2="$(mktemp -d /tmp/aotriton-tune-pkg.XXXXXX)"
trap 'rm -rf "$STAGE" "$STAGE2"' EXIT

# Everything `pip install .` reads, and nothing else:
#   pyproject.toml   build-system requirements
#   setup.py         the python/ -> aotriton package_dir mapping
#   CMakeLists.txt   setup.py parses the version out of it (single source of
#                    truth), and raises RuntimeError if it cannot
#   python/          the package tree itself, plus its package_data
#
# Listed explicitly rather than copying the whole checkout: the tree also holds
# third_party/ and .git, none of which the install reads.
for item in pyproject.toml setup.py CMakeLists.txt python; do
  if [ ! -e "$SRC/$item" ]; then
    echo "Error: $SRC/$item not found; cannot build the aotriton package." >&2
    exit 1
  fi
  cp -a "$SRC/$item" "$STAGE/"
done

# --no-deps because the image already installed requirements-tuning.txt at
# build time. Without it, every worker start would re-resolve the full
# dependency set against the network before the worker could come up.
#
# --no-build-isolation for the same reason, one level down. --no-deps does not
# cover the BUILD dependencies: by default pip builds in an isolated
# environment and downloads pyproject.toml's `requires` (setuptools>=64) into
# it from PyPI, every time. That is a network round trip on every worker start,
# and it fails outright on a host that cannot reach PyPI -- which is how this
# first surfaced: "pip subprocess to install build dependencies did not run
# successfully". The image already has setuptools and wheel from
# requirements.txt, so building against them is both correct and offline.
python -m pip install -q --no-deps --no-build-isolation "$STAGE"

# aotriton.tune (python/tune/) is a separate distribution, not in the wheel
# above -- install it too, from its own staging copy. Its setup.py reads
# CMakeLists.txt two parents up, hence $STAGE2's layout below.
mkdir -p "$STAGE2/python"
cp -a "$SRC/CMakeLists.txt" "$STAGE2/"
cp -a "$SRC/python/tune" "$STAGE2/python/"
python -m pip install -q --no-deps --no-build-isolation "$STAGE2/python/tune"

# Report what the worker will import. `aotriton` is a namespace package (no
# __file__), so report its merged search path(s), plus that aotriton.tune
# resolved.
python - <<'PY'
import aotriton
print(f"aotriton namespace path(s): {list(aotriton.__path__)}")
import aotriton.tune
print(f"aotriton.tune installed: {aotriton.tune.__file__}")
PY
