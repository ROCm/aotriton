#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

# /workspaces must already be a writable PVC mount.
bash .devcontainer/configure-trust.sh
install -d -m 700 "${CODEX_HOME:-/workspaces/.codex}"
python .devcontainer/doctor.py --static

# Keep the workspace usable while troubleshooting Python dependencies.
if [[ "${AOTRITON_DEV_INSTALL:-1}" == "0" ]]; then
    echo "AOTriton setup skipped (AOTRITON_DEV_INSTALL=0)."
    exit 0
fi

git submodule update --init --recursive
python -m pip install -r requirements-dev.txt -r requirements-tuning.txt
# This installs the Python code generator only. The HIP library, pyaotriton,
# pinned compiler and gfx908 kernel images are built separately with CMake.
python -m pip install --no-build-isolation --no-deps -e . -e ./python/tune

echo "AOTriton development dependencies and editable code generator installed."
echo "Run: python .devcontainer/doctor.py"
echo "Build instructions: .devcontainer/README.md"
echo "Then sign into Codex with: codex login --device-auth"
