#!/bin/bash

set -ex

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
root_cmake="${SCRIPT_DIR}/../CMakeLists.txt"

aotriton_major=$(grep 'set(AOTRITON_VERSION_MAJOR_INT' "${root_cmake}"|cut -d ' ' -f 2|cut -d ')' -f 1)
aotriton_minor=$(grep 'set(AOTRITON_VERSION_MINOR_INT' "${root_cmake}"|cut -d ' ' -f 2|cut -d ')' -f 1)
native_arch=$(rocm_agent_enumerator|grep -v gfx000|head -n 1)
ngpus=$(rocm_agent_enumerator|grep -v gfx000|wc -l)
small_vram=$(amd-smi static -g 0 -v --json| python -c 'import json, sys; j = json.load(sys.stdin); print(int(j["gpu_data"][0]["vram"]["size"]["value"] / 1024.0 < 60))')
