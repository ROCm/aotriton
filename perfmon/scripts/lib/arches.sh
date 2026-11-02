# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# The MAJOR arch list and the subject pool of an arch. Source it.
#
# perfmon builds one subject per release for the five major arches together,
# under installed/perfmon/MAJOR/, and one per release for every other arch,
# under installed/perfmon/<arch>/ (for external users). Everything that
# locates a subject maps a device arch to its pool with perfmon_subject_pool;
# the Python side does the same (python/perfmon/exaid_perfmon.py's
# subject_pool). The list is defined once, as AOTRITON_MAJOR_ARCHES in
# python/gpu_targets.py, and read here as a literal through `ast`.

PERFMON_MAJOR_POOL="MAJOR"

_PERFMON_ARCHES_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Prints the major arches, space-separated.
perfmon_major_arches() {
  python3 -I - "${_PERFMON_ARCHES_LIB_DIR}/../../../python/gpu_targets.py" <<'PYEOF'
import ast, sys
tree = ast.parse(open(sys.argv[1]).read())
for node in tree.body:
    if isinstance(node, ast.Assign) and any(
            getattr(t, 'id', None) == 'AOTRITON_MAJOR_ARCHES' for t in node.targets):
        print(' '.join(ast.literal_eval(node.value)))
        break
else:
    sys.exit('AOTRITON_MAJOR_ARCHES not found in ' + sys.argv[1])
PYEOF
}

# Prints the arches of AOTRITON_SUPPORTED_GPUS, space-separated, without the
# _mod<n> suffix.
perfmon_supported_arches() {
  python3 -I - "${_PERFMON_ARCHES_LIB_DIR}/../../../python/gpu_targets.py" <<'PYEOF'
import ast, sys
tree = ast.parse(open(sys.argv[1]).read())
for node in tree.body:
    if isinstance(node, ast.Assign) and any(
            getattr(t, 'id', None) == 'AOTRITON_SUPPORTED_GPUS' for t in node.targets):
        seen = []
        for gpu in ast.literal_eval(node.value):
            arch = gpu.split('_mod')[0]
            if arch not in seen:
                seen.append(arch)
        print(' '.join(seen))
        break
else:
    sys.exit('AOTRITON_SUPPORTED_GPUS not found in ' + sys.argv[1])
PYEOF
}

# perfmon_subject_pool <device arch> -- MAJOR for a major arch, else the arch.
perfmon_subject_pool() {
  local arch="$1" major
  major=" $(perfmon_major_arches) " || return 1
  case "${major}" in
    *" ${arch} "*) printf '%s\n' "${PERFMON_MAJOR_POOL}" ;;
    *) printf '%s\n' "${arch}" ;;
  esac
}
