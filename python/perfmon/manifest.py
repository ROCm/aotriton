# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The perfmon application's manifest, queried by the shared node and fleet
scripts with

    python3 -m aotriton.perfmon.manifest <key> [--arch A]

(see `.def/lib/manifest.sh`). perfmon has no second category level: every
node serves every perfmon task of its arch. Image keys come with the fleet
work.
"""

import sys

from aotriton.pgdef.manifest import main

MANIFEST = {
    'app': 'perfmon',
    'category': {
        'level1_values': [],
    },
    'install': {
        # What `deploy` syncs to a worker, relative to the workdir: the core
        # (one per ROCm, shared by every arch) and the arch's subjects.
        'subtree': ['installed/perfmon/core', 'installed/perfmon/{arch}'],
    },
    'deploy': {
        # the workers.db role of the hosts that run this application
        'role': 'Perfmon',
    },
    'fetchbuild': {
        # relative to the build node's installed/ directory
        'path': 'perfmon/',
        'exclude': [],
    },
}


if __name__ == '__main__':
    sys.exit(main(MANIFEST, sys.argv[1:], prog='python3 -m aotriton.perfmon.manifest'))
