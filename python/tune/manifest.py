# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The tune application's manifest: every piece of per-application shell knowledge
in one dict, queried by the `.tune` scripts with

    python3 -m aotriton.tune.manifest <key> [--arch A] [--cat_level1 kernel|op]

(see `.def/lib/manifest.sh`). Tuning containers still run as root and install
aotriton when the container starts; moving them to the non-root,
installed-in-the-image contract is a separate change.
"""

import sys

from aotriton.pgdef.manifest import main

MANIFEST = {
    'app': 'tune',
    'category': {
        'level1_values': ['kernel', 'op'],
    },
    'install': {
        # what `deploy` syncs to a worker, relative to the workdir
        'subtree': {
            'kernel': 'installed/{arch}',
            'op': 'installed/test/{arch}',
        },
    },
    'deploy': {
        # the workers.db role of the hosts that run this application
        'role': 'Tuner',
    },
    'fetchbuild': {
        # what to fetch from the build node, relative to its installed/ directory,
        # and what to leave out of it (rsync --exclude patterns)
        'path': {'kernel': '', 'op': 'test/'},
        'exclude': {'kernel': ['/database/', '/test/'], 'op': []},
    },
}


if __name__ == '__main__':
    sys.exit(main(MANIFEST, sys.argv[1:], prog='python3 -m aotriton.tune.manifest'))
