# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""disable_core_dumps(): RLIMIT_CORE is (0, 0) afterwards, and children inherit it."""

import subprocess
import sys

CODE = '''
import resource
import subprocess
import sys
from aotriton.pgdef.localq.node import disable_core_dumps

disable_core_dumps()
assert resource.getrlimit(resource.RLIMIT_CORE) == (0, 0)
child = subprocess.run([sys.executable,
                        '-c',
                        'import resource; print(resource.getrlimit(resource.RLIMIT_CORE))'],
                       capture_output=True,
                       text=True,
                       check=True)
print(child.stdout.strip())
'''


def test_disable_core_dumps_inherited():
    out = subprocess.run([sys.executable, '-c', CODE],
                         capture_output=True,
                         text=True,
                         check=True)
    assert out.stdout.strip() == '(0, 0)'
