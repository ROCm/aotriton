# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os

AOTRITON_DEBUG_GENERATOR = bool(int(os.getenv('AOTRITON_DEBUG_GENERATOR', default='0')))

'''
Recommended logging method with f-string:
log(lambda:f_string)

See: https://discuss.python.org/t/safer-logging-methods-for-f-strings-and-new-style-formatting/13802/10
'''
def _log(*objects, sep=' ', end='\n', file=None, flush=False):
    def _translate(o):
        return o() if callable(o) and o.__name__ == "<lambda>" else o
    print(*[_translate(o) for o in objects], sep=sep, end=end, file=file, flush=flush)

def _nolog(*objects, sep=' ', end='\n', file=None, flush=False):
    pass

log = _log if AOTRITON_DEBUG_GENERATOR else _nolog
