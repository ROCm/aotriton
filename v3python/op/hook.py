# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

class Hook(object):
    def __init__(self, function_name, target, cookie='cookie'):
        self._function = function_name
        self._target = target
        self._cookie = cookie

    @property
    def function(self):
        return self._function

    @property
    def target(self):
        return self._target

    @property
    def cookie(self):
        return self._cookie
