# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

class DirectKernelArguments(object):
    NAME = None
    INCLUDE = None
    NAMESPACE = None

    def __init__(self):
        pass

    @property
    def full_name(self):
        return f'{self.NAMESPACE}::{self.NAME}'
