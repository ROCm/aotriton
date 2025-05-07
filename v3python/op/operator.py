# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from ..base import (
    Interface
)

class Operator(Interface):
    TUNE_NAME = 'optune'

    @property
    def enum_name(self):
        # CamelName = self.NAME.replace('_', ' ').title().replace(' ', '')
        return f'kOp_{self.class_name_base}'
