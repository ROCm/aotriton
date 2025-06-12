# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Callable

@dataclass
class CSVTranslator:
    column : str = ''
    iface_param : str = ''
    value_translator : Callable = None

    def get_iface_param(self):
        return self.column if not self.iface_param else self.iface_param

    def translate_tc(self, tc):
        value = tc.triton_compile_signature
        if self.value_translator is None:
            return value
        return self.value_translator(value)
