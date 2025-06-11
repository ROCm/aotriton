# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

@dataclass
class CSVTranslator:
    column : str = ''
    iface_param : str = ''
    value_translator = None

    def get_iface_param(self):
        return self.column if self.iface_param is None else self.iface_param

    def translate_tc(self, tc):
        value = tc.triton_compile_signature
        if self.value_translator is None:
            return value
        return self.value_translator(value)
