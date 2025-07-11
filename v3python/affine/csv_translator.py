# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Callable

@dataclass
class CSVTranslator:
    column : str = ''
    iface_param : str = ''
    value_translator : Callable = None  # Either Callable[Functional, AnyValue], or Callable[AnyValue]

    def get_iface_param(self):
        return self.column if not self.iface_param else self.iface_param

    def translate_tc(self, tc, *, functional=None):
        value = tc.triton_compile_signature
        if self.value_translator is None:
            return value
        if functional is not None:
            return self.value_translator(functional, value)
        return self.value_translator(value)

    def translate_csv_property(self, df, *, functional=None):
        value = df[self.column].iat[0]
        if self.value_translator is None:
            return value
        if functional is not None:
            return self.value_translator(functional, value)
        return self.value_translator(value)
