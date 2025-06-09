# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

@dataclass
class CSVTranslator:
    colunm : str = ''
    iface_param : str = ''
    value_translator = None
