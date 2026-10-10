# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path

CODEGEN_ROOT = Path(__file__).resolve().parent

# We use [[ ]] instead of { } for C++ code template
def get_template(name):
    with open(CODEGEN_ROOT / 'template' / name, 'r') as f:
        return f.read().replace('{', '{{').replace('}', '}}').replace('[[', '{').replace(']]', '}')
