# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parent.parent.parent

# We use [[ ]] instead of { } for C++ code template
def get_template(name):
    with open(SOURCE_ROOT / 'v2src' / 'template' / name, 'r') as f:
        return f.read().replace('{', '{{').replace('}', '}}').replace('[[', '{').replace(']]', '}')
