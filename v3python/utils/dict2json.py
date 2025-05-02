# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

def dict2json(d) -> str:
    import json
    print(f'{d=}')
    return json.dumps(d)
