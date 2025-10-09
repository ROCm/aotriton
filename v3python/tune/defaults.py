# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

_default_device_type: str = 'cuda'
_default_device_id: int = 0

def default_device_type() -> str:
    return _default_device_type

def default_device_id() -> int:
    return _default_device_id

def default_device_string() -> str:
    return f'{_default_device_type}:{_default_device_id}'

def set_default_device(device_id: int, *, device_type='cuda'):
    global _default_device_type
    global _default_device_id
    _default_device_type = device_type
    _default_device_id = device_id
