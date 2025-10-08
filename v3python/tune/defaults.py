# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

_default_device = 'cuda'

def default_device():
    return _default_device

def set_default_device(device_id, *, device_type='cuda'):
    global _default_device
    _default_device = f'{device_type}:{device_id}'
