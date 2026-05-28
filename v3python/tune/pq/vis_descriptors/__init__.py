# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from .flash import FLASH_DESCRIPTOR

# Registry: id -> descriptor dict
DESCRIPTORS: dict[str, dict] = {
    FLASH_DESCRIPTOR['id']: FLASH_DESCRIPTOR,
}
