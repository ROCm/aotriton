# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

__all__ = [
    "Factories",
]

# Imported lazily: factories pulls pandas and the ATI IR, which the tensor
# file interpreter (python -m aotriton.database.interpreter) does not need.
def __getattr__(name):
    if name == 'Factories':
        from .factories import Factories
        return Factories
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
