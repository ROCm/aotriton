# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass


@dataclass(frozen=True)
class ExactTuning:
    """A partial profile: exact key matches, with the default image at index 0.

    Families opt in per functional and may further restrict runtime coverage
    using a C++ expression over the kernel's `params` struct. Missing key values
    and missing combinations of known values select the default image.
    """

    runtime_condition: str = 'true'
