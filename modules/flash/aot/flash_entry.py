# Copyright © 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The FlashEntry formatting struct used by the LUT-missing-entry diagnostic.

This is the codegen-side subset of v3python.tune.flash.module.FlashEntry: the
dataclass fields and `as_text()` (the only surface `_gen_missing_entries` uses).
The tuning-side parse/dacite helpers stay in v3python.tune; keeping this copy free
of v3python.tune is what severs the last v3python edge from the codegen closure.

`as_text()` output MUST stay byte-identical to the original (it is the diagnostic
line the tuner re-parses).

TODO: Merge with modules/flash/tune in ATI Phase 2: Modularization.
"""

from dataclasses import dataclass, asdict


@dataclass
class FlashEntry:
    dtype: str = 'float16'
    hdim: int | tuple[int, int] = 16  # tuple[int, int] for hdim_qk != hdim_v
    seqlen_q: int = 16
    seqlen_k: int = 16
    causal: bool | tuple[int, int] = 0
    dropout_p: float = 0.0
    bias_type: int = 0

    def as_text(self) -> str:
        def tr(v) -> str:
            if isinstance(v, str):
                return f"'{v}'"
            if isinstance(v, tuple):
                return '(' + ','.join(tr(x) for x in v) + ')'
            if isinstance(v, list):
                return '[' + ','.join(tr(x) for x in v) + ']'
            return str(v)
        return ';'.join([f"{k}={tr(v)}" for k, v in asdict(self).items()])
