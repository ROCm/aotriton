# Copyright © 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from aotriton.tune.utils import parse_python
from dataclasses import dataclass, asdict

'''
CAVEAT about imports
FlashEntry/FlashInputMetadata are dual purpose classes, which may not have
torch/pyaotriton package installed in the environment.

Any GPU related imports must be deferred to the related function instead import
at the beginning of the file.

`dacite` is likewise an optional, tuning-only dependency (see
python/test/test_tune_infra.py's module docstring) -- `from_dict()` below
imports it lazily, inside the method, so merely importing this module (e.g.
via registry.load_flash_entry_module(), used by non-GPU tools like
.tune/webui/tasks.py) never requires dacite to be installed.
'''

@dataclass
class FlashEntry:
    dtype: str = 'float16'
    hdim: int | tuple[int, int] = 16  # tuple[int, int] for hdim_qk != hdim_v
    seqlen_q: int = 16
    seqlen_k: int = 16
    causal: bool | tuple[int, int] = 0
    dropout_p: float = 0.0
    bias_type: int = 0

    @staticmethod
    def parse_text(line: str) -> "FlashEntry":
        d = parse_python(line)
        return FlashEntry(**d)

    @staticmethod
    def from_dict(d: dict) -> "FlashEntry":
        from dacite import from_dict
        from aotriton.tune.utils import dacite_tuple
        return from_dict(data_class=FlashEntry, data=d, config=dacite_tuple)

    def as_posix(self) -> str:
        return ','.join([f"{k}={v}" for k, v in asdict(self).items()])

    def as_text(self) -> str:
        # KEEP BYTE-IDENTICAL to modules/flash/aot/flash_entry.py's FlashEntry.as_text()
        # (codegen-side copy, torch-free -- see that file's docstring, and
        # python/test/test_tune_infra.py's test_flash_entry_as_text_matches_codegen_copy).
        def tr(v) -> str:
            if isinstance(v, str):
                return f"'{v}'"
            if isinstance(v, tuple):
                return '(' + ','.join(tr(x) for x in v) + ')'
            if isinstance(v, list):
                return '[' + ','.join(tr(x) for x in v) + ']'
            return str(v)
        return ';'.join([f"{k}={tr(v)}" for k, v in asdict(self).items()])

    @property
    def qkh(self):
        return self.seqlen_q * self.seqlen_k * self.hdim

# Field names match mptune/flash/tuner.py and/or _core_test_backward.py
@dataclass
class FlashInputMetadata(FlashEntry):
    N_HEADS: int | tuple[int, int] = 5
    BATCH: int = 3
    sm_scale: str | float = 'l1'
    storage_flip: bool | tuple[int, int] = False
    prng_seed: int = 0x9be9_98d4_cf17_5339

    @staticmethod
    def parse_text(line: str) -> "FlashEntry":
        d = parse_python(line)
        return FlashInputMetadata(**d)

    @staticmethod
    def from_dict(d: dict) -> "FlashInputMetadata":
        from dacite import from_dict
        from aotriton.tune.utils import dacite_tuple
        return from_dict(data_class=FlashInputMetadata, data=d, config=dacite_tuple)
