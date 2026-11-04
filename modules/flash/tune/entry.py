# Copyright © 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from aotriton.utils import parse_pon, render_pon
from dataclasses import dataclass, asdict

'''
CAVEAT about imports
FlashEntry/FlashInputMetadata are dual purpose classes, which may not have
torch/pyaotriton package installed in the environment.

Any GPU related imports must be deferred to the related function instead import
at the beginning of the file.

`dacite` is likewise an optional, tuning-only dependency (see
python/tune/tests/test_tune_infra.py's module docstring) -- `from_dict()` below
imports it lazily, inside the method, so merely importing this module (e.g.
via registry.load_flash_entry_module(), used by non-GPU tools like
.def/webui/tasks.py) never requires dacite to be installed.
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
        d = parse_pon(line)
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
        # python/tune/tests/test_tune_infra.py's test_flash_entry_as_text_matches_codegen_copy).
        return render_pon(asdict(self))

    @property
    def qkh(self):
        return self.seqlen_q * self.seqlen_k * self.hdim

# ---------------------------------------------------------------------------
# Input sizes: accuracy tests versus benchmarks
#
# The tuner uses two different (BATCH, N_HEADS) sets for one database entry.
#
# ACCURACY TESTS use TEST_BATCH x TEST_N_HEADS = 3 x 5 (the FlashInputMetadata
# defaults below). Both are prime on purpose: a wrong batch or head stride
# cannot cancel against a power-of-two extent, so indexing and stride errors
# show up as a wrong result.
#
# BENCHMARKS use benchmark_shape(seqlen_q). Timing at 3 x 5 is misleading for
# short sequences. batch x heads x seqlen_q is the number of query rows the
# GPU has to work on; at 3 x 5 x 128 the whole launch is a few thousand rows,
# the GPU is mostly idle, and the timing is dominated by launch overhead. The
# tuner then prefers the candidates (kernel configs, and for the op level the
# backend) that win on an idle GPU, and they lose at real sizes. Measured on
# gfx950: the crossover is around batch x heads ~ 50, while real workloads run
# batch x heads of 864-8192 at seqlen 128-512. The gfx950 fp16 hdim 64 dropout
# .dbpatch fixes at large batch came from this; they cost up to 0.08 ms at
# batch x heads <= 96 (23 of 150 entries measured), which is accepted:
# small batch with short sequences is under-utilized by nature.
#
# Why the database is not keyed by batch and heads: a (batch, heads, seqlen_q,
# seqlen_k) table is a rank-4 tensor on top of the dtype/hdim/causal/dropout/
# bias axes, which the dense tuning tensor cannot afford. A decision tree over
# batch x heads could work but is non-trivial, and it is not worth building
# since the Triton kernels are being retired. So one benchmark size per
# seqlen_q is chosen to represent the large-batch regime.
# ---------------------------------------------------------------------------
TEST_BATCH = 3
TEST_N_HEADS = 5

# batch x heads is chosen so that batch x heads x seqlen_q reaches
# BENCH_QUERY_ROWS (2**17), clamped to [TEST_BATCH * TEST_N_HEADS,
# BENCH_MAX_BATCH_HEADS]:
#
#   seqlen_q          16..128  256   512   1024  2048  4096  8192
#   batch x heads       1024   512   256   128    64    32    16
#   (batch, n_heads)  128x8   64x8  32x8  16x8   8x8   4x8   2x8
#
# Where batch x heads x seqlen_q already fills the GPU the floor keeps the
# old 3 x 5. N_HEADS is the realistic BENCH_N_HEADS = 8; GQA entries keep
# their (q heads, kv heads) ratio and scale batch only. The memory clamp
# (FlashTune._clamp_memory_usage) still applies on top, so long sequences
# with a large reference are shrunk exactly as before.
BENCH_QUERY_ROWS = 1 << 17
BENCH_MAX_BATCH_HEADS = 1024
BENCH_N_HEADS = 8


def benchmark_shape(seqlen_q: int,
                    n_heads: int | tuple[int, int] = TEST_N_HEADS) -> tuple[int, int | tuple[int, int]]:
    """(BATCH, N_HEADS) to time a database cell with `seqlen_q` query rows.

    `n_heads` is the heads of the case being timed: an int, or a (q, kv) pair
    for GQA, in which case the ratio is kept and only BATCH is scaled."""
    floor = TEST_BATCH * TEST_N_HEADS
    target = BENCH_QUERY_ROWS // seqlen_q
    target = max(floor, min(BENCH_MAX_BATCH_HEADS, target))
    if target <= floor:
        return TEST_BATCH, n_heads
    if isinstance(n_heads, int):
        return -(-target // BENCH_N_HEADS), BENCH_N_HEADS
    return -(-target // n_heads[0]), n_heads


# Field names match mptune/flash/tuner.py and/or _core_test_backward.py
@dataclass
class FlashInputMetadata(FlashEntry):
    # Accuracy-test sizes. Benchmark sizes come from benchmark_shape().
    N_HEADS: int | tuple[int, int] = TEST_N_HEADS
    BATCH: int = TEST_BATCH
    sm_scale: str | float = 'l1'
    storage_flip: bool | tuple[int, int] = False
    prng_seed: int = 0x9be9_98d4_cf17_5339

    @staticmethod
    def parse_text(line: str) -> "FlashEntry":
        d = parse_pon(line)
        return FlashInputMetadata(**d)

    @staticmethod
    def from_dict(d: dict) -> "FlashInputMetadata":
        from dacite import from_dict
        from aotriton.tune.utils import dacite_tuple
        return from_dict(data_class=FlashInputMetadata, data=d, config=dacite_tuple)
