#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""gfx1201 dQ launcher: VarlenBits bit 18 reaches the kernel argument, no GPU.

Builds the launcher and intercepts `abi.run_compiled`, so nothing is compiled
or launched: it checks only that the word's bit 18 (`principal_diagonal_choice`)
reaches the `varlen_bits` kernel argument unchanged, for a dense word and a
varlen dict alike.

gfx1201-specific because it reads the argument at gfx1201's position in the
packed kernel arguments. The numerics are `tests/test_bwd_dq_principal_diagonal.py`.
Needs FlyDSL and `AOTRITON_FLYDSL_KERNEL_ROOT`, not a GPU.

    AOTRITON_FLYDSL_KERNEL_ROOT=<flydsl-checkout> \\
        python -m pytest modules/flash/flyc/gfx1201/test_bwd_dq_launcher.py
"""

import os
import sys
from pathlib import Path

import pytest

_KERNEL_ROOT = os.environ.get("AOTRITON_FLYDSL_KERNEL_ROOT")
if _KERNEL_ROOT and _KERNEL_ROOT not in sys.path:
    sys.path.append(_KERNEL_ROOT)

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not _KERNEL_ROOT, reason="needs AOTRITON_FLYDSL_KERNEL_ROOT")

HEAD_DIM = 64
NUM_HEADS = 2
DTYPE = torch.float16

# `varlen_bits` is the fourteenth kernel argument: Q K V B DO DQ DB LSE Delta,
# the four seqinfo pointers, then the word.
_VARLEN_BITS_ARG = 13


@pytest.fixture(scope="module")
def dq_fn():
    import tuning_dq
    from fmha_meta import FmhaInputMetadata

    meta = FmhaInputMetadata(num_heads=NUM_HEADS,
                             head_dim=HEAD_DIM,
                             dtype_str="f16",
                             causal_type=3)
    meta, knobs = tuning_dq.plan("gfx1201", meta)
    return tuning_dq.builder("gfx1201", meta, knobs)


@pytest.fixture(scope="module")
def abi():
    import fmha_utils

    return fmha_utils


def _bits(abi, p):
    return abi.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT if p else abi.VARLEN_PRINCIPAL_DIAGONAL_TOP_LEFT


def _captured_varlen_bits(abi, dq_fn, monkeypatch, *, varlen=None):
    captured = []
    monkeypatch.setattr(abi, "run_compiled", lambda cache, exe, *args: captured.append(args))
    sq = 64
    dense = varlen is None or varlen.get("seqinfo_q0") is None
    batch, num_seqlens = (1, 0) if dense else (1, int(varlen["seqinfo_q0"].numel()) - 1)
    q = torch.zeros(batch, NUM_HEADS, sq, HEAD_DIM, dtype=DTYPE)
    rows = torch.zeros(batch * NUM_HEADS if dense else NUM_HEADS, sq)
    dq_fn(q,
          q,
          q,
          q,
          q.clone(),
          rows,
          rows.clone(),
          batch,
          sq,
          seqlen_k=sq,
          num_seqlens=num_seqlens,
          window=(8, 0),
          varlen=varlen)
    (args,) = captured
    return int(args[_VARLEN_BITS_ARG])


@pytest.mark.parametrize("p", (None, 0, 1))
def test_launcher_dense_word_bit(abi, dq_fn, monkeypatch, p):
    """`varlen=None` is word 0; a dense word reaches the kernel as built."""
    varlen = None if p is None else abi.varlen_dense(principal_diagonal=_bits(abi, p))
    got = _captured_varlen_bits(abi, dq_fn, monkeypatch, varlen=varlen)
    assert got == (abi.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT if p else 0)


@pytest.mark.parametrize("carried", (0, 1))
def test_launcher_varlen_bit(abi, dq_fn, monkeypatch, carried):
    """A varlen dict's bit 18 reaches the kernel, and the layout bits with it."""
    cu = torch.tensor([0, 32, 64], dtype=torch.int32)
    varlen = abi.varlen_compact(cu, cu, 32, 32, lse_tokens=64)
    layout = varlen["bits"]
    varlen["bits"] = abi.varlen_bits(abi.VARLEN_COMPACT_SIDE,
                                     abi.VARLEN_COMPACT_SIDE,
                                     principal_diagonal=_bits(abi, carried))
    got = _captured_varlen_bits(abi, dq_fn, monkeypatch, varlen=varlen)
    assert got == layout | _bits(abi, carried)


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v"]))
