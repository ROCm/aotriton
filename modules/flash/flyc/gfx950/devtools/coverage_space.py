# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The sweep space of the gfx950 flyc coverage sweep, and the record format.

Pure Python, no torch and no FlyDSL, so the driver and the summarizer can
enumerate and read records without paying for either import.

A **group** is everything one worker process serves: `(kernel, head_dim,
dtype, mask, feature)`. Every candidate build of the group, and every
`(layout, seqlen_q, seqlen_k)` cell, run in that one process, so compile cost
amortizes over the shapes and timing can interleave the candidates.

A **record** is one `(group, layout, seqlen_q, seqlen_k, candidate, pass)`.
`record_key` is its identity; the driver and the workers skip any key already
present in the output directory, which is what makes a sweep resumable.
"""

import glob
import json
import os

KERNELS = ("fwd", "bwd")

# Every rung of the shared ladder, plus three widths between rungs (padded heads).
LADDER = (32, 64, 96, 128, 160, 192, 224, 256, 384, 512)
OFF_RUNG = (40, 100, 200)
HEAD_DIMS = tuple(sorted(LADDER + OFF_RUNG))

DTYPES = ("bf16", "fp16")
# The knob/metadata spelling of each dtype.
DTYPE_STR = {"bf16": "bf16", "fp16": "f16"}

SEQLENS = (1, 16, 64, 100, 128, 256, 1000, 1024, 4096, 8192)

# mask -> (causal, window build, runtime window or None, principal diagonal)
#
# `causal_br` is the default causal build (no window): bottom-right aligned,
# row i keeps j <= i + (sk - sq). `causal_tl` needs a window build fed the
# top-left sentinels. The two numeric windows are measured from the diagonal
# the principal-diagonal bit names (VarlenBits bit 18):
#
#   win_br_128_0   flash-attention's local causal window: j in [i+d-128, i+d]
#   win_tl_64_64   a symmetric band around the top-left diagonal: j in [i-64, i+64]
#
# with d = sk - sq of the sequence. The sentinel values are resolved by the
# worker, which imports `fmha_utils`.
MASKS = ("none", "causal_br", "causal_tl", "win_br_128_0", "win_tl_64_64")
MASK_SPEC = {
    "none": dict(causal=False, window_build=False, window=None, principal_br=False),
    "causal_br": dict(causal=True, window_build=False, window=None, principal_br=False),
    "causal_tl": dict(causal=True, window_build=True, window=("TL", "TL"), principal_br=False),
    "win_br_128_0": dict(causal=True, window_build=True, window=(128, 0), principal_br=True),
    "win_tl_64_64": dict(causal=True, window_build=True, window=(64, 64), principal_br=False),
}

FEATURES = ("none", "bias", "dropout")

# Dropout runtime parameters, fixed for the whole sweep.
DROPOUT_P = 0.2
PHILOX_SEED = 1234
PHILOX_OFFSET2 = 77

LAYOUTS_CHECK = ("dense", "varlen")
LAYOUTS_TIME = ("dense",)

PASSES = ("compile", "check", "time")

STATUSES = ("OK", "WRONG", "FAULT", "BUILD_ERROR", "REFUSED", "NOT_BUILDABLE", "SKIPPED")


def combo_valid(mask, feature):
    """Bias excludes causal and windows by construction (`make_traits`)."""
    return not (feature == "bias" and mask != "none")


def groups(kernels=KERNELS, head_dims=HEAD_DIMS, dtypes=DTYPES, masks=MASKS, features=FEATURES):
    out = []
    for kernel in kernels:
        for hd in head_dims:
            for dtype in dtypes:
                for mask in masks:
                    for feature in features:
                        if combo_valid(mask, feature):
                            out.append(dict(kernel=kernel,
                                            hd=int(hd),
                                            dtype=dtype,
                                            mask=mask,
                                            feature=feature))
    return out


def group_tag(g):
    return f"{g['kernel']}-hd{g['hd']}-{g['dtype']}-{g['mask']}-{g['feature']}"


def record_key(r):
    """Identity of a record. `layout/sq/sk` are None for compile records."""
    return (r["kernel"],
            r["hd"],
            r["dtype"],
            r["mask"],
            r["feature"],
            r.get("layout"),
            r.get("sq"),
            r.get("sk"),
            r["candidate"],
            r["pass"])


def load_records(out_dir):
    """Every record in `out_dir/records*.jsonl`, skipping torn lines."""
    recs = []
    for path in sorted(glob.glob(os.path.join(out_dir, "records*.jsonl"))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return recs


def done_keys(out_dir):
    return {record_key(r) for r in load_records(out_dir)}


def append_record(path, rec):
    """One line, one `write`, O_APPEND: safe against concurrent appenders."""
    line = json.dumps(rec, sort_keys=True, default=str) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line.encode())
    finally:
        os.close(fd)


def live_pairs(sq, sk, mask):
    """Live (row, col) pairs of one sequence under `mask`; the FLOP count's unit."""
    n = 0
    d = sk - sq
    for i in range(sq):
        lo, hi = 0, sk - 1
        if mask == "causal_br":
            hi = min(hi, i + d)
        elif mask == "causal_tl":
            hi = min(hi, i)
        elif mask == "win_br_128_0":
            lo, hi = max(lo, i + d - 128), min(hi, i + d)
        elif mask == "win_tl_64_64":
            lo, hi = max(lo, i - 64), min(hi, i + 64)
        n += max(0, hi - lo + 1)
    return n
