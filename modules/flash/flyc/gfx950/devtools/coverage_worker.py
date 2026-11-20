#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One group of the gfx950 flyc coverage sweep, in one process. Run by `coverage_sweep.py`.

A group is `(kernel, head_dim, dtype, mask, feature)`; see `coverage_space`.
This process discovers the group's candidate builds, then serves one pass:

- `compile`: build every candidate with `FLYDSL_COMPILE_ONLY=1` (set by the
  driver) so the JIT disk cache is warm before anything is timed.
- `check`: every `(layout, sq, sk)` cell, dense and packed varlen, against an
  fp32 torch reference; one record per (cell, candidate).
- `time`: every dense cell at fixed FLOPs, candidates interleaved with
  alternating order; one record per (cell, candidate).

**Fault protocol.** Before each GPU launch whose fault would kill the process
it prints `@@BEGIN <json key>` and after the synchronize `@@END`, both flushed.
Before building a candidate it prints `@@COMPILE <build id>` and after
`@@COMPILED`. The driver attributes a crash to the last unmatched marker,
writes the FAULT (or BUILD_ERROR) record itself, and restarts this worker,
which skips every key already recorded.

Environment: `AOTRITON_FLYDSL_KERNEL_ROOT`, `FLYDSL_EXTRA_SOURCE_DIRS` (the JIT
cache key), `ROCM_PATH`; see `coverage_sweep.py`.
"""

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_FLYC = _HERE.parents[1]
_KERNEL_DIR = _FLYC.parent / "kernel"
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_FLYC))
_KERNEL_ROOT = os.environ.get("AOTRITON_FLYDSL_KERNEL_ROOT")
if not _KERNEL_ROOT:
    sys.exit("AOTRITON_FLYDSL_KERNEL_ROOT is unset")
sys.path.append(_KERNEL_ROOT)

import torch  # noqa: E402

import coverage_space as space  # noqa: E402
import fmha_utils as fmha  # noqa: E402
import tuning_dkdv  # noqa: E402
import tuning_dq  # noqa: E402
import tuning_fused_bwd  # noqa: E402
import tuning_fwd  # noqa: E402
from fmha_meta import CAUSAL_BOTTOM_RIGHT, CAUSAL_NONE, CAUSAL_WINDOW, FmhaInputMetadata  # noqa: E402

ARCH = "gfx950"

TORCH_DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16}

# Correctness pass shape. One build serves every H (AOT builds pin
# num_heads=1), so the check runs small and the timing runs at H=16.
CHECK_BATCH = 2
CHECK_HEADS = 2
TIME_HEADS = 16
# Forward builds are compiled for num_heads=1, exactly as the AOT descriptions
# do; the forward serves any head count. The split dK/dV build is head-count
# specific (its launcher refuses any other `(num_heads_q, num_heads_k)`), so
# every backward build is compiled for the head count of the pass that runs it:
# CHECK_HEADS for `check`, TIME_HEADS for `time`, both for `compile`.
BUILD_HEADS = 1

# Tolerances, from the existing JIT tests:
#   forward O    1.5e-2 absolute (test_fwd_principal_diagonal_gfx950), scaled
#                by max(1, max|ref|) since dropout and Sk=1 put |O| above 1
#   forward LSE  1e-2 absolute on live rows (test_fwd_masked_lse_gfx950 uses 8e-3)
#   backward     max|err| <= 2e-2 * max|ref| per gradient (test_bwd_fused_gfx950),
#                floored at 1e-3 of the cell's largest gradient (`bwd_tolerance`)
FWD_O_TOL = 1.5e-2
FWD_LSE_TOL = 1e-2
BWD_TOL = 2e-2
BWD_FLOOR = 1e-3  # see `bwd_tolerance`

# Fixed-FLOPs timing: B*H*live_pairs == 4*8*4096*4096, B capped below.
REF_WORK = 4 * 8 * 4096 * 4096
MAX_BATCH = 65535
TIME_MEM_CAP = 40 * 2**30
TIME_ROUNDS = 7
TIME_SAMPLE_SECONDS = 2e-3
TIME_MAX_PER_SAMPLE = 100
# Backward timing batches keep O below 2**31 elements. **This is a limitation of
# this JIT harness, not a product bug; do not re-report it as one.**
#
# The split backward JIT-compiles AOTriton's Triton `bwd_preprocess` and passes
# its strides as Python ints, which Triton types as i32 whenever they fit. Its
# `batch * stride_oz` offset product is then formed in 32 bits and wraps past
# 2**31: at Sq=1000, H=16, hd128 `stride_oz` is 2,048,000, which wraps at batch
# 1049. Measured at hd128, Sq=1000, Sk=16: 0 bad rows at B=1040, 819095 at
# B=1100 (a silently wrong delta), an illegal-address fault at B=2097.
#
# The AOT build declares strides as hidden `u64:8` arguments
# (`python/template_instantiation/builder/kernel.py:59`), so the shipped kernel
# forms 64-bit offsets. Verified: on a B=1100 run (O = 2.25e9 elements) the AOT
# `triton_split` and `flyc` backwards give dQ/dK/dV for the last 4 batch entries
# identical (max diff 0.0) to running those 4 entries alone.
INT32_ELEMS = 2**31

OUT = None  # records file


# ---------------------------------------------------------------------------
# Protocol helpers
# ---------------------------------------------------------------------------


def marker(tag, payload=""):
    sys.stdout.write(f"@@{tag} {payload}\n")
    sys.stdout.flush()


def emit(rec):
    rec.setdefault("ts", time.time())
    space.append_record(OUT, rec)


def base_record(group, layout, sq, sk, candidate, pass_name):
    return dict(group,
                layout=layout,
                sq=sq,
                sk=sk,
                candidate=candidate,
                **{"pass": pass_name})


def _short(e):
    s = f"{type(e).__name__}: {e}"
    return s if len(s) < 2000 else s[:2000] + "..."


# ---------------------------------------------------------------------------
# Compiled-kernel statistics, read off the JIT artifact (no extra launch)
# ---------------------------------------------------------------------------

_META_RE = re.compile(r"gpu\.kernel_metadata<\"([^\"]+)\".*?metadata = \{([^}]*)\}", re.S)
_STAT_KEYS = ("vgpr_count",
              "agpr_count",
              "sgpr_count",
              "vgpr_spill_count",
              "sgpr_spill_count",
              "group_segment_fixed_size",
              "private_segment_fixed_size")


def compiled_stats(fn, module):
    """Register/LDS/scratch numbers of `fn`'s compiled kernel, or None before its first launch."""
    comp = getattr(module, "_COMPILED", None)
    if comp is None or fn.__closure__ is None:
        return None
    for cell in fn.__closure__:
        try:
            obj = cell.cell_contents
        except ValueError:
            continue
        try:
            cf = comp.get(obj)
        except TypeError:
            continue
        if cf is None:
            continue
        text = getattr(getattr(cf, "_keepalive", None), "_ir_text", None)
        if not text:
            return None
        m = _META_RE.search(text)
        if not m:
            return None
        stats = {}
        for item in m.group(2).split(","):
            if "=" not in item:
                continue
            k, v = item.split("=", 1)
            k = k.strip()
            if k in _STAT_KEYS:
                try:
                    stats[k] = int(v.split(":")[0].strip())
                except ValueError:
                    pass
        return stats
    return None


# ---------------------------------------------------------------------------
# Philox dropout mask, in torch (independent of the kernels)
# ---------------------------------------------------------------------------

_M32 = 0xFFFFFFFF
_KA, _KB = 0x9E3779B9, 0xBB67AE85
_MA, _MB = 0xD2511F53, 0xCD9E8D57


def _mulhi32(a, b):
    """High 32 bits of a 32x32 product, exact in int64 (split b in 16-bit halves)."""
    lo = b & 0xFFFF
    hi = b >> 16
    return ((a * hi) + ((a * lo) >> 16)) >> 16


def philox_keep(seed, offset_base, plane, n_rows, n_cols, rows, cols, p):
    """Keep mask `[len(rows), len(cols)]` of one plane, as `philox.Philox` (width 32, 10 rounds) draws it.

    The offset scheme is `Philox.grid_plane`/`grid_offset`: a row-major grid
    per plane, four randoms per offset along the row, element `(r, c)` at slot
    `c % 4` of offset `base + plane*n_rows*ceil(n_cols/4) + r*ceil(n_cols/4) + c//4`.
    Keep is the signed compare `int32(u) > int((p - 0.5) * 0xFFFFFFFF)`.
    """
    dev = rows.device
    row_stride = (n_cols + 3) // 4
    base = offset_base + plane * n_rows * row_stride
    noff = (int(cols.max().item()) // 4) + 1
    off = base + rows.to(torch.int64)[:, None] * row_stride + torch.arange(noff, device=dev, dtype=torch.int64)[None, :]
    c0 = off & _M32
    c1 = (off >> 32) & _M32
    c2 = torch.zeros_like(c0)
    c3 = torch.zeros_like(c0)
    k0 = seed & _M32
    k1 = (seed >> 32) & _M32
    for _ in range(10):
        _c0, _c2 = c0, c2
        c0 = _mulhi32(_MB, _c2) ^ c1 ^ k0
        c2 = _mulhi32(_MA, _c0) ^ c3 ^ k1
        c1 = (_MB * _c2) & _M32
        c3 = (_MA * _c0) & _M32
        k0 = (k0 + _KA) & _M32
        k1 = (k1 + _KB) & _M32
    vals = torch.stack((c0, c1, c2, c3), dim=-1).reshape(len(rows), noff * 4)
    vals = vals[:, cols.to(torch.int64)]
    vals = torch.where(vals >= 2**31, vals - 2**32, vals)
    thr = max(-(2**31), min(2**31 - 1, int((p - 0.5) * 0xFFFFFFFF)))
    return vals > thr


# ---------------------------------------------------------------------------
# Masks and the fp32 reference
# ---------------------------------------------------------------------------


def window_args(mask):
    """`(window or None, varlen principal-diagonal word)` for the kernel call."""
    spec = space.MASK_SPEC[mask]
    w = spec["window"]
    if w is not None:
        w = tuple(fmha.WINDOW_TOPLEFT if x == "TL" else fmha.WINDOW_BOTRIGHT if x == "BR" else x for x in w)
    pd = fmha.VARLEN_PRINCIPAL_DIAGONAL_BOTTOM_RIGHT if spec["principal_br"] else fmha.VARLEN_PRINCIPAL_DIAGONAL_TOP_LEFT
    return w, pd


def attn_keep(mask, rows, sq, sk, device):
    """`[len(rows), sk]` attention keep mask of one sequence (A-2 predicate)."""
    i = rows.view(-1, 1)
    j = torch.arange(sk, device=device).view(1, -1)
    d = sk - sq
    if mask == "none":
        return torch.ones(len(rows), sk, dtype=torch.bool, device=device)
    if mask == "causal_br":
        return j <= i + d
    if mask == "causal_tl":
        return j <= i
    if mask == "win_br_128_0":
        return (j >= i + d - 128) & (j <= i + d)
    if mask == "win_tl_64_64":
        return (j >= i - 64) & (j <= i + 64)
    raise ValueError(mask)


def reference(q, k, v, do, mask, bias, drop, scale, need_bwd):
    """fp32 attention forward (+ backward) of one sequence, all heads, chunked over rows.

    q/do `[H, sq, D]`, k/v `[H, sk, D]` (any float dtype), bias `[H, sq, sk]`
    or None, drop a callable `rows -> [H, len(rows), sk]` keep mask or None.
    Returns (o, lse, dq, dk, dv) in fp32; lse is +inf on rows with no live key,
    the forward's contract. Gradients are None unless `need_bwd`.

    The backward is written out (not autograd) so it can be chunked: with
    `Pd = P * keep / (1 - p)`, `O = Pd V`, `delta = rowsum(dO * O)`,
    `dS = P * (dO V^T * keep / (1 - p) - delta)`, `dQ = dS K * scale`,
    `dK = dS^T Q * scale`, `dV = Pd^T dO`.
    """
    H, sq, D = q.shape
    sk = k.shape[1]
    dev = q.device
    qf, kf, vf = q.float(), k.float(), v.float()
    dof = do.float() if need_bwd else None
    o = torch.empty(H, sq, v.shape[2], device=dev)
    lse = torch.empty(H, sq, device=dev)
    dq = torch.empty(H, sq, D, device=dev) if need_bwd else None
    dk = torch.zeros(H, sk, D, device=dev) if need_bwd else None
    dv = torch.zeros(H, sk, v.shape[2], device=dev) if need_bwd else None
    keep_scale = 1.0 / (1.0 - space.DROPOUT_P)
    chunk = max(1, int(2**26 // max(1, H * sk)))
    for r0 in range(0, sq, chunk):
        r1 = min(sq, r0 + chunk)
        rows = torch.arange(r0, r1, device=dev)
        s = torch.einsum("hqd,hkd->hqk", qf[:, r0:r1], kf) * scale
        if bias is not None:
            s = s + bias[:, r0:r1].float()
        am = attn_keep(mask, rows, sq, sk, dev)
        s = s.masked_fill(~am.unsqueeze(0), float("-inf"))
        lse_c = torch.logsumexp(s, dim=-1)
        dead = torch.isinf(lse_c)
        p = torch.exp(s - torch.where(dead, torch.zeros_like(lse_c), lse_c).unsqueeze(-1))
        p = torch.where(dead.unsqueeze(-1), torch.zeros_like(p), p)
        del s
        keep = drop(rows) if drop is not None else None
        pd = p * keep * keep_scale if keep is not None else p
        o_c = torch.einsum("hqk,hkd->hqd", pd, vf)
        o[:, r0:r1] = o_c
        lse[:, r0:r1] = torch.where(dead, torch.full_like(lse_c, float("inf")), lse_c)
        if need_bwd:
            do_c = dof[:, r0:r1]
            dpd = torch.einsum("hqd,hkd->hqk", do_c, vf)
            dp = dpd * keep * keep_scale if keep is not None else dpd
            # delta from O rounded to the input dtype: the backward is handed
            # that O (and the fused kernel forms delta from it), so this is the
            # gradient the kernels are asked for. With fp32 O the exact dQ/dK
            # of a one-key row is 0 and bf16 rounding of O reads as error.
            delta = (do_c * o_c.to(q.dtype).float()).sum(-1, keepdim=True)
            ds = p * (dp - delta)
            dq[:, r0:r1] = torch.einsum("hqk,hkd->hqd", ds, kf) * scale
            dk += torch.einsum("hqk,hqd->hkd", ds, qf[:, r0:r1]) * scale
            dv += torch.einsum("hqk,hqd->hkd", pd, do_c)
            del dpd, dp, ds
        del p, pd
    return o, lse, dq, dk, dv


def padded(shape, d, dtype, device="cuda", fill=None):
    """A `[..., :d]` view of a ceil8(d)-wide allocation: the 8xD input contract."""
    dp = (d + 7) // 8 * 8
    if fill is None:
        t = torch.randn(*shape, dp, device=device, dtype=dtype)
    else:
        t = torch.full((*shape, dp), fill, device=device, dtype=dtype)
    return t[..., :d]


def like(t, fill=0.0):
    """`padded` with `t`'s shape and dtype: keeps the 8xD pitch `*_like` would drop."""
    return padded(t.shape[:-1], t.shape[-1], t.dtype, fill=fill)


# ---------------------------------------------------------------------------
# Problems: one cell's inputs, in either layout
# ---------------------------------------------------------------------------


def varlen_seqs(sq, sk):
    """Four ragged sequences whose maxima are exactly (sq, sk)."""
    return [(sq, sk),
            ((sq + 1) // 2, sk),
            (sq, max(1, (sk + 2) // 3)),
            (max(1, sq // 3), max(1, sk // 2))]


class Problem:
    """One cell's tensors and its launch-time arguments.

    `segments` is `(batch, q_row0, sq, k_row0, sk, seq_idx)` per sequence, so
    the reference loop is the same for both layouts. `seq_idx` is the grid z
    that indexes the philox plane.
    """

    def __init__(self, group, layout, sq, sk, heads, batch, need_bwd, seed):
        torch.manual_seed(seed)
        self.group = group
        self.layout = layout
        self.sq, self.sk = sq, sk
        self.H = heads
        hd = group["hd"]
        dt = TORCH_DTYPE[group["dtype"]]
        self.dt = dt
        self.scale = 1.0 / math.sqrt(hd)
        self.window, pdiag = window_args(group["mask"])
        if layout == "dense":
            self.B = batch
            self.segments = [(b, 0, sq, 0, sk, b) for b in range(batch)]
            Tq, Tk = sq, sk
            self.varlen = fmha.varlen_dense(principal_diagonal=pdiag)
            self.num_seqlens = 0
            self.cu_q = None
        else:
            seqs = varlen_seqs(sq, sk)
            cq, ck = [0], [0]
            for a, b in seqs:
                cq.append(cq[-1] + a)
                ck.append(ck[-1] + b)
            self.B = 1
            self.segments = [(0, cq[z], a, ck[z], b, z) for z, (a, b) in enumerate(seqs)]
            Tq, Tk = cq[-1], ck[-1]
            self.cu_q = torch.tensor(cq, device="cuda", dtype=torch.int32)
            self.cu_k = torch.tensor(ck, device="cuda", dtype=torch.int32)
            self.varlen = dict(bits=fmha.varlen_bits(fmha.VARLEN_COMPACT_SIDE,
                                                     fmha.VARLEN_COMPACT_SIDE,
                                                     principal_diagonal=pdiag),
                               seqinfo_q0=self.cu_q,
                               seqinfo_q1=None,
                               seqinfo_k0=self.cu_k,
                               seqinfo_k1=None,
                               max_seqlen_q=sq,
                               max_seqlen_k=sk,
                               lse_tokens=None)
            self.num_seqlens = len(seqs)
        self.Tq, self.Tk = Tq, Tk
        B, H = self.B, heads
        self.q = padded((B, H, Tq), hd, dt)
        self.k = padded((B, H, Tk), hd, dt)
        self.v = padded((B, H, Tk), hd, dt)
        self.do = padded((B, H, Tq), hd, dt) if need_bwd else None
        # One bias for every kernel, per the varlen bias contract (the Triton
        # kernels'; `ParityKernelContext._score_slab_byte_base`): sequence z
        # owns slot z of a `(num_seqs, H, max_seqlen_q, max_seqlen_k)` bias and
        # reads its top-left `seqlen_q[z] x seqlen_k[z]` corner, rows and
        # columns local to the sequence. Random per slot, so a kernel that
        # reads another sequence's slot is WRONG rather than lucky.
        if group["feature"] == "bias":
            self.bias = torch.randn(len(self.segments), H, sq, sk, device="cuda", dtype=dt)
        else:
            self.bias = None
        self.dropout = group["feature"] == "dropout"

    def kw(self):
        """Keyword arguments common to every kernel of this cell."""
        kw = dict(scale=self.scale, varlen=self.varlen, num_seqlens=self.num_seqlens)
        if self.window is not None:
            kw["window"] = self.window
        if self.bias is not None:
            kw["bias"] = self.bias
        if self.dropout:
            kw.update(dropout_p=space.DROPOUT_P,
                      philox_seed=space.PHILOX_SEED,
                      philox_offset1=None,
                      philox_offset2=space.PHILOX_OFFSET2)
        return kw

    def drop_fn(self, seq_idx, seg_sk):
        """Keep-mask callable of one sequence; the planes use the max lengths."""
        if not self.dropout:
            return None
        H = self.H

        def f(rows):
            cols = torch.arange(seg_sk, device="cuda")
            return torch.stack([philox_keep(space.PHILOX_SEED,
                                            space.PHILOX_OFFSET2,
                                            seq_idx * H + h,
                                            self.sq,
                                            self.sk,
                                            rows,
                                            cols,
                                            space.DROPOUT_P) for h in range(H)])
        return f

    def reference(self, need_bwd):
        """Per-layout fp32 references, assembled in the kernels' tensor layout."""
        B, H, hd = self.B, self.H, self.group["hd"]
        mask = self.group["mask"]
        self.ref_o = torch.zeros(B, H, self.Tq, hd, device="cuda")
        self.ref_lse = torch.zeros(B, H, self.Tq, device="cuda")
        if need_bwd:
            self.ref_dq = torch.zeros(B, H, self.Tq, hd, device="cuda")
            self.ref_dk = torch.zeros(B, H, self.Tk, hd, device="cuda")
            self.ref_dv = torch.zeros(B, H, self.Tk, hd, device="cuda")
        self.ref_live = torch.zeros(B, H, self.Tq, dtype=torch.bool, device="cuda")
        for b, q0, sq, k0, sk, z in self.segments:
            qs, ks = slice(q0, q0 + sq), slice(k0, k0 + sk)
            # Local rows and columns in both layouts, slot `z` of the
            # sequence (the batch entry, for the dense layout).
            bias = self.bias[z, :, 0:sq, 0:sk] if self.bias is not None else None
            o, lse, dq, dk, dv = reference(self.q[b, :, qs],
                                           self.k[b, :, ks],
                                           self.v[b, :, ks],
                                           self.do[b, :, qs] if need_bwd else None,
                                           mask,
                                           bias,
                                           self.drop_fn(z, sk),
                                           self.scale,
                                           need_bwd)
            self.ref_o[b, :, qs] = o
            self.ref_lse[b, :, qs] = lse
            self.ref_live[b, :, qs] = torch.isfinite(lse)
            if need_bwd:
                self.ref_dq[b, :, qs] = dq
                self.ref_dk[b, :, ks] = dk
                self.ref_dv[b, :, ks] = dv

    def lse_shape(self):
        return (self.B * self.H, self.Tq)


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


class Candidate:
    """A named build choice. `builds` maps a build id to a lazily built callable."""

    def __init__(self, name, status=None, error=None):
        self.name = name
        self.status = status  # None (buildable) or REFUSED/NOT_BUILDABLE/SKIPPED
        self.error = error


class FwdCandidate(Candidate):

    def __init__(self, name, meta, knobs, status=None, error=None):
        super().__init__(name, status, error)
        self.meta = meta
        self.knobs = knobs
        self.fn = None
        self.build_error = None
        self.stats = None

    def build_id(self, sk=None):
        return self.name

    def knob_dict(self, sk=None):
        return asdict(self.knobs) if self.knobs is not None else None


def group_meta(group, heads):
    """The group's JIT metadata: the tile is left to `plan`."""
    spec = space.MASK_SPEC[group["mask"]]
    if spec["window_build"]:
        causal_type = CAUSAL_WINDOW
    else:
        causal_type = CAUSAL_BOTTOM_RIGHT if spec["causal"] else CAUSAL_NONE
    return FmhaInputMetadata(num_heads=heads,
                             block_dmodel=None,
                             head_dim=group["hd"],
                             causal_type=causal_type,
                             bias=group["feature"] == "bias",
                             dropout=group["feature"] == "dropout",
                             dtype_str=space.DTYPE_STR[group["dtype"]])


def fwd_candidates(group):
    from gfx950.tuning_fwd import Gfx950Knobs

    meta = group_meta(group, BUILD_HEADS)
    variants = [("default", {}), ("exact_kv_tiles", dict(exact_kv_tiles=True))]
    for g in Gfx950Knobs._SUPPORTED_GEOMETRIES:
        w, bm, bn, gr = g
        variants.append((f"geom_w{w}_m{bm}_n{bn}_g{gr}",
                         dict(num_waves=w, block_m=bm, block_n=bn, head_dim_granule=gr)))
    out, seen = [], {}
    for name, over in variants:
        try:
            # `plan` keeps the LSE store, as AOT binaries do.
            planned, kn = tuning_fwd.plan(ARCH, meta, overrides=over)
        except (ValueError, NotImplementedError, TypeError) as e:
            out.append(FwdCandidate(name, meta, None, status="REFUSED", error=_short(e)))
            continue
        key = json.dumps(asdict(kn), sort_keys=True)
        if key in seen:
            # Same resolved knobs as an earlier candidate: the same build.
            out.append(FwdCandidate(name, planned, kn, status="SKIPPED", error=f"same build as {seen[key]}"))
            continue
        seen[key] = name
        out.append(FwdCandidate(name, planned, kn))
    return out


class SplitCandidate(Candidate):
    """Triton bwd_preprocess (delta) + flyc dK/dV + flyc dQ, each by its own policy."""

    def __init__(self, group, heads):
        super().__init__("split")
        self.heads = heads
        meta = group_meta(group, heads)
        self.hd = group["hd"]
        self.dkdv_meta = self.dq_meta = meta
        self.knobs = {}
        try:
            self.dkdv_meta, self.knobs["dkdv"] = tuning_dkdv.plan(ARCH, meta)
            # No dB: the fused candidate does not produce it either.
            self.dq_meta, self.knobs["dq"] = tuning_dq.plan(ARCH, meta, overrides={"store_db": False})
        except (ValueError, NotImplementedError) as e:
            self.status, self.error = "REFUSED", _short(e)
        self.fns = None
        self.build_error = None
        self.stats = None
        self.pre = None

    def build_id(self, sk=None):
        return "split"

    def knob_dict(self, sk=None):
        return {k: asdict(v) for k, v in self.knobs.items()}


class FusedCandidate(Candidate):
    """The fused single-pass backward; one build per KV rung (64/128/256)."""

    def __init__(self, group, heads):
        super().__init__("fused")
        self.heads = heads
        from gfx950.tuning_fused_bwd import KV_LADDER

        self.rungs = {}
        for kv in KV_LADDER:
            meta = group_meta(group, heads)
            try:
                meta, kn = tuning_fused_bwd.plan(ARCH, meta, max_seqlen_k=kv)
                self.rungs[kv] = dict(meta=meta, knobs=kn, fn=None, build_error=None, stats=None, status=None)
            except (ValueError, NotImplementedError) as e:
                self.rungs[kv] = dict(meta=meta, knobs=None, status="NOT_BUILDABLE", error=_short(e))
        self.kv_ladder = KV_LADDER

    def rung(self, sk):
        for kv in self.kv_ladder:
            if sk <= kv:
                return kv
        return None

    def build_id(self, sk=None):
        kv = self.rung(sk) if sk is not None else None
        return f"fused@kv{kv}" if kv is not None else "fused"

    def knob_dict(self, sk=None):
        kv = self.rung(sk)
        if kv is None or self.rungs[kv]["knobs"] is None:
            return None
        return asdict(self.rungs[kv]["knobs"])


def bwd_candidates(group, heads):
    return [SplitCandidate(group, heads), FusedCandidate(group, heads)]


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def get_fwd_fn(c):
    if c.fn is None and c.build_error is None:
        marker("COMPILE", c.build_id())
        try:
            c.fn = tuning_fwd.builder(ARCH, c.meta, c.knobs)
        except Exception as e:  # noqa: BLE001
            c.build_error = _short(e)
        marker("COMPILED", c.build_id())
    return c.fn


def get_split_fns(c):
    if c.fns is None and c.build_error is None:
        marker("COMPILE", "split")
        try:
            c.fns = (tuning_dkdv.builder(ARCH, c.dkdv_meta, c.knobs["dkdv"]),
                     tuning_dq.builder(ARCH, c.dq_meta, c.knobs["dq"]))
            sys.path.append(str(_KERNEL_DIR))
            from bwd_preprocess import bwd_preprocess
            c.pre = bwd_preprocess
        except Exception as e:  # noqa: BLE001
            c.build_error = _short(e)
        marker("COMPILED", "split")
    return c.fns


def get_fused_fn(c, sk):
    kv = c.rung(sk)
    r = c.rungs[kv]
    if r.get("fn") is None and r.get("build_error") is None:
        marker("COMPILE", c.build_id(sk))
        try:
            r["fn"] = tuning_fused_bwd.builder(ARCH, r["meta"], r["knobs"])
        except Exception as e:  # noqa: BLE001
            r["build_error"] = _short(e)
        marker("COMPILED", c.build_id(sk))
    return r.get("fn")


def fused_unavailable(c, sk):
    """(status, error) when the fused candidate cannot serve `sk`, else None."""
    kv = c.rung(sk)
    if kv is None:
        return "NOT_BUILDABLE", f"seqlen_k {sk} exceeds the fused kernel's widest KV block ({max(c.kv_ladder)})"
    r = c.rungs[kv]
    if r.get("status"):
        return r["status"], r.get("error")
    return None


# ---------------------------------------------------------------------------
# Launch wrappers
# ---------------------------------------------------------------------------


def launch_fwd(fn, pb, o, lse):
    fn(pb.q, pb.k, pb.v, o, pb.B, pb.sq, seqlen_k=pb.sk, lse=lse, **pb.kw())


def launch_pre(c, pb, o, do, delta):
    """AOTriton's Triton `bwd_preprocess`: delta = rowsum(O * dO), same varlen word."""
    import triton
    rung = c.dkdv_meta.block_dmodel_v
    bits = int(pb.varlen["bits"])
    dummy = pb.cu_q if pb.cu_q is not None else torch.zeros(1, device="cuda", dtype=torch.int32)
    grid = (triton.cdiv(pb.sq, 128), pb.H, pb.num_seqlens or pb.B)
    c.pre[grid](o,
                do,
                delta,
                *o.stride(),
                *do.stride(),
                dummy,
                bits,
                pb.sq,
                dummy,
                pb.group["hd"],
                BLOCK_M=128,
                D_HEAD=rung,
                PADDED_HEAD=pb.group["hd"] != rung)


def launch_split(c, pb, o, do, lse, delta, dq, dk, dv):
    fdk, fdq = c.fns
    launch_pre(c, pb, o, do, delta)
    kw = pb.kw()
    fdk(pb.q, pb.k, pb.v, do, dk, dv, lse, delta, pb.B, pb.sq, seqlen_k=pb.sk, **kw)
    fdq(pb.q, pb.k, pb.v, do, dq, lse, delta, pb.B, pb.sq, seqlen_k=pb.sk, **kw)


def launch_fused(fn, pb, o, do, lse, dq, dk, dv):
    fn(pb.q, pb.k, pb.v, o, do, dq, dk, dv, lse, pb.B, pb.sq, seqlen_k=pb.sk, **pb.kw())


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------


def err_stats(got, ref, where=None):
    """(max abs err, max |ref|, NaN count) over `where` (bool, broadcast) or everything."""
    g = got.float()
    nan = int(torch.isnan(g).sum().item()) if where is None else int((torch.isnan(g) & where).sum().item())
    diff = (g - ref).abs()
    if where is not None:
        diff = torch.where(where, diff, torch.zeros_like(diff))
        refm = torch.where(where, ref.abs(), torch.zeros_like(ref)).max().item() if ref.numel() else 0.0
    else:
        refm = ref.abs().max().item() if ref.numel() else 0.0
    e = torch.nan_to_num(diff, nan=float("inf")).max().item() if diff.numel() else 0.0
    return e, refm, nan


# ---------------------------------------------------------------------------
# Passes
# ---------------------------------------------------------------------------


def run_compile(group, cands, done):
    """Build every candidate once with FLYDSL_COMPILE_ONLY=1; one record per build."""
    jobs = []
    if group["kernel"] == "fwd":
        for c in cands:
            jobs.append((c, None))
    else:
        for split, fused in cands:
            jobs.append((split, None))
            for kv in fused.kv_ladder:
                jobs.append((fused, kv))
    for c, kv in jobs:
        bid = c.build_id(kv)
        heads = getattr(c, "heads", 1)
        if group["kernel"] == "bwd":
            bid = f"{bid}@H{heads}"
        rec = base_record(group, None, None, None, bid, "compile")
        if space.record_key(rec) in done:
            continue
        status = c.status
        err = c.error
        if kv is not None and status is None:
            r = c.rungs[kv]
            status, err = r.get("status"), r.get("error")
        rec.update(knobs=c.knob_dict(kv) if kv is None or c.rungs[kv]["knobs"] is not None else None)
        if status is not None:
            emit(dict(rec, status=status, error=err))
            continue
        sk = kv or 64
        pb = Problem(group, "dense", 64, sk, heads, 1, need_bwd=group["kernel"] == "bwd", seed=0)
        t0 = time.time()
        try:
            if group["kernel"] == "fwd":
                fn = get_fwd_fn(c)
                if fn is None:
                    raise RuntimeError(c.build_error)
                launch_fwd(fn, pb, like(pb.q), torch.empty(pb.lse_shape(), device="cuda"))
            elif c.name == "split":
                fns = get_split_fns(c)
                if fns is None:
                    raise RuntimeError(c.build_error)
                lse = torch.zeros(pb.lse_shape(), device="cuda")
                delta = torch.zeros(pb.lse_shape(), device="cuda")
                fdk, fdq = fns
                kw = pb.kw()
                fdk(pb.q, pb.k, pb.v, pb.do, like(pb.k), like(pb.v), lse, delta, 1, 64, seqlen_k=sk, **kw)
                fdq(pb.q, pb.k, pb.v, pb.do, like(pb.q), lse, delta, 1, 64, seqlen_k=sk, **kw)
            else:
                fn = get_fused_fn(c, sk)
                if fn is None:
                    raise RuntimeError(c.rungs[kv]["build_error"])
                launch_fused(fn,
                             pb,
                             like(pb.q),
                             pb.do,
                             torch.zeros(pb.lse_shape(), device="cuda"),
                             like(pb.q),
                             like(pb.k),
                             like(pb.v))
            emit(dict(rec, status="OK", compile_s=round(time.time() - t0, 2)))
        except Exception as e:  # noqa: BLE001
            emit(dict(rec, status="BUILD_ERROR", error=_short(e), compile_s=round(time.time() - t0, 2)))
        print(f"compiled {bid} in {time.time() - t0:.1f}s", flush=True)


def classify_exception(e):
    """Map an exception from a launch to a record status.

    A `ValueError` raised by a launcher's host-side `_args` is the build
    refusing the call (REFUSED); one raised while tracing is a BUILD_ERROR.
    """
    msg = str(e)
    if isinstance(e, ValueError):
        tb = e.__traceback__
        while tb is not None:
            if tb.tb_frame.f_code.co_name == "_args":
                return "REFUSED"
            tb = tb.tb_next
    if "hipError" in msg or "illegal" in msg.lower() or "memory access" in msg.lower():
        return "FAULT"
    return "BUILD_ERROR"


def run_check(group, cands, seqlens, done, policy_pick):
    is_fwd = group["kernel"] == "fwd"
    for layout in space.LAYOUTS_CHECK:
        for sq in seqlens:
            for sk in seqlens:
                pending = [c for c in cands
                           if space.record_key(base_record(group, layout, sq, sk, c.name, "check")) not in done]
                if not pending:
                    continue
                t_cell = time.time()
                pb = Problem(group,
                             layout,
                             sq,
                             sk,
                             CHECK_HEADS,
                             CHECK_BATCH,
                             need_bwd=not is_fwd,
                             seed=sq * 100003 + sk)
                pb.reference(need_bwd=not is_fwd)
                if not is_fwd:
                    o_in = pb.ref_o.to(pb.dt)
                    o_in = padded((pb.B, pb.H, pb.Tq), group["hd"], pb.dt, fill=0.0).copy_(o_in)
                    lse_in = pb.ref_lse.reshape(pb.lse_shape()).contiguous()
                    ref_delta = (pb.do.float() * o_in.float()).sum(-1).reshape(pb.lse_shape())
                for c in pending:
                    rec = base_record(group, layout, sq, sk, c.name, "check")
                    rec.update(build_id=c.build_id(sk), knobs=c.knob_dict(sk), policy_pick=policy_pick(c.name, sq, sk))
                    if c.status is not None:
                        emit(dict(rec, status=c.status, error=c.error))
                        continue
                    if not is_fwd and c.name == "fused":
                        un = fused_unavailable(c, sk)
                        if un:
                            emit(dict(rec, status=un[0], error=un[1]))
                            continue
                    t0 = time.time()
                    key = json.dumps(space.record_key(rec))
                    try:
                        if is_fwd:
                            fn = get_fwd_fn(c)
                            if fn is None:
                                emit(dict(rec, status="BUILD_ERROR", error=c.build_error))
                                continue
                            o = padded((pb.B, pb.H, pb.Tq), group["hd"], pb.dt, fill=float("nan"))
                            lse = torch.full(pb.lse_shape(), float("nan"), device="cuda")
                            marker("BEGIN", key)
                            launch_fwd(fn, pb, o, lse)
                            torch.cuda.synchronize()
                            marker("END", key)
                            if c.stats is None:
                                import gfx950.fwd as M
                                c.stats = compiled_stats(fn, M)
                            res = judge_fwd(pb, o, lse)
                            stats = c.stats
                        elif c.name == "split":
                            fns = get_split_fns(c)
                            if fns is None:
                                emit(dict(rec, status="BUILD_ERROR", error=c.build_error))
                                continue
                            dq, dk, dv = (padded((pb.B, pb.H, n), group["hd"], pb.dt, fill=float("nan"))
                                          for n in (pb.Tq, pb.Tk, pb.Tk))
                            delta = torch.full(pb.lse_shape(), float("nan"), device="cuda")
                            marker("BEGIN", key)
                            launch_split(c, pb, o_in, pb.do, lse_in, delta, dq, dk, dv)
                            torch.cuda.synchronize()
                            marker("END", key)
                            if c.stats is None:
                                import gfx950.dkdv as MK
                                import gfx950.dq as MQ
                                c.stats = dict(dkdv=compiled_stats(fns[0], MK), dq=compiled_stats(fns[1], MQ))
                            res = judge_bwd(pb, dq, dk, dv)
                            de, dm, dn = err_stats(delta, ref_delta)
                            res["errs"]["delta"] = de
                            res["nan"]["delta"] = dn
                            stats = c.stats
                        else:
                            fn = get_fused_fn(c, sk)
                            r = c.rungs[c.rung(sk)]
                            if fn is None:
                                emit(dict(rec, status="BUILD_ERROR", error=r["build_error"]))
                                continue
                            dq, dk, dv = (padded((pb.B, pb.H, n), group["hd"], pb.dt, fill=float("nan"))
                                          for n in (pb.Tq, pb.Tk, pb.Tk))
                            marker("BEGIN", key)
                            launch_fused(fn, pb, o_in, pb.do, lse_in, dq, dk, dv)
                            torch.cuda.synchronize()
                            marker("END", key)
                            if r["stats"] is None:
                                import gfx950.fused_bwd as MF
                                r["stats"] = compiled_stats(fn, MF)
                            res = judge_bwd(pb, dq, dk, dv)
                            stats = r["stats"]
                    except Exception as e:  # noqa: BLE001
                        status = classify_exception(e)
                        marker("END", key)
                        emit(dict(rec, status=status, error=_short(e), traceback=traceback.format_exc()[-3000:]))
                        if status == "FAULT":
                            os._exit(3)  # the context is gone; skip teardown, which can hang
                        continue
                    emit(dict(rec, **res, stats=stats, run_s=round(time.time() - t0, 3)))
                print(f"check {layout} {sq}x{sk} {time.time() - t_cell:.1f}s", flush=True)
                del pb
                torch.cuda.empty_cache()


def judge_fwd(pb, o, lse):
    mask = pb.group["mask"]
    lse = lse.view(pb.B, pb.H, pb.Tq)
    oe, om, on = err_stats(o, pb.ref_o)
    live = pb.ref_live
    le, lm, ln = err_stats(lse, pb.ref_lse, live)
    o_tol = FWD_O_TOL * max(1.0, om)
    errs = dict(O=oe, LSE=le)
    nan = dict(O=on, LSE=ln)
    fails = []
    if on:
        fails.append(f"O has {on} NaN")
    if oe > o_tol:
        fails.append(f"O err {oe:.3e} > {o_tol:.3e}")
    if ln:
        fails.append(f"LSE has {ln} NaN on live rows")
    if le > FWD_LSE_TOL:
        fails.append(f"LSE err {le:.3e} > {FWD_LSE_TOL:.1e}")
    dead = ~live
    n_dead = int(dead.sum().item())
    dead_bad = 0
    if n_dead and not space.MASK_SPEC[mask]["window_build"]:
        # The forward's contract for a row with no live key (bottom-right
        # causal, sq > sk): LSE = +inf. Window builds leave it unspecified.
        dead_bad = int(((lse != float("inf")) & dead).sum().item())
        if dead_bad:
            fails.append(f"LSE not +inf on {dead_bad}/{n_dead} dead rows")
    return dict(status="WRONG" if fails else "OK",
                error="; ".join(fails) or None,
                errs=errs,
                ref_max=dict(O=om, LSE=lm),
                nan=nan,
                dead_rows=n_dead,
                dead_lse_bad=dead_bad)


def bwd_tolerance(name, refm):
    """`BWD_TOL * max|ref|`, floored at `BWD_FLOOR` of the cell's largest gradient.

    The floor is for cells whose exact dQ/dK vanish -- one live key per row
    (Sk=1, top-left causal with Sq=1) makes dS identically zero -- where the
    reference is fp32 summation noise (~1e-7) and a purely relative bound
    compares noise against noise.
    """
    return BWD_TOL * max(refm[name], BWD_FLOOR * max(refm.values()), 1e-6)


def judge_bwd(pb, dq, dk, dv):
    errs, refm, nan, fails = {}, {}, {}, []
    for name, got, ref in (("dQ", dq, pb.ref_dq), ("dK", dk, pb.ref_dk), ("dV", dv, pb.ref_dv)):
        e, m, n = err_stats(got, ref)
        errs[name], refm[name], nan[name] = e, m, n
    for name in ("dQ", "dK", "dV"):
        e, m, n = errs[name], refm[name], nan[name]
        if n:
            fails.append(f"{name} has {n} NaN")
        tol = bwd_tolerance(name, refm)
        if e > tol:
            fails.append(f"{name} err {e:.3e} > {tol:.3e} (max|ref| {m:.3e})")
    return dict(status="WRONG" if fails else "OK",
                error="; ".join(fails) or None,
                errs=errs,
                ref_max=refm,
                nan=nan)


def choose_batch(group, sq, sk, heads):
    pairs = max(1, space.live_pairs(sq, sk, group["mask"]))
    B = max(1, round(REF_WORK / (heads * pairs)))
    dp = (group["hd"] + 7) // 8 * 8
    per_b = heads * (2 * sq + 2 * sk) * dp * 2
    if group["kernel"] == "bwd":
        per_b = heads * (5 * sq + 6 * sk) * dp * 2 + heads * sq * 8
    if group["feature"] == "bias":
        per_b += heads * sq * sk * 2
    B = min(B, MAX_BATCH, max(1, TIME_MEM_CAP // per_b))
    if group["kernel"] == "bwd":
        B = min(B, max(1, (INT32_ELEMS - 1) // (heads * max(sq, sk) * dp)))
    return B, pairs


def run_time(group, cands, seqlens, done, check_status, policy_pick):
    is_fwd = group["kernel"] == "fwd"
    H = TIME_HEADS
    for sq in seqlens:
        for sk in seqlens:
            pending = [c for c in cands
                       if space.record_key(base_record(group, "dense", sq, sk, c.name, "time")) not in done]
            if not pending:
                continue
            t_cell = time.time()
            B, pairs = choose_batch(group, sq, sk, H)
            flops = 4.0 * B * H * pairs * group["hd"] * (1.0 if is_fwd else 2.5)
            pb = Problem(group, "dense", sq, sk, H, B, need_bwd=not is_fwd, seed=1)
            calls = {}
            recs = {}
            stats = {}
            for c in pending:
                rec = base_record(group, "dense", sq, sk, c.name, "time")
                rec.update(build_id=c.build_id(sk),
                           knobs=c.knob_dict(sk),
                           policy_pick=policy_pick(c.name, sq, sk),
                           B=B,
                           H=H,
                           flops=flops,
                           live_pairs=pairs)
                if c.status is not None:
                    emit(dict(rec, status=c.status, error=c.error))
                    continue
                if not is_fwd and c.name == "fused":
                    un = fused_unavailable(c, sk)
                    if un:
                        emit(dict(rec, status=un[0], error=un[1]))
                        continue
                prev = check_status.get((sq, sk, c.name))
                if prev == "FAULT":
                    emit(dict(rec, status="SKIPPED", error="FAULT in the check pass at this cell"))
                    continue
                recs[c.name] = rec
                if is_fwd:
                    fn = get_fwd_fn(c)
                    if fn is None:
                        emit(dict(rec, status="BUILD_ERROR", error=c.build_error))
                        recs.pop(c.name)
                        continue
                    o = like(pb.q)
                    lse = torch.empty(pb.lse_shape(), device="cuda")
                    calls[c.name] = (lambda fn=fn, o=o, lse=lse: launch_fwd(fn, pb, o, lse))
                    stats[c.name] = c
                else:
                    if not hasattr(pb, "o_t"):
                        pb.o_t = like(pb.q).copy_(pb.q * 0.2)
                        pb.lse_t = torch.full(pb.lse_shape(), math.sqrt(sk) + 3.0, device="cuda")
                        pb.delta_t = torch.empty(pb.lse_shape(), device="cuda")
                        pb.grads_t = (like(pb.q), like(pb.k), like(pb.v))
                    if c.name == "split":
                        fns = get_split_fns(c)
                        if fns is None:
                            emit(dict(rec, status="BUILD_ERROR", error=c.build_error))
                            recs.pop(c.name)
                            continue
                        calls[c.name] = (lambda c=c: launch_split(c,
                                                                  pb,
                                                                  pb.o_t,
                                                                  pb.do,
                                                                  pb.lse_t,
                                                                  pb.delta_t,
                                                                  *pb.grads_t))
                    else:
                        fn = get_fused_fn(c, sk)
                        if fn is None:
                            emit(dict(rec, status="BUILD_ERROR", error=c.rungs[c.rung(sk)]["build_error"]))
                            recs.pop(c.name)
                            continue
                        calls[c.name] = (lambda fn=fn: launch_fused(fn, pb, pb.o_t, pb.do, pb.lse_t, *pb.grads_t))
            # Warm up each candidate; a host-side refusal surfaces here.
            single = {}
            for name in list(calls):
                key = json.dumps(space.record_key(recs[name]))
                try:
                    marker("BEGIN", key)
                    for _ in range(2):
                        calls[name]()
                    torch.cuda.synchronize()
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    calls[name]()
                    e.record()
                    e.synchronize()
                    marker("END", key)
                    single[name] = s.elapsed_time(e) * 1e-3
                except Exception as ex:  # noqa: BLE001
                    marker("END", key)
                    status = classify_exception(ex)
                    emit(dict(recs.pop(name), status=status, error=_short(ex)))
                    calls.pop(name)
                    if status == "FAULT":
                        os._exit(3)  # skip teardown, which can hang after a fault
            if calls:
                per = max(1, min(TIME_MAX_PER_SAMPLE, math.ceil(TIME_SAMPLE_SECONDS / max(1e-7, min(single.values())))))
                names = list(calls)
                samples = {n: [] for n in names}
                for r in range(TIME_ROUNDS):
                    for n in (names if r % 2 == 0 else names[::-1]):
                        key = json.dumps(space.record_key(recs[n]))
                        marker("BEGIN", key)
                        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        s.record()
                        for _ in range(per):
                            calls[n]()
                        e.record()
                        e.synchronize()
                        marker("END", key)
                        samples[n].append(s.elapsed_time(e) * 1e-3 / per)
                for n in names:
                    med = statistics.median(samples[n])
                    sp = (max(samples[n]) - min(samples[n])) / med if med else 0.0
                    c = next(x for x in cands if x.name == n)
                    st = None
                    if is_fwd:
                        if c.stats is None:
                            import gfx950.fwd as M
                            c.stats = compiled_stats(c.fn, M)
                        st = c.stats
                    elif n == "fused":
                        st = c.rungs[c.rung(sk)].get("stats")
                    else:
                        st = c.stats
                    emit(dict(recs[n],
                              status="OK",
                              us=med * 1e6,
                              tf=flops / med * 1e-12,
                              per_sample=per,
                              rounds=TIME_ROUNDS,
                              spread=sp,
                              stats=st,
                              check_status=check_status.get((sq, sk, n))))
            print(f"time {sq}x{sk} B={B} {time.time() - t_cell:.1f}s", flush=True)
            del pb, calls
            torch.cuda.empty_cache()


def make_policy_pick(group):
    if group["kernel"] == "fwd":
        return lambda name, sq, sk: name == "default"
    from gfx950.tuning_fused_bwd import prefer_fused

    def pick(name, sq, sk):
        fused = prefer_fused(head_dim=group["hd"],
                             seqlen_q=sq,
                             seqlen_k=sk,
                             bias=group["feature"] == "bias",
                             dropout=group["feature"] == "dropout")
        return name == ("fused" if fused else "split")
    return pick


def main():
    global OUT
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--group", required=True, help="JSON object: kernel, hd, dtype, mask, feature")
    p.add_argument("--pass_name", choices=space.PASSES, required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--out", required=True, help="records file to append to")
    p.add_argument("--seqlens", type=int, nargs="+", default=list(space.SEQLENS))
    p.add_argument("--crashed_builds", default="[]", help="JSON list of build ids that crashed this worker")
    args = p.parse_args()
    OUT = args.out
    group = json.loads(args.group)
    torch.cuda.init()
    done = space.done_keys(args.out_dir)
    if group["kernel"] == "fwd":
        cands = fwd_candidates(group)
    elif args.pass_name == "compile":
        cands = [bwd_candidates(group, h) for h in (CHECK_HEADS, TIME_HEADS)]
    else:
        cands = bwd_candidates(group, CHECK_HEADS if args.pass_name == "check" else TIME_HEADS)
    crashed = set(json.loads(args.crashed_builds))
    for c in (cands if group["kernel"] == "fwd" or args.pass_name != "compile" else [x for p in cands for x in p]):
        if isinstance(c, FusedCandidate):
            for kv, r in c.rungs.items():
                if f"fused@kv{kv}" in crashed and r.get("status") is None:
                    r["status"], r["error"] = "BUILD_ERROR", "the build crashed the worker process"
        elif c.name in crashed and c.status is None:
            c.status, c.error = "BUILD_ERROR", "the build crashed the worker process"
    pick = make_policy_pick(group)
    if args.pass_name == "compile":
        run_compile(group, cands, done)
    elif args.pass_name == "check":
        run_check(group, cands, args.seqlens, done, pick)
    else:
        check_status = {}
        for r in space.load_records(args.out_dir):
            if (r.get("pass") == "check" and r.get("layout") == "dense"
                    and all(r.get(k) == group[k] for k in ("kernel", "hd", "dtype", "mask", "feature"))):
                check_status[(r["sq"], r["sk"], r["candidate"])] = r["status"]
        run_time(group, cands, args.seqlens, done, check_status, pick)


def philox_self_test():
    """`philox_keep` against the mask the kernels draw, read back through the split dK/dV kernel.

    With Q = K = 0 and LSE = 0 every P is 1, and with dO the identity
    `dV[j, i] = keep[i, j] / (1 - p)` (`tests/test_bwd_fused._keep_mask`).
    Dense, B=2 H=2, a few shapes. Exits nonzero on any mismatch.
    """
    width = 128
    bad = 0
    for B, H, sq, sk in ((2, 2, 100, 300), (2, 2, 128, 1), (1, 3, 37, 4100), (2, 2, 1, 64)):
        meta = FmhaInputMetadata(num_heads=H,
                                 block_dmodel=None,
                                 head_dim=width,
                                 causal_type=CAUSAL_NONE,
                                 dropout=True,
                                 dtype_str="bf16")
        f = tuning_dkdv.builder(ARCH, *tuning_dkdv.plan(ARCH, meta))
        zq = torch.zeros(B, H, sq, width, device="cuda", dtype=torch.bfloat16)
        zk = torch.zeros(B, H, sk, width, device="cuda", dtype=torch.bfloat16)
        do = torch.zeros_like(zq)
        do[..., torch.arange(sq), torch.arange(sq)] = 1.0
        rows = torch.zeros(B * H, sq, device="cuda")
        dk, dv = torch.zeros_like(zk), torch.zeros_like(zk)
        f(zq,
          zk,
          zk,
          do,
          dk,
          dv,
          rows,
          rows,
          B,
          sq,
          seqlen_k=sk,
          scale=1.0,
          dropout_p=space.DROPOUT_P,
          philox_seed=space.PHILOX_SEED,
          philox_offset1=None,
          philox_offset2=space.PHILOX_OFFSET2)
        torch.cuda.synchronize()
        got = dv[..., :sq].float().transpose(-1, -2) != 0
        r = torch.arange(sq, device="cuda")
        c = torch.arange(sk, device="cuda")
        want = torch.stack([torch.stack([philox_keep(space.PHILOX_SEED,
                                                     space.PHILOX_OFFSET2,
                                                     b * H + h,
                                                     sq,
                                                     sk,
                                                     r,
                                                     c,
                                                     space.DROPOUT_P) for h in range(H)]) for b in range(B)])
        n = int((got != want).sum().item())
        print(f"philox self-test B={B} H={H} sq={sq} sk={sk}: {n} mismatches of {want.numel()}, "
              f"keep rate {want.float().mean().item():.3f}", flush=True)
        bad += n
    sys.exit(1 if bad else 0)


if __name__ == "__main__" and "--philox_self_test" in sys.argv:
    philox_self_test()
elif __name__ == "__main__":
    main()
