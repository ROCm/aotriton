#!/usr/bin/env python3
# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How much precision `flyc_attn_fwd` loses to a large `sm_scale`, JIT.

The forward used to reach `qk_scale = sm_scale * log2e` by folding it into Q
and **rounding the product back to the input dtype**
(`DualwaveQLoader.scale_all`). Q is already bf16/f16, so that second rounding
is pure loss: every element picks up a fresh relative error of up to `2**-9`,
the QK dot multiplies it by the logit magnitude, and the result lands *in the
exponent* of `exp2(S - m)`, where it becomes a relative error on the softmax
weight of `2**(|S| * 2**-9)`. That is invisible at
`sm_scale = rsqrt(head_dim)`, where `|S|` is O(1), and several percent at
`sm_scale = 1`, where `|S|` grows as `sqrt(head_dim)`. Max |err| against fp64,
bf16, the shapes below:

    head_dim   rsqrt(d)   sm_scale 0.4   sm_scale 1.0
       64       0.0024       0.0150         0.0490      <- folded (before)
                0.0019       0.0126         0.0161      <- scaled  (after)
      256       0.0028       0.0405         0.0977
                0.0031       0.0161         0.0233
      512       0.0037       0.0631         0.1483
                0.0020       0.0127         0.0154

`BwdDqSoftmaxHelper.scale_and_sub_lse` measured the same thing from the other
end and stopped folding, on the grounds that the forward "tolerates that
because O is a normalised average and the error largely cancels". It does not
cancel far enough. `test_transformers.py::test_mem_eff_attention_single_query
_tail` calls SDPA at `scale=1.0, head_dim=512, bf16` and misses on 1.4% of its
output elements by up to 0.29 -- not in the tail block the name refers to (that
is a CUDA tile-geometry story with no ROCm analogue), but everywhere, because
every row pays this.

**What this file pins is the arithmetic, not the shape.** A `q_len` sweep
across every block boundary -- 63/64/65, 127/128/129, 255/256/257, 288/289,
320/321, causal and not, head_dim 64/256/512 -- is clean to 0.01 at
`rsqrt(head_dim)`, so no tile boundary is involved. The single knob that moves
the error is `sm_scale * sqrt(head_dim)`, which is what the cases below vary.
`SEQLEN_Q` is 289 anyway, because that is what the reporting UT used and a
regression should be reproduced at the shape it was reported at.

**Why a JIT test and why here.** Same reason as
`test_fwd_masked_lse_gfx950.py`: reaching this from the AOT suite costs a full
`ninja install` plus a by-hand deletion of the flyc hsaco/aks2/zip outputs,
because no ninja edge depends on these sources. `modules/flash/tests/
test_forward.py` covers the same ground at release tolerances; this one builds
one kernel in-process and compares against fp64, which is what lets it name the
cause rather than only the symptom.

Requires a gfx950 device, `AOTRITON_FLYDSL_KERNEL_ROOT`, and a ROCm the FlyDSL
compiler can link against; skips cleanly without any of them.

    AOTRITON_FLYDSL_KERNEL_ROOT=<flydsl-checkout> \\
        python -m pytest modules/flash/flyc/test_fwd_qk_scale_gfx950.py
"""

import math
import os
import sys
from pathlib import Path

import pytest

# See `test_fwd_masked_lse_gfx950.py` for why the kernel root is appended here
# rather than by a conftest: the vendored kernels import each other by bare
# name, and pytest's rootless import mode has already put this directory on
# `sys.path`.
_KERNEL_ROOT = os.environ.get("AOTRITON_FLYDSL_KERNEL_ROOT")
if _KERNEL_ROOT and _KERNEL_ROOT not in sys.path:
    sys.path.append(_KERNEL_ROOT)

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not _KERNEL_ROOT
    or not torch.cuda.is_available()
    or not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"),
    reason="needs a gfx950 device and AOTRITON_FLYDSL_KERNEL_ROOT",
)

LOG2E = 1.4426950408889634

BATCH = 1
NUM_HEADS = 4
SEQLEN_Q = 289
SEQLEN_K = 289

# 64 and 256 take the dual-wave kernel (`flash_attn_func_gfx950.make_body`),
# 512 the wide one (`fmha_wide_gfx950.make_wide_body`). They reintroduce the
# scale at different places -- the dual-wave body through `_FwdGemmHelper.qk`,
# which serves six call sites across its software pipeline, the wide body with
# one explicit `scale_scores` after its D-stage loop -- so covering both is
# what says the scale rides every route and not just the one a single shape
# happens to hit.
HEAD_DIMS = (64, 256, 512)

DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16}

# `None` means `rsqrt(head_dim)`: the case that always worked, here to say the
# fix did not buy the large-scale cases with the ordinary one. 1.0 is what
# `test_transformers.py` passes; 0.4 is the smallest scale at which the fold
# was still measurable, and catches a partial regression.
SM_SCALES = (None, 0.4, 1.0)

# The worst error left once the Q rounding is gone, measured across every case
# below: 0.0233, at bf16 head_dim 256 sm_scale 1.0. The forward is still a
# low-precision algorithm -- K and V are read in the input dtype, P is cast
# back to it before the PV MFMA, and the f32 accumulators are summed in tile
# order -- but what remains is roughly flat in `sm_scale`, which is the point.
NOISE_FLOOR = 0.025
# 1.5x the floor. The nearest *pre-fix* data point is 0.0405 (bf16 head_dim 256
# sm_scale 0.4), so the margin above is deliberately narrower than the margin
# below; `_folded_q_model` below is what carries the discrimination at the
# larger scales, where the two are an order of magnitude apart.
MAX_ABS_ERR = 0.035


def _build(head_dim, dtype_str, causal=False, bias=False):
    """A forward for one `(head_dim, dtype)`, compiled in-process.

    `flash_attn_func_gfx950._COMPILED` memoises, and FlyDSL caches to disk on
    top of that, so the parametrization below pays for each distinct build
    once rather than once per case.
    """
    from flash_attn_func_gfx950 import (
        build_flash_attn_func_gfx950_module_primary as build_primary,
    )
    from fmha_tuning_gfx950 import FmhaInputMetadata, fmha_knobs

    meta = FmhaInputMetadata(
        num_heads=NUM_HEADS,
        head_dim=head_dim,
        causal=causal,
        bias=bias,
        dtype_str=dtype_str,
    )
    return build_primary(meta, fmha_knobs("gfx950").resolve(meta))


def _inputs(head_dim, dtype, seed):
    torch.manual_seed(seed)
    shape_q = (BATCH, NUM_HEADS, SEQLEN_Q, head_dim)
    shape_k = (BATCH, NUM_HEADS, SEQLEN_K, head_dim)
    q = torch.randn(shape_q, device="cuda", dtype=dtype)
    k = torch.randn(shape_k, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    # NaN rather than zeros, for the reason `test_fwd_masked_lse_gfx950.py`
    # gives: an unwritten element must not be able to pass as a correct one.
    out = torch.full(shape_q, float("nan"), device="cuda", dtype=dtype)
    return q, k, v, out


def _resolve(sm_scale, head_dim):
    return 1.0 / math.sqrt(head_dim) if sm_scale is None else sm_scale


def _reference(q, k, v, sm_scale, causal=False, bias=None):
    """fp64 attention. The bar every variant below is measured against."""
    s = q.double() @ k.double().transpose(-1, -2) * sm_scale
    if bias is not None:
        s = s + bias.double()
    if causal:
        sq, sk = q.shape[-2], k.shape[-2]
        qi = torch.arange(sq, device=q.device).view(-1, 1)
        ki = torch.arange(sk, device=q.device).view(1, -1)
        s = s.masked_fill(ki > qi + (sk - sq), float("-inf"))
    return s.softmax(-1) @ v.double()


def _folded_q_model(q, k, v, sm_scale, causal=False):
    """fp64 attention over the scores the *folded* kernel formed.

    Q scaled in f32 and rounded back to its own dtype, then a base-2 softmax --
    `scale_all` verbatim, with everything else exact. This is the
    discriminator: a kernel that folds tracks this model to the noise floor at
    every `sm_scale`, while its distance to `_reference` grows with
    `sm_scale * sqrt(head_dim)`. Without it the assertion below says only "too
    much error", which a tolerance bump answers; with it, it says *which*
    rounding produced the error.
    """
    qs = (q.float() * (sm_scale * LOG2E)).to(q.dtype)
    s = qs.double() @ k.double().transpose(-1, -2)  # already in the log2 domain
    if causal:
        sq, sk = q.shape[-2], k.shape[-2]
        qi = torch.arange(sq, device=q.device).view(-1, 1)
        ki = torch.arange(sk, device=q.device).view(1, -1)
        s = s.masked_fill(ki > qi + (sk - sq), float("-inf"))
    p = torch.exp2(s - s.max(-1, keepdim=True).values)
    return (p / p.sum(-1, keepdim=True)) @ v.double()


def _max_err(got, want):
    return (got.double() - want).abs().max().item()


# --- A second, unrelated defect that these shapes walk into -----------------
#
# **fp16 plus `lazy_rescale` plus a large `sm_scale` returns inf.** Not the
# rounding this file is about, and not introduced by fixing it: the counts and
# indices are bit-identical before and after, and it reproduces on the shipped
# 0.14.1 library through `torch.nn.functional.scaled_dot_product_attention`.
#
# `lazy_rescale` holds the running row max until a tile overruns it, which lets
# `P = exp2(S - m_row)` exceed 1 before `cast_p` writes it in the input dtype.
# bf16 carries f32's exponent range and does not care; fp16 tops out at 65504,
# and the excursion at these scales passes it, so P is infinite going into the
# PV MFMA and O comes back inf. The LSE is *correct* in the same run, which is
# what says the row max and the running sum are fine and only the P cast is
# not. Building the same shape with `lazy_rescale=False` returns finite output
# and is also more accurate (2.2e-3 -> 1.1e-3 at head_dim 64, sm_scale 1.0).
#
# The wide kernel (head_dim 512) rescales eagerly and is unaffected at every
# scale here. Listed as exact measured triples rather than fitted from a
# threshold: the boundary is a property of the input distribution as much as of
# the knob, and a formula here would claim to know where it is.
#
# `strict`, so that fixing the knob policy turns these into failures and
# whoever fixes it is told to come back and delete the list.
_FP16_LAZY_RESCALE_INF = frozenset(
    {
        ("f16", 64, 1.0),
        ("f16", 256, 0.4),
        ("f16", 256, 1.0),
    }
)


def _cases(dtypes=tuple(sorted(DTYPES)), head_dims=HEAD_DIMS, scales=SM_SCALES):
    for dtype_str in dtypes:
        for head_dim in head_dims:
            for sm_scale in scales:
                marks = ()
                if (dtype_str, head_dim, sm_scale) in _FP16_LAZY_RESCALE_INF:
                    marks = pytest.mark.xfail(
                        strict=True,
                        reason="fp16 + lazy_rescale overflows P to inf at a large "
                        "sm_scale; separate pre-existing defect, see the comment "
                        "on _FP16_LAZY_RESCALE_INF",
                    )
                yield pytest.param(
                    dtype_str,
                    head_dim,
                    sm_scale,
                    marks=marks,
                    id=f"{dtype_str}-d{head_dim}-s{'rsqrt' if sm_scale is None else sm_scale}",
                )


@pytest.mark.parametrize("dtype_str,head_dim,sm_scale", list(_cases()))
def test_qk_scale_survives_a_large_sm_scale(dtype_str, head_dim, sm_scale):
    scale = _resolve(sm_scale, head_dim)
    fn = _build(head_dim, dtype_str)
    q, k, v, out = _inputs(head_dim, DTYPES[dtype_str], seed=head_dim)

    fn(q, k, v, out, BATCH, SEQLEN_Q, seqlen_k=SEQLEN_K, scale=scale)
    torch.cuda.synchronize()

    assert bool(torch.isfinite(out).all()), (
        f"head_dim={head_dim} {dtype_str} sm_scale={scale:g}: "
        f"{int((~torch.isfinite(out)).sum())} non-finite output elements"
    )

    ref = _reference(q, k, v, scale)
    folded = _folded_q_model(q, k, v, scale)
    err = _max_err(out, ref)
    err_folded = _max_err(folded, ref)

    assert err <= MAX_ABS_ERR, (
        f"head_dim={head_dim} {dtype_str} sm_scale={scale:g}: max error {err:.4f} "
        f"against fp64, over {MAX_ABS_ERR}. The folded-Q model is off by "
        f"{err_folded:.4f} and the kernel is {_max_err(out, folded):.4f} from it "
        f"-- if that second number is the smaller one, the scale is being rounded "
        f"into Q again; see ParityQLoader.scale_all"
    )
    # The floor does not depend on `sm_scale`, so a kernel that is merely
    # *inside* the bound while still tracking `_folded_q_model` is caught here
    # rather than passing: at head_dim 512 sm_scale 1.0 the two models are 0.15
    # apart, so half the folded error is not a bound the fold can meet.
    if err_folded > 2 * MAX_ABS_ERR:
        assert err < err_folded / 2, (
            f"head_dim={head_dim} {dtype_str} sm_scale={scale:g}: kernel error "
            f"{err:.4f} is no better than the folded-Q model's {err_folded:.4f}"
        )


@pytest.mark.parametrize("dtype_str,head_dim,sm_scale", list(_cases(dtypes=("bf16",))))
def test_qk_scale_under_causal(dtype_str, head_dim, sm_scale):
    """The masked path, where `-inf` scores meet the scale multiply.

    A separate test rather than another axis on the one above because the
    causal body is a different set of call sites -- the dual-wave loop reaches
    `reduce_max` through `causal_mask_prologue_if_needed` instead of
    `seq_pad_mask_if_needed` -- and because the ordering constraint is only
    visible here. `scale_scores` runs *before* every mask, so the multiply sees
    nothing but finite raw dot products; applying it after would multiply the
    `-inf` the mask writes, which is correct in IEEE and undefined under the
    `fastmath<fast>` this module compiles with. Triton takes the same
    precaution from the other side, writing `-inf` into `qk` *before* the
    `+= Qk_scale * dot` (`fwd_kernel_inner.py:134`).

    bf16 only: the fp16 excursion above is a `lazy_rescale` property, not a
    masking one, and repeating it here would only duplicate the xfails.
    """
    scale = _resolve(sm_scale, head_dim)
    fn = _build(head_dim, dtype_str, causal=True)
    q, k, v, out = _inputs(head_dim, DTYPES[dtype_str], seed=head_dim + 1)

    fn(q, k, v, out, BATCH, SEQLEN_Q, seqlen_k=SEQLEN_K, scale=scale)
    torch.cuda.synchronize()

    assert bool(torch.isfinite(out).all()), (
        f"head_dim={head_dim} causal sm_scale={scale:g}: non-finite output "
        f"(NaN would also mean rows were never written)"
    )
    err = _max_err(out, _reference(q, k, v, scale, causal=True))
    assert err <= MAX_ABS_ERR, (
        f"head_dim={head_dim} causal sm_scale={scale:g}: max error {err:.4f}"
    )


@pytest.mark.parametrize("sm_scale", [0.0, 1.0], ids=["ScaleZero", "ScaleOne"])
def test_bias_survives_a_same_shape_bias_free_build(sm_scale):
    """A bias build must apply its bias, **even after a bias-free one is built**.

    The second half is the whole test and is why the bias-free build is made
    first rather than skipped. `traits.cache_tag` -- upstream's, and the thing
    every builder's `_cache_tag` used to start from -- does not name
    `BIAS_TYPE`, so these two builds hashed to the same FlyDSL disk-cache entry
    and the bias one silently received the bias-free binary. The symptom was
    output bit-identical to a `bias=False` run for a zero bias, a constant
    bias, a per-row bias *and* a key-varying one, which reads exactly like a
    kernel that never loads its bias. It is not: built alone against an empty
    cache, the same kernel is accurate to 4e-3. `traits_cache_key` is the fix.

    Not reachable from AOTriton's AOT path -- the dispatcher hands every
    `BIAS_TYPE=1` forward to the Triton backend, and each functional is
    compiled in its own invocation, so all 72 `BIAS_TYPE=1` entries in the
    shipped `flyc_attn_fwd.zip` are distinct binaries. What it broke is
    in-process work: `devtools/` sweeps, and tests like this one.

    Both `sm_scale` ends are here because they constrain opposite things. At
    1.0 the bias must survive whatever reintroduces `qk_scale`. At **0.0** --
    which `modules/flash/tests/test_forward.py` parametrizes for every bias
    case -- the scores vanish and the softmax is over the bias alone, which is
    what rules out ever expressing the bias as `bias / sm_scale` in a raw-score
    domain, as a free FMA fusion of `qk_scale` into `sub_m` would have needed.

    Bias excludes causal (`fmha_traits_gfx950.make_traits` rejects the pair),
    so this is the dense path only.
    """
    head_dim = 64
    q, k, v, out = _inputs(head_dim, torch.bfloat16, seed=7)
    bias = torch.randn(
        (BATCH, NUM_HEADS, SEQLEN_Q, SEQLEN_K), device="cuda", dtype=torch.bfloat16
    )

    # First, and deliberately: this is the build whose binary a colliding key
    # hands to the one below.
    plain = _build(head_dim, "bf16")
    plain_out = torch.full_like(out, float("nan"))
    plain(q, k, v, plain_out, BATCH, SEQLEN_Q, seqlen_k=SEQLEN_K, scale=sm_scale)

    fn = _build(head_dim, "bf16", bias=True)
    fn(q, k, v, out, BATCH, SEQLEN_Q, seqlen_k=SEQLEN_K, scale=sm_scale, bias=bias)
    torch.cuda.synchronize()

    assert _max_err(out, plain_out.double()) > 0.0, (
        f"sm_scale={sm_scale:g}: the bias build returned the bias-free build's "
        f"output exactly -- the two collided in the JIT cache; see traits_cache_key"
    )
    err = _max_err(out, _reference(q, k, v, sm_scale, bias=bias))
    assert err <= MAX_ABS_ERR, f"sm_scale={sm_scale:g}: bias max error {err:.4f}"


if __name__ == "__main__":
    sys.exit(pytest.main([str(Path(__file__).resolve()), "-v"]))
