# Attention Kernels: Common Mistakes

This is a list of mistakes that AOTriton's Triton attention kernels have made
and fixed, written for authors of new attention kernels. A new backend
(a hand-written FlyDSL kernel, an assembly kernel, a port to a new
architecture) meets the same numerics. It is likely to make the same
mistakes, and most of them are **silent**: the output looks plausible, and the
standard test suite passes it.

The motivating case: in 2024 the Triton kernels stopped multiplying Q by
`sm_scale * log2(e)` and rounding it back to fp16/bf16 before `QK^T` (7a165f1e).
Two years later a new kernel made the same choice, and `test_op_bwd` could not
tell.

Every entry gives the mistake, why it is wrong, which inputs expose it, where it
was fixed, and which test guards it. Commits are `git show`-able. PR and issue
numbers are on `ROCm/aotriton`.

## The contract, in one place

These are the conventions the Triton kernels settled on. A kernel that computes
a different but "equivalent" quantity has to be exactly as precise and has to
agree with every kernel it hands data to.

| quantity | convention | why |
|---|---|---|
| scores | `S = Q K^T` in fp32; `qk_scale = sm_scale * log2(e)` multiplies the **fp32** `S` | see [Rounding a scaled operand](#1-rounding-a-scaled-operand-to-the-input-dtype) |
| softmax | base 2: `p = exp2(qk_scale * S + log2(e) * bias - m)`; `m` is the max of the *same* rounded values | see [The max and the exponent](#3-the-max-and-the-exponent-computed-from-differently-rounded-values) |
| bias | multiplied by `log2(e)` **in fp32**, identically in fwd and bwd | see [Constants multiplied in low precision](#4-constants-multiplied-in-low-precision) |
| saved LSE | **natural base**, `logsumexp(sm_scale * S + bias)`; the backward multiplies it by `log2(e)` on load | see [LSE base](#5-the-lse-in-the-wrong-base-or-units) |
| fully masked row | `O = 0`, `LSE = +inf`, `dQ = 0` | `+inf` makes the backward's `exp(S - LSE)` vanish |
| `Delta` | `rowsum(dO * O)` in fp32, from the O the forward **stored** | see [Delta in low precision](#6-delta-in-low-precision) |
| `dS` | `P * (dP - Delta)`, the gradient with respect to the *scaled* logits | `sm_scale` then multiplies dQ and dK exactly once |
| `dB` | `dS` itself, no `log2(e)`, no `sm_scale` | the bias enters the softmax unscaled |
| dropout | `l_i` sums P **before** the mask; `1/(1-p)` applied once; identical Philox offsets and word order in every kernel | see [Dropout](#9-dropout) |

## Why the standard tests pass these mistakes

`test_op_bwd` (`modules/flash/tests/test_backward.py`) compares against PyTorch's
own low-precision SDPA and accepts a multiple of that reference's error, the
fudge factor. It is 3x-14x on O and 36x-768x on the gradients
(`_compute_fudge_factors` in `_common_test.py`). Those factors exist because
PyTorch's low-precision reference is itself far from exact. But they also leave
room for a whole class of mistakes that cost "only" 3x-15x.

The inputs make it worse. Since #105 the tests use `sm_scale = 1/D` or
`1/sqrt(D)` over N(0, 1) inputs, so the logits have a standard deviation of
0.09-1.0. Most precision mistakes put their error into the **exponent**, where
it scales with `|S|`, so at these logits it is invisible.

A tolerance that keeps having to go up is itself a symptom. 04b5df8c raised dQ's
fudge factor to 180 with the note *"We need to study why dQ requires a much
larger fudge factor."* The answer was mistake 1 below; fixing it (7a165f1e)
brought dQ's factor down to 32.

### What `test_common_mistakes` does instead

`test_common_mistakes` (`modules/flash/tests/test_backward.py`, core in
`_core_test_backward.py`) measures against a **floor**, not against PyTorch.

- **The exact answer:** fp64 attention forward and backward, computed from the
  same fp16/bf16 inputs the kernel got.
- **The floor:** the same fp64 computation, rounded to the input dtype exactly
  where a correct flash kernel has to round and nowhere else:
  - P before `P @ V` and `P^T @ dO`;
  - dS before `dS @ K` and `dS^T @ Q`;
  - every output as it is stored.

  Its distance from the exact answer is the least error any kernel with those
  roundings can have.
- **The checks:**
  - **O, dQ, dK, dV:** relative RMS error at most **2x** the floor. The Triton
    kernels measure 0.7x-1.02x on gfx90a over six seeds.
  - **LSE:** within `2**-16` of the largest score. The Triton kernels use at
    most 4% of that bound, because nothing in the LSE is rounded to the input
    dtype. Rounding anything in the logit domain to fp16 costs `2**-11` of the
    largest score, and to bf16 `2**-8`.
  - **Fully masked rows:** exactly zero in O and dQ.
  - **NaN:** none, anywhere.

Three cases, fp16 and bf16:

| case | inputs | aimed at |
|---|---|---|
| `sharp_softmax` | logits with standard deviation 16 (`q`, `k` ~ N(0, 16), `sm_scale = 1/sqrt(128)`), 257 x 519 | mistakes 1, 2, 5, 6 |
| `zero_sm_scale` | `sm_scale = 0`, top-left causal, 257 x 519 | mistake 7 |
| `bottom_right_masked_rows` | bottom-right causal, 301 x 190: 111 rows attend to nothing, 111 is off every BLOCK_M grid | mistakes 7, 8 |

Each case was shown to catch its mistake by re-introducing that mistake into the
Triton kernels. The JIT twin of the test is in
`modules/flash/kernel/test_backward.py`, and `run_mutants.sh` in this directory
re-runs the whole check ([Reproducing the measurements](#reproducing-the-measurements)).
On gfx90a:

| mistake re-introduced | fails | message |
|---|---|---|
| fwd: Q pre-scaled and rounded (1) | `sharp_softmax`, `bottom_right_masked_rows` | LSE off by 13x-100x its bound on `sharp_softmax`, 2.9x-45x on `bottom_right_masked_rows` |
| bwd: Q (dq kernel) and K (dk/dv kernel) pre-scaled and rounded (1) | `sharp_softmax` | dQ 3.8x-4.6x the floor |
| S rounded to the input dtype (2) | `sharp_softmax`, `bottom_right_masked_rows` | LSE off by 10x-150x its bound |
| LSE stored in base 2 (5) | all three | LSE off by 2.5-34 |
| running max initialised to `-inf` (7) | `bottom_right_masked_rows` | NaN in O |
| no `sm_scale == 0` guard in the backward (7) | `zero_sm_scale` | NaN in dQ |
| Delta reduced in the input dtype (6) | `sharp_softmax` | dQ 2.6x-2.8x the floor |

Running it:

```bash
# AOTriton build; FWD_IMPL/BWD_IMPL pin the backend under test
PYTHONPATH=install_dir/lib/ pytest modules/flash/tests/test_backward.py -k test_common_mistakes
# Triton JIT, against the kernel sources directly
cd modules/flash/kernel && TRITON_F32_DEFAULT=ieee pytest test_backward.py -k test_common_mistakes
```

## Precision

### 1. Rounding a scaled operand to the input dtype

**Mistake.** Folding the softmax scale into an input tile before the GEMM:

```python
qk_scale = sm_scale * 1.44269504089
q = (q * qk_scale).to(q.type.element_ty)   # rounded to fp16/bf16 here
qk = tl.dot(q, k)
p = tl.math.exp2(qk - m)
```

It saves one fp32 multiply per score, which is why it keeps coming back.

**Why it is wrong.** `q * qk_scale` is not representable in the input dtype, so
each element picks up a relative error of up to `2**-8` in bf16 (`2**-11` in
fp16). That error passes through the dot product into `S` and then into the
**exponent**, where it becomes a relative error in P of about `ln(2) * |error in
S|`. That grows with the logits. The correct form rounds nothing:

```python
qk += qk_scale * tl.dot(q, k)   # fp32 accumulator, fp32 scale
```

The backward makes it worse. The backward recomputes P from the LSE the forward
saved. If the forward rounded a scaled Q and the backward rounds a scaled K (or
does not round at all), the two P's disagree and the recomputed rows no longer
sum to 1. All three backward kernels used to do this: the forward scaled Q, the
dq kernel scaled Q, and the dk/dv kernel scaled K.

**Exposed by:** large logits. Either large-magnitude Q and K or a large
`sm_scale` will do; bf16 more than fp16. Measured on `sharp_softmax`:
- Forward pre-scaling: LSE off by 1.5e-2 (fp16) to 1.2e-1 (bf16), where the
  correct kernel is off by 2e-5. O is 3.3x the floor, dQ/dK 4.2x-4.8x, dV
  13x-15x.
- Backward pre-scaling alone: the LSE and O are untouched; dQ/dK are 3.8x-5.0x
  the floor and dV 14x-17x.

**History.** Present from the first kernel. Removed in 7a165f1e (#46, PR #45:
*"Fix numerical errors from scaling input tensors with log_2(e) as
preprocessing"*), for the forward and both backward kernels.

**Guard.** `test_common_mistakes[sharp_softmax]`.

### 2. Rounding any logit-domain value to the input dtype

**Mistake.** The same error by another route: keeping `S` in fp16/bf16 before
the softmax, for example staging it through LDS at the input width, or
requesting a low-precision GEMM output. Anything on the path from `S` to
`exp2(...)` that is rounded to the input dtype puts an error of up to
`2**-8` (bf16) or `2**-11` (fp16) of `|S|` into the exponent.

**Exposed by / guard.** As mistake 1: `test_common_mistakes[sharp_softmax]`
shows LSE off by 2e-2 (fp16) to 1.8e-1 (bf16).

**Related.** An unmerged fix for gfx1151
(`upstream/fix/gfx1151-bf16-dot-fp32-accumulator`) adds an explicit
`tl.dot(..., out_dtype=tl.float32)` for the same reason: request the fp32
accumulator explicitly, because a `.to(tl.float32)` after the dot is too late.

### 3. The max and the exponent computed from differently rounded values

**Mistake.** Taking the running max of one rounding of the scaled scores and
exponentiating another:

```python
m_ij = tl.maximum(m_i, qk_scale * tl.max(qk, 1))     # multiply, rounded
p = tl.math.exp2(qk * qk_scale - m_ij[:, None])      # compiles to an FMA: rounded once
```

**Why it is wrong.** For the row maximum, `qk * qk_scale - m_ij` should be
exactly 0. It is not: the FMA rounds once and the multiply rounds separately.
The difference can be up to half an ulp of `qk * qk_scale`, and at large scores
an ulp is huge. Issue #54 works it through: bf16 inputs of 133120, `sm_scale = 0.25`,
give `qk * qk_scale ~= 6.4e9`. The "zero" comes out as 247.375, so
`exp2(247.375) = inf`, and the output is NaN.

**Correct form.** Scale once, then derive both the max and the exponent from
that one rounded tensor:

```python
qk += qk_scale * tl.dot(q, k)
m_ij = tl.maximum(m_i, tl.max(qk, 1))
p = tl.math.exp2(qk - m_ij[:, None])
```

**Exposed by:** very large logits. Output is inf/NaN.

**History.** Introduced by 7a165f1e (as the fix for mistake 1). Fixed in 0.7.3b
and 6f8cbcac (#57). `fwd_kernel_inner.py` carries a *"DO NOT USE the following
FMA optimization pattern"* comment.

**Still open.** The backward still computes `exp2(qk_scale * qk - l_i)` against
the LSE the forward produced from a separately rounded product. It is marked
`FIXME: Potential bug https://github.com/ROCm/aotriton/issues/54` in
`bwd_inner_dq.py`, `bwd_inner_dk_dv.py` and `bwd_inner_fuse.py`. Do not copy
it.

**Guard.** `test_large_bf16_nan_values` (forward only).

### 4. Constants multiplied in low precision

**Mistake.** `qk += bias * 1.44269504089` with a bf16 `bias`. A Python literal
takes the tensor's dtype, so `log2(e)` is rounded to 1.4453125 before the
multiply. The backward applied the same factor in fp32.

**Why it is wrong.** The backward recomputes P from the LSE the forward saved, so
the two must build the log2-domain logits bit-identically. With a
bias-dependent mismatch, P comes back scaled by
`2**(bias * (log2e_fp32 - log2e_bf16))`: 0.973 at bias 16, 0.891 at bias 64.

**Correct form.** `qk += bias.to(qk.dtype) * 1.44269504089`, and the same for
ALiBi. In general, every scalar multiplied into a low-precision tensor runs at
that tensor's precision: upcast first.

**Exposed by:** large bias magnitudes, especially bf16. The random biases in
`test_op_bwd_with_matrix_bias` are too small to show it.

**History.** Introduced with bias support in 0f51e916 (#14). Re-introduced in the
forward by 6f8cbcac (#57) while fixing mistake 3. Fixed in 44a2d9a6 (#228).

**Guard.** `test_matrix_bias_fwd_bwd_symmetry`: a single key, so softmax is
exactly 1 whatever the bias; asserts LSE == bias, O == V and dV == dO to 1e-5.

### 5. The LSE in the wrong base or units

**Mistake.** Storing `m_i + log2(l_i)` (base 2, the kernel's internal domain),
or an LSE without the `sm_scale`, or one without the bias.

**Why it is wrong.** The LSE is an output. PyTorch's context parallelism merges
partial attention with it, and other backends' backward kernels read it. A
forward and backward that agree with each other will still pass every
O/gradient test with a base-2 LSE; only a check on the LSE value itself sees it.
The contract is the natural-base `logsumexp(sm_scale * S + bias)`. The forward
multiplies by `ln(2)` on store and the backward by `log2(e)` on load.

**Exposed by:** any input, once something checks the LSE. It is off by a factor
of 1.44.

**History.** Base 2 until 52a37783 (#108).

**Guards.**
- `test_logsumexp_scaling`: Q = K = V = I, `sm_scale = 1/4`, so every row's LSE
  is `ln(15 + e**0.25)`.
- `test_common_mistakes`: checks the LSE of every case to fp32 accuracy.

### 6. Delta in low precision

**Mistake.** Reducing `Delta = rowsum(dO * O)` in the input dtype, for example
with a low-precision dot or a bf16 accumulator.

**Why it is wrong.** `dS = P * (dP - Delta)`. `dP` and `Delta` are close for the
dominant keys of a sharp softmax, so an absolute error in `Delta` becomes a large
relative error in `dS`, and from there in dQ and dK. dV does not use `Delta` and
stays clean, which is the telltale sign. The Triton kernels compute it with
`composed_inner_product_fp32` (upcast, multiply, sum in fp32) from the stored O.

**Exposed by:** a sharp softmax. Measured on `sharp_softmax`: dQ and dK at
2.6x-2.8x the floor, dV at 1.0x.

**Guard.** `test_common_mistakes[sharp_softmax]`.

## Infinities, NaN and masked rows

### 7. `-inf` arithmetic

Masking writes `-inf` into the scores. Three different ways to turn it into a NaN
have each shipped.

- **Running max initialised to `-inf`.** On a row with no unmasked key in the
  block, `m_i - m_ij` and `qk - m_ij` evaluate `-inf - (-inf) = NaN`.
  - Fix: initialise `m_i` to `-3.40282e+38` (finite) and `l_i` to 1.0. A fully
    masked row then stays at `p = 0`.
  - Fixed in 6f8cbcac (#57).
  - Guard: `test_common_mistakes[bottom_right_masked_rows]`, which fails with
    NaN in O.
- **`sm_scale == 0`.** If the mask is applied before the scale, a masked score
  is `-inf * 0 = NaN`. The Triton forward adds the scaled dot product *onto* the
  `-inf`, which is safe. The backward masks first and scales inside `exp2`, so
  it needs `if qk_scale == 0.0: p = tl.where(isnan(p), 0.0, p)`.
  - Fixed in aef9087d (#48).
  - Guard: `test_common_mistakes[zero_sm_scale]`, which fails with NaN in dQ.
    `test_op_bwd` never uses `sm_scale = 0`.
- **Dividing by `sm_scale`.** The backward still computes the bias term as
  `qk += bias * (1.0 / sm_scale)` and then multiplies everything by `qk_scale`.
  **This is still broken:** with a matrix bias and `sm_scale = 0` the Triton
  backward returns NaN in dQ, dK and dV. That is measured on the JIT kernels,
  non-causal, at 128 x 128 and 257 x 519, while the forward is correct.
  `test_common_mistakes` has no case for it because it would fail on the Triton
  kernels today. Do not copy it:
  multiply the bias by `log2(e)` directly, as the forward does.

### 8. Fully masked rows

With bottom-right causal attention and `seqlen_q > seqlen_k`, the first
`seqlen_q - seqlen_k` query rows attend to nothing. The contract is `O = 0`,
`LSE = +inf` and `dQ = 0`. The LSE is `+inf` and not `-inf` so that the
backward's `exp(S - LSE)` is 0.

What can go wrong, and has:

- **A whole tile of such rows takes an early exit.** The exit must still write
  `O = 0` and `LSE = +inf` for the tile.
  - In a persistent kernel, the exit must not leak into the next tile. Triton
    has no `return` inside a `while`, so the Triton kernel emulated one with
    `continue_condition = False` and never re-armed it. Every later tile the same
    workgroup claimed was skipped, leaving Out and LSE unwritten.
  - This needs more tiles than workgroups, so it looked nondeterministic.
  - Fixed in eb3638e1 (#235, #237).
  - Guard: `test_bottom_right_fully_masked_rows`, which sizes the batch from the
    CU count.
- **Such rows share a tile with live rows.** They go through the normal loop and
  meet mistake 7 above.
  - Guard: `test_common_mistakes[bottom_right_masked_rows]`. Its 111 masked rows
    are off the grid of every BLOCK_M.
  - Known divergence: the Triton forward stores `(-FLT_MAX + log2(1)) * ln(2) ~=
    -2.36e38` as the LSE of these rows, not `+inf` (comment in
    `fwd_kernel.py`). Nothing reads it today, because the backward re-masks
    those elements, so `test_common_mistakes` only asserts that it is not NaN. A
    new kernel should store `+inf`.
- **Early exits in the backward.** A dq or dk/dv program with no work must still
  store zeros for the block it owns. The buffers come from `torch.empty`, so a
  skipped store is garbage, not zero. Fixed in 406b3c64 (#55).

## Dropout

### 9. Dropout

- **Sum P before the mask.** `l_ij = tl.sum(p, 1)` is taken **before** dropout
  zeroes elements ("CAVEAT: Must update l_ij before applying dropout"). `1/(1-p)`
  multiplies the output once, in the epilogue.
- **The backward uses the mask asymmetrically.**
  - dV accumulates the *dropped and rescaled* P.
  - `dP = dO V^T` gets the mask and `1/(1-p)`.
  - `dS = P * (dP - Delta)` uses the *undropped* P.
  - `Delta` sits outside the mask. Before bf27b09e the kernel folded `-Delta`
    into the `dP` accumulator, where the mask would have hit it.
- **The mask must be bit-identical in every kernel.** Forward, dq, dk/dv and the
  fused kernel must use one Philox offset formula:
  - The stride is tied to `Max_seqlen_k`, not the per-sequence length (varlen).
  - Offsets are 64-bit. Before 0f430532 (#71) they were 32-bit and wrapped once
    `B * H * Sq * Sk` reached `2**32`.
  - The per-head base is built from the batch and head indices. 85d120cd built
    it from a varlen `batch_index` that is always 0; 6f8cbcac fixed it.
- **The word order is part of the contract.** `tl.join` appends a minor axis, so
  joining `(r0, r1), (r2, r3)` lays the words out as `r0 r2 r1 r3`.
  - 204fd112 (#226) fixed the order to `r0 r1 r2 r3`, so column `4n + i` is
    `r_i`. A new kernel that must reproduce Triton's mask has to match it.
  - No mask-derived test can see this: they all take the reference mask from
    the kernel under test.
  - Guard: `modules/flash/kernel/test_dropout_layout.py`, against an independent
    host-side Philox-4x32-10.
- **Guard for the rest:** `test_op_bwd` with `dropout_p = 0.5`. These mistakes
  are large enough for the fudge factors to catch.

## Addressing

### 10. Every tensor has its own strides

Most of the non-numerical bugs in the history have one form: tensor A addressed
with tensor B's strides, silently correct as long as the tests give A and B the
same layout.

| tensor | borrowed | fixed in | exposed by |
|---|---|---|---|
| V | Q's strides (old fused bwd) | 5ab01f96 | V's layout != Q's |
| dK, dV, dQ | K's, V's, Q's | 9044fe5e (#8) | gradient layouts != input layouts |
| dO | Q's strides and base | d1d5bcb9 | dO from a transpose |
| O, dO in `bwd_preprocess` | none: contiguous assumed | 1884ade1 | BSHD inputs |
| dO | O's strides (split kernels) | fbb36df1 (#95) | dO's layout != O's |
| V (forward) | K's sequence stride | 8625c4fa (#123) | a real model (DINOv3) with K and V laid out differently |
| dO (fused dk/dv) | O's strides | eb3638e1 (#236) | ring attention: O BHSD, dO BSHD |
| batch, head | `off_hz * stride_h` | 1884ade1 | any non-BHSD tensor |

**Guard.** `test_memory_layouts` assigns every tensor, outputs included, its own
outer-axis permutation, round-robin.

Store each output in its own dtype, too: 81e3ee0e found every cast hard-coded to
fp16, and f9bb46f2 stored dV in dK's dtype.

### 11. Padding

- **Sequence padding must be `-inf`, not zero.** A zero-padded K column gives
  `S = 0`, and `exp(0 - m)` still enters the denominator. Mask the scores,
  guard the LSE and Delta loads, and mask the stores. Fixed in 81ca5e35 (#7).
  Guard: the odd sequence lengths of `test_fast` and `test_irregulars`.
- **Head-dimension padding: mask every sub-block, not only the last.** A kernel
  that splits a non-power-of-two head dimension into power-of-two pieces cannot
  assume only the trailing piece is ragged once `hdim_qk != hdim_vo`. Fixed in
  0373139c (#135).
- **The 8xD input contract.** Loads and stores are 8 columns wide, so a kernel
  touches `ceil8(hdim)` columns of every row. A buffer descriptor whose range
  ends at `hdim` makes the hardware drop the dword holding the last real column
  and the first pad column. The last element of every slab is then never
  written or read. Fixed for the gfx950 flyc kernels in 53b677b7 (#231).
  Guard: `test_prime_hdim` and `test_memory_layouts` fill the slack with NaN.
  A mask that multiplies by zero instead of discarding hides a finite leak, but
  not a NaN.

### 12. GQA: per-query-head state

Everything indexed by the query head must use `off_h_q`, and only K, V, dK and
dV use `off_h_k`. That means the Q tile, dO, LSE, Delta, the bias, the dropout
offset, dQ and dB.

- 45708bb7 (#49): the forward's K/V head index was inverted, and the dk/dv
  kernel built LSE, Delta and Philox offsets with `H_K` in place of `H_Q`.
- 68a82508 (#170): the dk/dv kernel pointed the bias at `off_h_k` once, outside
  the per-query-head loop, so a whole group read one head's bias.

**Guard.** `test_fast` with `N_HEADS=(10, 2)` and a matrix bias.

### 13. Causal alignment and bounds

- **One alignment in forward and backward.** 85d120cd shipped a bottom-right
  forward and a top-left backward, hidden because causal tests with
  `seqlen_q != seqlen_k` were skipped. 406b3c64 (#55) re-aligned them.
- **Loop bounds.** Align the causal loop start to the block grid of the loop
  axis (f9bb46f2) and clamp to the other sequence length. Use floor division for
  window bounds, which can be negative; Triton's `//` truncates toward zero
  (fd4048c9, #96).
- **Grids.** Size each backward grid by the axis it owns: dk/dv by `seqlen_k`,
  dq by `seqlen_q` (6558b6af).
- **Keep BLOCK_M and BLOCK_N apart.** 14642800 and f9bb46f2 each confused them.
  Tests with `BLOCK_M == BLOCK_N` cannot see it.

### 14. Varlen

- **Row stride is the allocated pitch, never the logical length.** Delta was
  addressed with `seqlen_q` as its row stride instead of `max_seqlen_q`
  (04cdead5, #150).
- **Mask stores with this sequence's length.** The fully-masked-row `+inf` LSE
  store was masked with `Max_seqlen_q`, so in the compact layout it overwrote the
  next sequence's LSE (c4cfe6c1, #222).
- **Compute offsets in 64 bits.** `(B * H) * S` overflows int32 (0147577c, #149).

**Guard.** `test_varlen.py`.

### 15. Optional outputs

A bias whose gradient is not wanted arrives as a null `dB` with all-zero
strides; PyTorch passes this for every bool-masked SDPA. The kernel must test the
strides before storing. 00ccbf3c (#22) required **all** strides to be zero for
"no dB" (it had been "any"). The flyc gfx1201 dq kernel wrote `dB`
unconditionally until 1c7c973c (#239).

**Guard.** `test_fast` with `bias_type='matrix,nograd'`.

## Not mistakes: what the hardware does

- **gfx90a's fp16 MFMA flushes subnormal inputs to zero.** With `seqlen_k` in the
  hundreds, many P values fall below fp16's smallest normal (`2**-14`), and dV
  measured 2.0x-2.5x a floor that keeps them. A floor that flushes them
  reproduces the kernel's dV to 3e-5, so it is the hardware, not the kernel
  (`ftz_check.py`). `test_common_mistakes` takes the larger of the two floors.
- **fp32 accumulation order.** A different reduction order changes the result by
  fp32 rounding. That is far below every tolerance above, and it is why
  `test_common_mistakes` does not run fp32 inputs: there, rounding to the input
  dtype is harmless, and fp32 accumulation error is not in the floor model.

## Reproducing the measurements

Everything in this directory runs the Triton JIT kernels in
`modules/flash/kernel/` on a GPU, with `TRITON_F32_DEFAULT=ieee`.

| file | what it shows |
|---|---|
| `mutants.py` | re-introduces each historical mistake into a copy of `modules/flash/kernel/`, written under `$MUTANTS_DIR` (default `/tmp/aotriton-mutants`). Every replacement must match exactly once, so a mutant the kernels have moved past fails loudly instead of testing nothing |
| `run_mutants.sh` | runs `test_common_mistakes` on the pristine kernels, which must pass, and on every mutant, which must fail; it prints the failure messages the table above quotes |
| `margins.py` | the pristine kernels' worst error/floor ratio and LSE margin over six seeds, per case: the 0.7x-1.02x and 4% quoted above |
| `ftz_check.py` | the fp16 subnormal flush described in [Not mistakes](#not-mistakes-what-the-hardware-does) |

```bash
docs/attention-kernel-numerical-error-lessons/run_mutants.sh
TRITON_F32_DEFAULT=ieee python docs/attention-kernel-numerical-error-lessons/margins.py
TRITON_F32_DEFAULT=ieee python docs/attention-kernel-numerical-error-lessons/ftz_check.py
```

## Adding a case

1. Add the mistake to `mutants.py` and confirm with `run_mutants.sh` that the
   case fails on it, and with `margins.py` that the pristine kernels pass with
   room to spare.
2. Keep the JIT twin in `modules/flash/kernel/test_backward.py` and the core in
   `modules/flash/tests/_core_test_backward.py` in step. The functions are meant
   to be identical, apart from the pytest wrapper.
3. Add the entry here, with the commit that fixed it and the numbers the case
   measured.
