# Generalized Sliding Window Attention

Sliding window attention (SWA) keeps, for each query row, a band of keys around
a diagonal: `window_left` keys to the left of it and `window_right` keys to the
right. In flash-attention a window size is either a non-negative bound or
"unbounded": a negative value (conventionally `-1`) removes that side of the
band.

AOTriton's windows are **generalized** (gSWA): every integer is a literal bound,
negative ones included. The only exceptions are the two sentinels
(`TopLeftAligned`, `BottomRightAligned`). A negative bound moves that edge of
the band past the diagonal, so the band need not contain the diagonal at all.
For example, `(window_left, window_right) = (-1, 3)` keeps `j ∈ [i+1, i+3]`,
where flash-attention would read the `-1` as "no left limit". Causal masking is
expressed through the same mechanism, as a window.

This document specifies flash-attention's behavior and AOTriton's, and shows how
to reproduce flash-attention's exact mask with AOTriton. That needs two things:
translating FA's negative "unbounded" values, and choosing the diagonal that
windows are measured from (see `principal_diagonal_choice`).

1. [Reference: flash-attention's behavior](#reference-flash-attentions-behavior)
2. [AOTriton's behavior](#aotritons-behavior)
3. [Matching flash-attention](#matching-flash-attention)
4. [Implementing PyTorch's flash attention ops with AOTriton](#implementing-pytorchs-flash-attention-ops-with-aotriton)

Notation used throughout:

* `i`: query row, 0-based; `j`: key column, 0-based.
* `sq`, `sk`: query and key lengths **of the sequence being processed**. Under
  varlen these are per sequence, not `max_seqlen_*`.
* `d = sk - sq`: offset of the bottom-right diagonal from the top-left one.
* Masks are drawn with `1` = kept.

---

## Reference: flash-attention's behavior

This section describes what flash-attention (FA) does, as established by
reading its source at commit `1dc8232f`. FA ships several implementations
behind one Python API, and they mostly agree. Where they differ, this section
says so. The names used below:

* **FA2**: the original CUDA kernels, `csrc/flash_attn/`.
* **FA3**: the Hopper kernels, `hopper/`.
* **cute**: the CuTe-DSL kernels, `flash_attn/cute/`.
* **Triton-AMD**: the Triton backend FA uses on AMD GPUs, which lives in aiter.
  FA pins aiter at `f0966b0`, which was not available, so the Triton-AMD
  statements were checked against aiter `a7c3b96e6` and may differ in detail.

Code locations are given in a "Where in the code" note after each part, for
readers who want to verify. The behavior is stated in prose and code
snippets so the section can be read without them. The masks in the examples
were worked out by hand from the code, not produced by running FA.

### FA-1. Which keys a query row sees

FA aligns its window to the **bottom-right** corner of the attention matrix.
Imagine the `sq × sk` score matrix with the last query row lined up against the
last key column. The diagonal through that corner, `j = i + d` with
`d = sk - sq`, is the reference line. `window_size_left = wl` allows up to `wl`
keys to the left of that line, and `window_size_right = wr` up to `wr` keys to
its right. A negative size means that side has no limit.

As code, after the API has normalized the window (FA-3):

```python
def fa_keeps(i, j, sq, sk, wl, wr):
    """True if query row i attends to key column j.  wl/wr < 0 = unbounded."""
    d = sk - sq                  # bottom-right diagonal is j == i + d
    if not (0 <= j < sk):
        return False
    if wl >= 0 and j < i + d - wl:
        return False             # too far left of the diagonal
    if wr >= 0 and j > i + d + wr:
        return False             # too far right of the diagonal
    return True
```

Consequences worth spelling out:

* **Causal is window `(-1, 0)`**: no left limit, nothing right of the diagonal,
  i.e. `j <= i + sk - sq`. The last query row sees every key. With a single
  query (`sq = 1`, decoding) the one row sees all `sk` keys.
* When `sq == sk`, the bottom-right and top-left diagonals coincide, and
  `[i - wl, i + wr]` is the visible range. FA's older docstrings state only this
  case.
* When `sq > sk`, the first `sq - sk` rows of a causal mask see **no key at
  all**. FA-4 covers what they produce.
* Every backend implements the same function. They differ only in how they
  compute it (e.g. as a `[left, right)` column range per row), and the forward
  and backward masks are identical.
* FA switched from top-left to bottom-right alignment in v2.1; the README's
  §2.1 records the change.

Where in the code:
FA2 `csrc/flash_attn/src/mask.h:53-61, 79-81, 172-197`;
FA3 `hopper/mask.h:88-124` (forward) and `128-159` (backward);
cute `flash_attn/cute/mask.py:297-360, 714-753`;
Triton-AMD `fwd_prefill.py:384-456`, `bwd.py:2957-2985`.
The CK path (`csrc/flash_attn_ck/mha_fwd.cpp:219-231`) builds a bottom-right
mask description `"b:<wl>,<wr>"`; the CK kernels themselves were not checked.

### FA-2. Varlen: the diagonal is placed per sequence

In a varlen batch each sequence has its own `sq` and `sk`, and so its own
diagonal offset `d = sk - sq`. FA computes `d` from each sequence's **actual**
lengths, inside the kernel. These come from the differences of `cu_seqlens_*`,
or from `seqused_k` when given, adjusted for left padding and appended KV.
`max_seqlen_q` / `max_seqlen_k` never affect where the diagonal is. They only
appear in the host-side normalization of FA-3.

So in one varlen call, a decode sequence (`sq = 1, sk = 100`) and a prefill
sequence (`sq = sk = 50`) each get a correctly bottom-right-aligned window. A
single host-side number could not express this, because `d` is 99 for the
first and 0 for the second.

Where in the code:
FA2 `src/block_info.h:17-24`, used by `src/flash_fwd_kernel.h:90-94, 295, 846`
and `src/flash_bwd_kernel.h:105, 320, 523-529`;
FA3 `hopper/seqlen.h:49-54, 79-88`, `hopper/block.h:21-28`;
cute `flash_attn/cute/seqlen_info.py:132-146`;
Triton-AMD `fwd_prefill.py:1028-1061`.

### FA-3. What the API does to the window before the kernel sees it

The kernels never see FA's `-1` directly. The host code rewrites
`(is_causal, wl, wr)` first, and the rewriting differs between FA2 and FA3.
The snippets below give the effective order of operations.

**FA2, forward:**

```python
def fa2_normalize(is_causal, wl, wr, seqlen_q, seqlen_k, has_alibi):
    # seqlen_q/k are max_seqlen_q/k in the varlen API
    if wl >= seqlen_k: wl = -1          # a bound this wide cannot exclude anything
    if wr >= seqlen_k: wr = -1
    if seqlen_q == 1 and not has_alibi:
        is_causal = False               # one query sees every key anyway
    if is_causal:
        wr = 0
    # set_params_fprop, shared with the backward:
    is_causal = (wl < 0 and wr == 0)    # causal is RE-DERIVED from the window
    if wl < 0 and wr >= 0: wl = seqlen_k  # numbers standing in for "unbounded"
    if wl >= 0 and wr < 0: wr = seqlen_k
    is_local = (wl >= 0 or wr >= 0) and not is_causal
    return is_causal, is_local, wl, wr
```

What this means for a caller:

* `causal=False, window=(-1, 0)` runs exactly as `causal=True`. The flag and
  the window are two spellings of the same thing.
* `causal=True` with a window `(wl, wr)` keeps `wl` and forces `wr = 0`.
* Causal and local are separate kernel variants. A window that has any finite
  side runs the local variant, even when it also touches the diagonal.
* For decoding with grouped-query attention (one query, more query heads than
  KV heads, no window, no dropout, no ALiBi, head size a multiple of 8), FA2
  reshapes Q so that the query heads of a group become rows
  (`seqlenq_ngroups_swapped`). This only changes performance; it is valid
  because no mask is active.
* In the varlen forward the `is_causal → wr = 0` step and the reshape check run
  before the two clamps. That only affects whether the reshape is used.
* The backward applies `is_causal → wr = 0` and then the clamps, and has no
  `seqlen_q == 1` shortcut.

**FA3, forward:**

```python
def fa3_normalize(is_causal, wl, wr, seqlen_q, seqlen_k, attention_chunk=0):
    # seqlen_q/k are max_seqlen_q/k in the varlen API
    if wl >= seqlen_k - 1: wl = -1
    if wr >= seqlen_q - 1: wr = -1      # note: compared with seqlen_q, not seqlen_k
    if seqlen_q == 1 and wl == -1 and wr == -1 and attention_chunk == 0:
        is_causal = False               # (kept True for one paged-KV case; same result)
    if is_causal:
        wr = 0
    # set_params_fprop:
    is_causal = (wl < 0 and wr == 0 and attention_chunk == 0)
    is_local = (wl >= 0 or wr >= 0 or attention_chunk >= 1) and not is_causal
    if wl < 0: wl = seqlen_k - 1        # numbers standing in for "unbounded"
    if wr < 0: wr = seqlen_q - 1
    return is_causal, is_local, wl, wr
```

The backward clamps the same way and also re-derives `is_causal` from the
window. FA3 has no Q reshape for decoding; it packs GQA heads into rows inside
the kernel instead (PackGQA), and maps a packed row back to its real query
index before masking.

**cute** uses `None`, not a negative number, for "unbounded":

```python
def cute_normalize(causal, wl, wr):
    if wl is not None and wl < 0 and wr is not None and wr < 0:
        wl = wr = None                  # both negative: no window
    if causal:
        wr = 0
    if wl is None and wr == 0:
        return "causal"
    return "local" if (wl is not None or wr is not None) else "none"
```

A single negative side, e.g. `(-1, 3)`, is **not** turned into `None`. It
reaches the mask as the number `-1`, which moves the left edge one key to the
right of the diagonal instead of removing it. cute callers must pass `None` for
an unbounded side.

Where in the code:
FA2 `csrc/flash_attn/flash_api.cpp:156-168` (`set_params_fprop`), `423-438`
(dense forward), `616-640` (varlen forward), `829, 879-880` and `1045, 1098-1099`
(backward), `1315-1329` (KV cache); kernel variant selection
`src/flash_fwd_launch_template.h:34, 70, 79`.
FA3 `hopper/flash_api.cpp:146-159, 796-805, 1360-1365`;
`hopper/flash_attn_interface.py:353-372`.
cute `flash_attn/cute/interface.py:280-301`, `mask.py:338`.

#### FA-3.1. When a bound is effectively unbounded, and where FA2 gets it wrong

Both FA2 and FA3 turn a large bound into `-1` and later turn `-1` back into a
number. That is harmless only if the thresholds and the substitutes are at
least as large as the point where a bound stops excluding anything. That point
is easy to derive from `fa_keeps`:

* **Left.** `j >= i + d - wl` excludes no column `j >= 0` for any row iff it
  holds for the last row `i = sq - 1`: `sq - 1 + sk - sq - wl <= 0`, i.e.
  **`wl >= sk - 1`**.
* **Right.** `j <= i + d + wr` admits every column `j <= sk - 1` for every row
  iff it holds for the first row `i = 0`: `sk - sq + wr >= sk - 1`, i.e.
  **`wr >= sq - 1`**.

FA3 clamps at exactly these thresholds (`wl >= seqlen_k - 1`,
`wr >= seqlen_q - 1`) and substitutes exactly these values, so FA3 is exact.
The thresholds are tight: one less and the bound does exclude a key. For
example, `(0, 0)` keeps only the diagonal element `j = i + d`, and FA3 leaves
it alone unless `sk <= 1` or `sq <= 1`, where the window really excludes
nothing.

FA2 uses `seqlen_k` both as the clamp threshold and as the substitute on both
sides. On the left that is merely looser than needed (`sk >= sk - 1`), so it is
still exact. On the right it is exact only while `sk >= sq - 1`. When the
query sequence is more than one longer than the key sequence (`sq > sk + 1`),
FA2's mask differs from `fa_keeps`:

* `(wl, -1)` with `wl >= 0`: `wr` becomes `sk`, too small, so the early
  rows lose keys they should see (example D).
* `(-1, wr)` with `sk <= wr < sq - 1`: the clamp turns `wr` into `-1`, which
  removes a right limit that did exclude columns.

Triton-AMD does not clamp at all, and is exact. CK clamps like FA2 (`csrc/flash_attn_ck/mha_fwd.cpp:212-213`).
This was established by reading the code, not by running it.

### FA-4. Query rows that see no key

When a row's window contains no valid key (for example the first `sq - sk`
rows of a causal mask with `sq > sk`), every FA backend writes an **output row
of zeros**. The docstrings promise only that.

The log-sum-exp (LSE) of such a row has no meaningful value, because the sum
it is the logarithm of is empty. Backends write different placeholders, and
**the sign does not matter**. Each backend's own backward pass gets zero
probability for the row from its own placeholder (FA-5), and a caller has
nothing to read from it. For reference:

| Backend | LSE of a row that sees no key |
|---|---|
| FA2 | `+inf` |
| FA3 | `-inf` (but `+inf` when `seqlen_k == 0`, which the host fills in) |
| cute | `-inf` |
| Triton-AMD | `-inf` for a single empty row; `0` when a whole tile of rows is empty |

Where in the code:
FA2 `src/flash_fwd_kernel.h:99-133, 1196-1204`, `src/softmax.h:179-180`,
`flash_api.cpp:526-528`;
FA3 `hopper/softmax.h:144, 151`, `hopper/epilogue_fwd.hpp:441, 447`,
`hopper/flash_api.cpp:1190-1193`;
cute `flash_attn/cute/softmax.py:276-285`;
Triton-AMD `fwd_prefill.py:1107-1138, 1411-1433`;
docstrings `flash_attn/flash_attn_interface.py:1109, 1185, 1424`.

### FA-5. Backward

The backward uses the same `fa_keeps` mask with the same per-sequence `d`, so
gradients flow exactly where the forward attended. Rows that saw no key get
zero gradient: FA2 reads their `+inf` LSE back so the recomputed probabilities
are 0, and FA3 treats a `-inf` LSE as 0 while the masked scores are `-inf`
anyway. Triton-AMD supports sliding windows in the backward only in its
"fused" mode.

Where in the code:
FA2 `src/flash_bwd_kernel.h:104-105, 318-320, 411, 504-530`;
FA3 `hopper/mask.h:128-159`, `hopper/block.h:83-100`,
`hopper/flash_bwd_preprocess_kernel.h:228`;
cute `flash_attn/cute/mask.py:376-420`;
Triton-AMD `bwd.py:2957-2985, 4664-4668`.

### FA-6. Summary of window values, and related features

| Window `(wl, wr)` | Keys visible to row `i` (`d = sk - sq`) |
|---|---|
| `(-1, -1)` | all keys (or causal, if `causal=True`) |
| `(-1, 0)` | `j <= i + d`, i.e. causal |
| `(-1, wr)` | `j <= i + d + wr` |
| `(wl, -1)` | `j >= i + d - wl` (but see FA-3.1 for FA2) |
| `(wl, wr)` | `i + d - wl <= j <= i + d + wr`, at most `wl + wr + 1` keys |

Any negative value behaves like `-1`, except in cute (FA-3).

Features that interact with the window:

* **ALiBi** measures distance from the same bottom-right diagonal: the bias is
  `-slope * |i + sk - sq - j|`.
* **FA3 `attention_chunk`** additionally restricts a row to keys in its own
  chunk, with chunks laid out along the bottom-right diagonal.
* **FA3 `sink_token_length`** keeps the first `sink` keys visible to every row
  regardless of the left bound.
* Neither FA3 feature applies to the backward.

The internal name for "has a window" is `Is_local` in FA2, `is_local` in FA3,
`local` in cute and `USE_SLIDING_WINDOW` in Triton-AMD.

Where in the code: ALiBi `src/mask.h:184`; `attention_chunk` and sinks
`hopper/mask.h:106, 116-124`; README `249-250, 275-277, 435-455`.

### FA-7. Worked examples

Masks drawn with `1` = kept, rows are queries, columns are keys.

**A. Decoding: `sq = 1, sk = 5`, causal.** `d = 4`, so the single row keeps
`j <= 4`, every key:
```
q0: 1 1 1 1 1
```

**B. Chunked prefill: `sq = 3, sk = 5`, window `(1, 0)`.** `d = 2`, so row `i`
keeps `j ∈ [i+1, i+2]`. A top-left alignment (`d = 0`) would give a different
mask, shown on the right:
```
      k0 k1 k2 k3 k4        top-left, for comparison
q0:    0  1  1  0  0        1 0 0 0 0
q1:    0  0  1  1  0        1 1 0 0 0
q2:    0  0  0  1  1        0 1 1 0 0
```

**C. More queries than keys: `sq = 5, sk = 3`, causal.** `d = -2`, so row `i`
keeps `j <= i - 2`. The first two rows see nothing and produce zero output:
```
      k0 k1 k2
q0:    0  0  0
q1:    0  0  0
q2:    1  0  0
q3:    1  1  0
q4:    1  1  1
```

**D. FA2's right-side substitute (FA-3.1): `sq = 5, sk = 2`, window `(0, -1)`.**
`d = -3`. As defined, row `i` keeps `j >= i - 3`. FA2 replaces `wr = -1` with
`wr = 2`, which also imposes `j <= i - 1`:
```
as defined (FA3, Triton-AMD)    FA2
q0: 1 1                         q0: 0 0
q1: 1 1                         q1: 1 0
q2: 1 1                         q2: 1 1
q3: 1 1                         q3: 1 1
q4: 0 1                         q4: 0 1
```

---

## AOTriton's behavior

### A-1. Parameters

| Field | Meaning |
|---|---|
| `causal_type` | `CausalType::None` (0): no mask. `CausalType::WindowedAttention` (3): gSWA, using the two windows below. Values 1 and 2 exist only inside the Triton kernel and are not compiled into the shipped binaries. |
| `window_left`, `window_right` | i32 each. Either a number or one of the sentinels `WindowValue::TopLeftAligned` (`0x80000001`) / `WindowValue::BottomRightAligned` (`0x80000002`). Each side is resolved independently, so a sentinel and a number may be mixed. |
| `varlen_bits.principal_diagonal_choice` | 1 bit (bit 18 of the varlen word). `VarlenPrincipalDiagonal::TOP_LEFT` (0, default) or `BOTTOM_RIGHT` (1). Selects the diagonal that **numeric** windows are measured from. Same field in `attn_fwd_params` and `attn_bwd_params`; a forward and its backward must agree. |

### A-2. Mask predicate

With `causal_type = WindowedAttention`, row `i` keeps column `j` iff

```
0 <= j < sk  AND  i - L <= j <= i + R
```

where `L`, `R` are resolved **per sequence** (from that sequence's `sq`, `sk`)
inside the kernel, with `p = principal_diagonal_choice`:

| side value | resolved `L` | resolved `R` |
|---|---|---|
| `TopLeftAligned` | `sq` (unbounded) | `0` |
| `BottomRightAligned` | `sq` (unbounded) | `sk - sq` |
| number `w` | `w - p·d` | `w + p·d` |

So with `p = 0` a numeric window is measured from the top-left diagonal `j = i`.
With `p = 1` it is measured from the bottom-right diagonal `j = i + d`, which is
exactly flash-attention's predicate (FA-1). **Sentinels are not shifted by
`p`**: they already name their diagonal.

Every non-sentinel value is a literal bound, negative ones included. This is
the "generalized" in gSWA. **No numeric value means "unbounded"**: an
unbounded side is spelled with a sentinel on the left, or with a large enough
number on either side (see [Matching flash-attention](#matching-flash-attention)).
For example, `(-1, 3)` with `p = 0` keeps `j ∈ [i+1, i+3]`. flash-attention
cannot express that window, because there a negative side means "unbounded".
Numeric windows are expected within `±2^30`, so that `w ± d` cannot overflow
i32. That bound only matters for contexts of about a billion tokens, so no
check enforces it.

Resolution happens in `parse_window` (`modules/flash/kernel/masked_load_store.py`)
for the Triton kernels and in `resolve_window`
(`modules/flash/flyc/fmha_common_gfx1201.py`) for the FlyDSL kernels (gfx950
and gfx1201). Both are called once per sequence, before any interval
arithmetic.

Why sentinels exist: under varlen, `sk - sq` differs per sequence, so no single
host-side number expresses "bottom-right causal". The sentinels and
`principal_diagonal_choice` are the two ways to defer the diagonal to the
kernel. The sentinels cover the two causal masks, and `p` covers every numeric
window.

### A-3. Dense calls

`principal_diagonal_choice` is honored for dense calls too (`d` is just
uniform). Dense callers may instead leave `p = 0` and convert on the host:
`window_left = wl - d`, `window_right = wr + d`. The two are equivalent.

### A-4. Fully-masked rows

**O is 0** on every backend. The LSE of such a row is a placeholder with no
meaning (FA-4) and is not part of the contract. Its sign does not matter.
Current values, for reference:

| Backend | LSE of a fully-masked row |
|---|---|
| FlyDSL (gfx950, gfx1201) | `+inf` for every such row |
| Triton | `+inf` when the whole M-tile is dead. A dead row inside a live tile gets about `-2.36e38` (`-FLT_MAX` init leaking out; a known divergence noted in `fwd_kernel.py`). |

The backward re-masks those elements, so neither value affects gradients.

### A-5. Compatibility

`principal_diagonal_choice` uses a bit that was `reserved`, so the struct
layout is unchanged. The meaning of the varlen word did change, so the params
versions are bumped: `attn_fwd_params::kVersion` 4 → 5 and
`attn_bwd_params::kVersion` 7 → 8.

* A caller built against the new header and run on an older library is refused
  (`params_version` newer than the library), so it can never silently get
  top-left numeric windows.
* A caller built against kVersion 4 / 7 keeps working unchanged: it is
  translated with `principal_diagonal_choice = 0`. Bit 18 is reserved in those
  versions, so a v4/v7 word with it set is still rejected with
  `hipErrorInvalidValue`.

---

## Matching flash-attention

Set `principal_diagonal_choice = BOTTOM_RIGHT` (1) and translate FA's
`(causal, window_size_left, window_size_right)` as below. "Unbounded" is FA's
negative value (or `None` in cute). Apply FA's `causal → wr = 0` first. **Never
pass an FA window size through unchanged when it is negative.** AOTriton reads
it as a literal bound, so `-1` would shift the edge one key past the diagonal
instead of removing it.

| flash-attention | `causal_type` | `window_left` | `window_right` |
|---|---|---|---|
| `(-1, -1)`, not causal | `None` | — | — |
| causal, or `(-1, 0)` | `WindowedAttention` | `BottomRightAligned` | `BottomRightAligned` |
| `(wl, wr)`, both `>= 0` | `WindowedAttention` | `wl` | `wr` |
| `(-1, wr)`, `wr > 0` | `WindowedAttention` | `BottomRightAligned` | `wr` |
| `(wl, -1)`, `wl >= 0` | `WindowedAttention` | `wl` | `max_seqlen_q` |

Why these are exact, for every sequence in the batch:

* A left `BottomRightAligned` resolves to `L = sq`, so `j >= i - sq`, which no
  `j >= 0` violates: unbounded.
* A right `window_right = W` with `p = 1` gives `R = W + d`, which admits every
  column iff `W >= sq - 1`. `max_seqlen_q` satisfies that for every sequence.
  (Symmetrically, `window_left = max_seqlen_k` is an unbounded left side, if a
  number is preferred over the sentinel.)
* For the same reason, FA's `>= seqlen_k` clamps on the left are no-ops on the
  mask and need no counterpart.

**Without `principal_diagonal_choice`** (dense only, or a library predating it):
use `p = 0` and pass `window_left = wl - (sk - sq)`,
`window_right = wr + (sk - sq)` for the numeric sides, keeping the sentinel
rows of the table unchanged. This cannot express bottom-right numeric windows
under varlen, because `sk - sq` differs per sequence. That was the gap reported
in pytorch/pytorch#201930.

### Remaining differences

* **FA2 right-side clamp** (FA-3.1). The table reproduces the exact
  predicate, which FA3, cute and Triton-AMD follow. FA2 and CK differ from it
  when `sq > sk + 1` and the right side is unbounded or `>= sk`. That is an
  FA2 defect, and it is deliberately not reproduced.
* **LSE of rows that see no key.** O matches (0). The LSE is a meaningless
  placeholder on both sides (FA-4, A-4), so differing values are not a
  difference in behavior.
* **FA's `seqlen_q == 1` causal shortcut** needs no counterpart: bottom-right
  causal with one query already keeps every key.

---

## Implementing PyTorch's flash attention ops with AOTriton

PyTorch's CUDA build implements its flash attention ops with a vendored copy of
FA2 (`aten/src/ATen/native/transformers/cuda/flash_attn/flash_api.cpp`). A ROCm
build that implements them with AOTriton should reproduce what the CUDA build
returns. The ops carry a leading underscore, but they are registered aten ops
(`aten/src/ATen/native/native_functions.yaml`), callable from Python as
`torch.ops.aten.<name>`, and public PyTorch APIs are built on them.

This section was written against PyTorch `2fbbabe852d`. File paths are relative
to `aten/src/ATen/native/transformers/` unless they start with `torch/` or
`test/`.

### P-1. The ops, and who calls them

| Op | Window? | Public API built on it |
|---|---|---|
| `aten._scaled_dot_product_flash_attention` (+ `_backward`) | no, only `is_causal` | `torch.nn.attention.bias.causal_lower_right` (`torch/nn/attention/bias.py:248-256`) |
| `aten._flash_attention_forward` / `_flash_attention_backward` | `window_size_left`, `window_size_right` | `torch.nn.attention.varlen.varlen_attn` (`torch/nn/attention/varlen.py`) |

Both are also called directly by external code, `_flash_attention_forward/backward`
most commonly. The CUDA kernel behind the first op is
`_scaled_dot_product_flash_attention_cuda` (`cuda/attention.cu:776`). It is a
thin wrapper that transposes to BSHD and calls `_flash_attention_forward` with
no window, so everything below follows from `_flash_attention_forward`.

**Why the result differs from `F.scaled_dot_product_attention`.** The public
`F.scaled_dot_product_attention(..., is_causal=True)` defines causal as
**top-left** (`torch.tril`). FA2 aligns it **bottom-right**. So PyTorch's
backend selector refuses the flash backend when `is_causal` is set and
`seqlen_q != seqlen_k` (`check_flash_causal_non_square_seqlens`,
`cuda/sdp_utils.cpp:497-513`). That check is in the selector, not in the ops.
Calling an op directly bypasses it, and gets FA2's bottom-right mask. `CausalBias`
relies on exactly that: its lower-right variant calls
`_scaled_dot_product_flash_attention(..., is_causal=True)` with
`seqlen_q != seqlen_k`, with the comment "for this particular op it means lower
right".

### P-2. What `_flash_attention_forward` computes on CUDA

```
_flash_attention_forward(q, k, v, cum_seq_q, cum_seq_k, max_q, max_k,
                         dropout_p, is_causal, return_debug_mask, *, scale,
                         window_size_left=None, window_size_right=None,
                         seqused_k=None, alibi_slopes=None)
  # cuda/attention.cu:1162.  None -> -1 (attention.cu:1201-1202)
  -> mha_varlen_fwd(...)   if cum_seq_q is given (packed THD)
  -> mha_fwd(...)          otherwise (dense BSHD)
```

Both FA2 entry points first normalize `(is_causal, wl, wr)` as in FA-3 (FA2),
then apply the mask of FA-1 with `d = sk - sq` of each sequence. `max_q`/`max_k`
enter only the normalization (FA-3.1).

Outputs:

* **Output**: rows that see no key are all zeros.
* **`softmax_logsumexp`**: fp32, natural log. `(B, H, sq)` dense; `(H, total_q)`
  varlen (`flash_api.cpp:502, 722`), which is AOTriton's
  `VarlenLseLayout::HT`. A row that sees no key gets `+inf`, a placeholder
  (FA-4). When
  `seqlen_k == 0` (dense) or `max_seqlen_k == 0` (varlen) no kernel runs: the
  output is zeroed and the whole LSE is filled with `+inf`
  (`flash_api.cpp:555-562, 795-803`).
* `_flash_attention_backward` (`cuda/attention_backward.cu`, reaching FA2
  `mha_bwd` / `mha_varlen_bwd`) applies `is_causal → wr = 0` and the same
  clamps, and uses the same mask. Rows that saw no key get zero gradient.

`_scaled_dot_product_flash_attention` adds nothing to this: it passes
`max_q = q.size(2)`, `max_k = k.size(2)`, no `cum_seq_*`, no window, and returns
empty `cum_seq_q`/`cum_seq_k` (`cuda/attention.cu:790-827`). Its mask is
therefore

```python
def sdpa_flash_cuda_keeps(i, j, sq, sk, is_causal):
    if not (0 <= j < sk):
        return False
    return (not is_causal) or j <= i + (sk - sq)      # bottom-right causal
```

### P-3. The AOTriton call

Translate to the exact predicate (FA-1), which is what FA3, cute and
Triton-AMD compute. No clamping step is needed: an unbounded left side becomes
the `BottomRightAligned` sentinel, and an unbounded right side becomes
`max_seqlen_q`. Both are exact for every sequence (see
[Matching flash-attention](#matching-flash-attention)).

```python
def fa_window_to_aotriton(is_causal, wl, wr, max_seqlen_q):
    """PyTorch flash window -> AOTriton params.

    wl/wr: None or negative = unbounded (PyTorch's convention).
    max_seqlen_q: q.size(seq) for dense, max_q for varlen.
    Returns (causal_type, window_left, window_right, principal_diagonal_choice).
    """
    wl = None if wl is None or wl < 0 else wl
    wr = None if wr is None or wr < 0 else wr
    if is_causal:
        wr = 0
    if wl is None and wr is None:              # full attention
        return CausalType.None, 0, 0, VarlenPrincipalDiagonal.TOP_LEFT
    if wl is None and wr == 0:                 # causal: sentinels, bit not needed
        return (CausalType.WindowedAttention,
                WindowValue.BottomRightAligned, WindowValue.BottomRightAligned,
                VarlenPrincipalDiagonal.TOP_LEFT)
    left = WindowValue.BottomRightAligned if wl is None else wl
    right = max_seqlen_q if wr is None else wr
    return (CausalType.WindowedAttention, left, right,
            VarlenPrincipalDiagonal.BOTTOM_RIGHT)
```

The same translation goes into `attn_fwd_params` and `attn_bwd_params`. The
varlen word is `{THD, CUMULATIVE, REUSE}` on both sides with `lse_layout = HT`,
plus the returned `principal_diagonal_choice`. Dense is `{}` plus the bit.

**Relation to the CUDA build.** The result is identical to PyTorch CUDA's except
where FA2's right-side substitute is wrong (FA-3.1): `sq > sk + 1` for some
sequence, together with an unbounded right side or `wr >= sk`. There AOTriton
gives the exact mask and FA2 does not. This is deliberate: FA2's behavior there
is a defect, FA3 and the other backends agree with the exact mask, and none of
the ops' callers can depend on it intentionally. Plain causal and full
attention, which is everything `_scaled_dot_product_flash_attention` and
`causal_lower_right` can ask for, are identical.

**`seqlen_k == 0`.** FA2 launches no kernel; it zeroes the output and fills the
LSE with a placeholder (`flash_api.cpp:555-562, 795-803`). Do the same on the
host rather than launching AOTriton with an empty key sequence.

The LSE of rows that see no key needs no fix-up. It is a placeholder on both
sides (FA-4, A-4).

Notes:

* The numeric case needs `principal_diagonal_choice = BOTTOM_RIGHT`. A dense
  call can instead keep it 0 and pass `wl - (sk - sq)`, `wr + (sk - sq)`. A
  varlen call has no alternative, because `sk - sq` differs per sequence.
* FA2's `seqlen_q == 1` causal shortcut is left out on purpose: it does not
  change the mask.
* Never pass a negative window through unchanged. AOTriton reads it as a
  literal bound (A-2).
* Dropout: AOTriton's Philox stream is not FA2's, so with `dropout_p > 0` the
  results agree in distribution but not bit for bit. This is outside the scope
  of this document.

### P-4. What PyTorch's ROCm build does today

`_flash_attention_forward` on ROCm first maps every negative window to "no
value" (`parse_window_size`, `hip/aotriton_adapter.h:186-199`). Then
`calculate_swa()` (`hip/flash_attn/aot/mha_all_aot.hip:83-103`) builds the
AOTriton window:

```c++
int window_left  = max_seqlen_q;            // defaults when a side has no value
int window_right = max_seqlen_k;
if (is_causal) {
  window_left  = WindowValue::BottomRightAligned;
  window_right = WindowValue::BottomRightAligned;
}
if (window_size_left.has_value() || window_size_right.has_value()) {
  window_left  = window_size_left.value_or(window_left);    // numeric, top-left
  window_right = window_size_right.value_or(window_right);
}
```

* **No window, `is_causal` only** (`_scaled_dot_product_flash_attention`,
  `causal_lower_right`, `varlen_attn` with the default `(-1, -1)`): both
  sentinels, bottom-right. This already matches CUDA.
* **Any finite side**: that side is passed as a number, which AOTriton
  measures from the **top-left** diagonal. This includes `varlen_attn`'s causal
  spelling `(-1, 0)`, where the right side becomes a numeric `0`. When
  `seqlen_q != seqlen_k` the mask is wrong. That is pytorch/pytorch#201930. The
  defaults `max_seqlen_q` / `max_seqlen_k` for a missing side are also
  top-left stand-ins, so a one-sided window is shifted as well.

`fa_window_to_aotriton` above replaces `calculate_swa()`. It also subsumes the
special case for `(-1, 0)` that the PR adds.

### P-5. `torch.nn.attention.varlen.varlen_attn` (public)

```python
varlen_attn(query, key, value, cu_seq_q, cu_seq_k, max_q, max_k, *,
            return_aux=None, scale=None, window_size=(-1, -1))
```

The function wraps the custom ops `torch_attn::_varlen_attn` /
`_varlen_attn_backward`, which call `aten._flash_attention_forward` /
`_flash_attention_backward` (`torch/nn/attention/varlen.py`). What it adds on
top of P-2:

* `query`/`key`/`value` are packed `(T, H, D)`. `cu_seq_q`/`cu_seq_k` are
  cumulative positions, `(N+1,)`. Lengths are per sequence, so `seqlen_q[z]`
  may differ from `seqlen_k[z]`, e.g. for decode or chunked prefill against a
  KV cache.
* `is_causal = (window_size == (-1, 0))`, and `window_size` is passed through
  as well. On CUDA this is redundant, because FA2 re-derives causal from the
  window. A different negative spelling such as `(-2, 0)` gives
  `is_causal=False` but still runs as causal.
* The docstring describes `(-1, -1)` as full attention, `(-1, 0)` as causal and
  `(W, 0)` as "causal attention with sliding window of size W". Because the
  CUDA implementation is FA2, all of these are **bottom-right per sequence**.
  The docstring does not say so.
* Dropout is fixed at 0, and the returned `rng_state` is zeros.
* `return_aux=AuxRequest(lse=True)` returns the LSE. The fake implementation
  (`_varlen_attn_fake`) and FA2 give it shape `(H, T_q)`; the docstring says
  `(T_q, H)`.
* The backward passes the same `is_causal` and `window_size`.

Test coverage (`test/test_varlen_attention.py`): `test_varlen_vs_sdpa`
compares against a padded `F.scaled_dot_product_attention` with a hand-built
mask, for windows `(-1, -1), (-1, 0), (4, 0), (-1, 4), (0, 0), (1, 0), (0, 1),
(0, -1), (4, 4), (8, 2), (1025, 1025)` (lines 350-420). The reference mask is
top-left (`torch.tril`, and `[i - wl, i + wr]`, lines 108-125), and every test
uses the same `cu_seq` for queries and keys, so `seqlen_q == seqlen_k` per
sequence. Top-left and bottom-right coincide there, so **the upstream tests
cannot tell the two alignments apart**. pytorch/pytorch#201930 adds cases with
`seqlen_q != seqlen_k` (decode, batched decode, chunked prefill, `q_len > k_len`)
for `(-1, 0)` only. AOTriton's own tests for `principal_diagonal_choice` should
cover the numeric windows of the list above with `seqlen_q != seqlen_k` as
well.
