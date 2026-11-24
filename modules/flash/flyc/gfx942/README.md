# gfx942 flyc attention: the flex_gemm model

How the gfx942 forward, dQ and dK/dV kernels are built, and what we learned
rewriting nine hand-written bodies onto shared components without losing a bit
or a TFLOP. Written for whoever applies the same model to another arch
(the gfx1201 refactor, gfx90a). The design record is
`flyc-gfx942-flex-gemm-design.md` (section R2) in the repo root, if present.

## The idea

FlexGEMM: a fixed, fast GEMM with hooks where feature code plugs in. In
attention the GEMMs are interleaved with softmax and memory traffic across
tiles, so what is shared is not a GEMM loop but **GEMM phases**; each body keeps
a short loop (its schedule) that calls them and the feature hooks.

| Module | Holds | Called as |
|---|---|---|
| `flex_gemm.py` | one `gemm` template: operand sources, phases, `GemmTraits`, the `Ctx`, loop runners, `ClusterOps` | `gemm_*` phases by name; the rest `flex_gemm.X` |
| `softmax.py` | feature hooks: forward `head`/`tail`/`rescale`/`epilogue`, backward `p_from_lse`/`ds_packs`/`p_ds_kv`/`p_ds_elems`, LSE/Delta rows | `softmax.X` |
| `tensor_io.py` | accumulator stores through the GEMM's lane map (O, dQ, dK, dV) | `tensor_io.X` |
| `utils.py` | small device helpers (MFMA wrappers, P rounding, masks, Philox planes, barriers) | by name |
| `host.py` | host-side launch glue: causal types, argument checks, the `_launch` callables in gfx1201's ABI (gfx942-only for now; to be shared once every arch settles) | `host.X` |
| `fwd*.py`, `dq*.py`, `dkdv*.py` | one body each: traits, kernel signature, prelude, the tile loop, epilogue | entry builders |

### One GEMM template, two presets

All nine GEMMs are acc += A·B with B a register operand passed in (resident Q,
K or V; or P / dS from the previous GEMM) and the accumulator passed in or
fresh. Only A's source differs, so there is one template and the presets only
build traits:

- `flex_gemm.trait_ab(...)`: contract over the head dim; A streamed row-major
  through LDS. Forward QK; dQ S, dP; dK/dV S, dP (roles of Q/dO and K/V swap).
- `flex_gemm.trait_tb(...)`: contract over the tile; A transposed. Forward PV;
  dQ's dQ; dK/dV's dV, dK. The transpose is the A source: `PermA` (v_perm on
  the way into LDS), `TransA` (built in LDS), `DerivedA` (derived on read from a
  row-major tile).

### Phases, not a loop

The dual-wave forward touches tiles j, j+1 and j+3 in one iteration (K(j+1)
reads in one cluster, QK(j+1) MFMAs in the next, K(j+3) DMA and V^T(j) reads,
then PV(j)). So a GEMM is a set of stateless phases, each taking an explicit
tile offset and a trace-time buffer index: `gemm_dma`, `gemm_load`,
`gemm_stage`, `gemm_read`, `gemm_mma`, `gemm_stream` (read and MMA
software-pipelined), the derived/transposed variants. Phases never wait, barrier,
set priorities or emit `sched_barrier`; they publish their op counts (`n_dma`,
`n_load`, `n_mma`, `n_reads`, `n_stage`) as Python ints so the schedule's
`s_waitcnt` and `sched_group_barrier` counts are derived, not hard-coded.

### The body

```python
def iteration(j, b, masked):                      # one tile; b = j % buffers, a Python int
    kv_next = kv(j + 1)
    mem0(kv_next, b, masked)                      # memory slot: phases + waits + barrier
    compute_begin()
    ctx.o = softmax.rescale(ctx.o, ctx.corr)
    s_next = gemm_mma(qk, ctx.k_reads, ctx.q_packs, a_masks=ctx.k_masks)   # QK(j+1)
    p = softmax.tail(ctx, kv(j))                                           # P(j)
    compute_end(f'C1.{b}')
    mem1(kv_next, b)
    compute_begin()
    ctx.o = gemm_mma(pv, ctx.vts, p, ctx.o)                                # PV(j)
    softmax.head(ctx, s_next, kv_next, masked, ctx.b_next)                 # max(j+1)
    compute_end(f'C3.{b}')

warmup()
ctx.tiles(iteration, j_clean, j_clean_end, n_loop)
results = drain(n_loop)
```

Memory work and waits live in named slots (`mem0`/`mem1`, or `x_end`/`y_end`);
per-cluster op counts are `flex_gemm.ClusterOps` set once at builder scope and
applied with `flex_gemm.spread`.

### Traits and `Ctx`

- **Traits** (`GemmTraits` and each body's traits) are frozen dataclasses of
  Python values, derived once at builder scope. Run-time values never go in
  traits; they are arguments and return values.
- **`Ctx`** holds the traits plus the run-time state. `ctx.carried` names what
  crosses the tile loop's back edge; `ctx.carry()` / `ctx.restore()` move it
  through FlyDSL's loop iter args. Values set before the loop (`set_q_rows`,
  `set_o_store`) need no carry. Store raw values only; derive (`fx.Index`
  wraps, pitches, offsets) inside the helper where the original code derived
  them.
- **Cache tag.** FlyDSL's JIT cache does not hash objects or module-prefixed
  calls, so every kernel references
  `flex_gemm.cache_tag(traits, flex_gemm, softmax, tensor_io, utils)`: all
  traits fields plus a digest of those modules' source (`utils` too: helpers
  reached through a module-prefixed call are not walked).

## Conventions

- Only `gemm_*` device phases are imported by name (the name reads as running a
  GEMM); everything else is module-prefixed.
- Opaque arguments (callables such as `lambda: lane_field(4, 2)`, predicates)
  are keyword-only (`*`) in helper signatures and passed by name.
- A wrapped call puts one argument per line, aligned under the first.
- Each block inside a kernel gets a 1-3 line comment: what it does, and why
  where it is not obvious (which tile a slot serves, why an op order is kept).
- What depends on one body's lane layout stays in the body with a 1-2 line
  comment saying why: dB's lane exchange (ds_swizzle in the narrow dQ, a 64-lane
  ds_bpermute deal in the wide one), the dK/dV team exchanges, xxwide's private
  staging.

## Numerics: named options, measured defaults

Where bodies computed the same thing differently, the difference became a named
option, never a silent unification; defaults are measured (faster at seqlen >= 8k),
and every value is held to the fp64 reference.

| Option | Values | Default |
|---|---|---|
| P's exponent (dQ) | `P_EXP_TWO_ROUNDINGS` (forward's P bit for bit), `P_EXP_FMA` | narrow two roundings, wide fma |
| `mask_at` (dK/dV knob) | `'score'` (-inf before exp2, as the Triton kernels), `'prob'` (P = 0 after) | narrow score, wide prob, xwide prob, xxwide score |
| `dropout_scale_at` (dK/dV knob) | `'elem'` (1/(1-p) on P before the bf16 pack), `'store'` (once on fp32 dV) | narrow, wide, xwide store; xxwide elem |

Both mask points zero P with a select, which discards the unselected operand
(inf/NaN included), so `'prob'` is as safe as dropout's keep select. The same
math still rounds differently: `bf16(P*s) != s*bf16(P)`, and moving a select
changes whether LLVM fuses f16 conversions into `v_fma_mix`. Check against the
Triton kernels (`modules/flash/kernel/*_inner*.py`) before calling a difference
a design choice; ours were accidents.

## How the rewrite was done

1. **Port body by body next to the untouched original** (`x_flex.py` beside
   `x.py`), same builder signature, kernel name and ABI.
2. **A/B bar per body:** bitwise outputs vs the original over the feature
   matrix, ISA compared per feature build, TFLOPS interleaved old/new, existing
   suites green. After every shared-module change, re-check the ISA of every
   body already ported.
3. **Switch over** by `git mv x_flex.py x.py`; convert the A/B tests into fp64
   reference tests.

Byte-identical ISA was reachable everywhere except where a numeric option
deliberately moved; it is the cheapest proof a refactor changed nothing.

## Lessons

**Keeping the ISA identical**
- A shared helper must emit ops in the caller's original order. Seen: LSE's
  `log2` computed after its buffer resource instead of before; a store offset
  computed after its mask; operand shuffles built before the first MFMA instead
  of interleaved (bf16 hides it behind a bitcast, f16 does not). Each moved
  instructions.
- Never hoist a derived value into the prelude "because it is constant": LLVM
  then sees it earlier and schedules differently.
- The toolchain flips some builds (mostly dropout) between two
  register-allocation forms from one compile to the next. Before calling a diff
  real, rebuild the baseline (3x); report flips.

**FlyDSL mechanics**
- A `@flyc.jit` called while a kernel traces is inlined: it may take Python
  callables, traits, `None`. Hooks passed as arguments compile to the same ISA as
  hand-inlined code.
- A plain `if` in a kernel/jit body becomes a run-time branch; use `const_expr`
  for trace-time ones. A run-time branch splits LLVM's scheduling region: keep
  hooks branch-free unless the branch is the point (the ballot-gated lazy max).
- Carried loops (`range(..., init=)` + `yield`) need the body as a function
  traced once per (buffer, masked) variant: hence `ctx.tiles(iteration, ...)`.
- Each branch of a run-time `if` that touches `ctx` starts with
  `ctx.restore()`, or one branch's region-local values leak into the other
  ("operand does not dominate").
- The JIT cache keys on same-directory helpers imported by name, closures and
  jit functions; not objects, classes or `module.func` calls. Hence the module
  digest in the cache tag. AOT builds and `conftest.py` run with the cache off.

**Register pressure**
- The narrow forward sits at 256 VGPRs. A masked tile whose buffer is known
  only at run time (`_masked_step`) spilled resident Q when the lane's band and
  key stayed live for it; rebuild such values at use behind asm LLVM cannot
  hoist (`utils.lane_now`). Check `.vgpr_spill_count` and look for `scratch_`
  in the loops, not just the total.
- A skip branch in such a tile (`softmax.rescale`) kept a second copy of O for
  the skip path; `softmax.rescale_always` (same bits) removed the last spills.

**Causal grids**
- Run the longest q tiles first (LPT). On a head-fastest grid that alone took
  narrow causal hd128 from 381 to 457 TFLOPS at B1 H32 S32k.
- Measured and not adopted for the narrow forward: pairing q tiles g and T-1-g
  per block as the wide body does (spills; slower at S8k), and renumbering
  blocks so each XCD's are consecutive (slower in every order tried).

**Testing**
- Fill every output with NaN before each run and assert none is left outside
  padded slack columns: every element a kernel owns must be written, fully
  masked rows included (O = 0, LSE = +inf, zero gradients).
- Exercise VarlenBits bit 18 (`principal_diagonal_choice`) with numeric windows
  over sq <, == and > sk, dense and mixed varlen, decode and chunked prefill;
  run key-less rows with both +inf and a finite LSE so the mask alone must
  zero them (`tests/test_bwd_dq_principal_diagonal.py` is the model).
- Hold every value of every numeric option to fp64, not just the default.

**Measuring**
- The GPU is power-capped (750 W): interleave old and new, take medians over
  10-30 repetitions. Concurrent runs on several GPUs produced ~50% outliers;
  rerun suspicious cases one at a time.

## Applying this to another arch

Shared by design: the model (phases, presets, `Ctx`, hooks, named options), the
conventions, the process and the tests. Arch-specific, to be redone per arch:

- MMA shape and lane maps (`GemmTraits.mfma`, `coords`, `group_coords`;
  `softmax.EXCHANGE`/`REDUCE` give the row exchange and reduction order per
  shape). gfx1201 is wave32 WMMA, so its lane maps and row exchanges differ.
- Operand sources: which transposes exist (LDS transpose reads, v_perm, derived
  reads), LDS size (64 KB on gfx942) and therefore buffering, whether LDS DMA
  (`buffer_load ... lds`) is available.
- P rounding: gfx942 has no f32->bf16 instruction, so `pack_operand` rounds in
  software (round to nearest, ties away; never truncate).

Start from one high-value body (we used the hd128 forward), prove bitwise and
TFLOPS parity, then port the rest.

## Future work

FlyDSL's layout API (`make_tiled_mma`, `make_fragment_A/B/C`,
`partition_C(make_view(0, layout))` for per-lane coordinates, `fx.gemm`,
copy atoms) is current upstream practice for GEMMs and would replace the
hand-written lane maps. Register fragments are promoted to SSA vectors by
FlyDSL's own passes. Not adopted here: it is a major rewrite whose benefit is
readability, and our schedule controls (`pin_values`, alias scopes,
`sched_group_barrier`, carried loop values) work on SSA values.
