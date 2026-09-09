# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI description of the flash attn_fwd FlyDSL backend (gfx1201 and gfx950).

DEMO / DESIGN SKETCH -- not wired into flash_entry.py yet. What is here is the
half the ahead-of-time compile driver reads: the marker, the hints dataclass,
the declared kernel-argument list, and the builder. Registering this as a
backend of `op_attn_fwd`, and the citation, disable and context-helper wiring
the code generator reads, come later. `python -m aotriton.flyc_compile` on this
file already produces an hsaco for either architecture, with no GPU; see
`docs/FlyDSL.md`.

A FlyDSL backend is a THIRD kind, and it takes one half from each of the two ATI
already has:

    triton (@ati.source)   compiled during the build, ATI owns the perf space,
                           1:1 argument names, functional axes ARE kernel params
    aiter  (@ati.affine.*) prebuilt .co, no perf space, no axes of its own —
                           inherits the operator's and filters them
    flyc   (@ati.flyc.*)   compiled during the build (from triton), inherits and
                           filters the operator's axes (from aiter), and — unique
                           to it — dispatches an hsaco whose kernarg list is NOT
                           the operator's

Three consequences, each handled below:

1. **The tuning table is programmatic.** `fmha_tuning_gfx1201.resolve_knobs()` is
   the sole producer of a schedule, so there is no `@ati.tune.schema`, no
   `@ati.tune.configs`, and no perf axes — ATI enumerates no perf variants here.

   That is NOT the same as one hsaco per functional. Today the count is one, but
   only because the shipped schedule targets long sequences and short ones fall to
   the Triton backend; a seqlen-dependent FlyDSL tuner will emit several. Nothing
   downstream may assume the count — hsacos are packed into per-functional .aks2
   archives exactly as Triton's are, which is also what keeps Triton's autotune
   code generator reusable for flyc later.

2. **No functional axes are declared here as kernel ARGUMENTS.** They belong to
   the operator (owned by the default triton backend); this backend inherits
   them and narrows. `flyc_attn_fwd` then reads `choices.NAME` (a `ChoiceView`)
   — the OPERATOR's choices, parsed from `--signature` text by the driver — and
   maps them to builder knobs. This is also why the axes cannot be re-declared
   as a PLAIN `@ati.scalar`: they are not arguments of the flyc kernel at all,
   they are build-time Python values.

   `BLOCK_DMODEL`/`PADDED_HEAD` ARE declared below, near the bottom of the
   stack — but as MARKERS (`options=` and no type), a different, narrower shape
   from every other `@ati.scalar` in this file. A marker never becomes a kernel
   argument, since `options=` and an explicit type are mutually exclusive, so it
   can never claim a kernel-argument slot the way `varlen_bits`'s `'i32'` does.
   It exists so that dispatch can be redirected to the rounded value this
   backend actually compiled, rather than the raw choice: this backend's ladder
   is a strict subset of the declared axis.

3. **The kernarg list is declared, because nothing else can supply it.** aiter hands
   the params struct to a C++ cookie and never names a kernarg; triton gets its list
   from the kernel signature. flyc dispatches the hsaco directly, and the hsaco's
   layout comes from `flash_attn_func_aiw_kernel`, so the wiring below is the
   description's real payload.
"""

from dataclasses import dataclass, asdict

import aotriton.template_instantiation as ati


@dataclass
class FlycFwdHints:
    """Tuning inputs the builder may read that are NOT functional axes.

    Registered with `@ati.flyc.hints` — namespaced with the rest of the FlyDSL-specific
    surface rather than added to `ati.tune.*`, which is the SHARED tuning vocabulary
    (schema / configs / binning / fallback all feed the LUT and the tuning DB; this
    feeds one builder and nothing else). `ati.affine.*` is the same pattern for aiter.

    FlyDSL would call these part of the **problem**, opposite its **schedule**
    (`fmha_tuning_gfx1201.py:410`: *"a caller states a problem, the tuning policy
    answers with a schedule"*) — worth knowing if you arrive from that file. Strictly
    the RUNTIME half of it: dtype / head_dim / causal are part of FlyDSL's problem too,
    but in ATI they are functional axes and arrive on `f`. `f` and `hints` together are
    what `FmhaInputMetadata` keeps in one dataclass.

    The defaults must reproduce the schedule the kernel ships today: `resolve_knobs`
    reads none of these, so every build is hint-independent until FlyDSL's tuner
    grows a seqlen dependence, and a caller with nothing better to say passes a
    default-constructed instance.
    """
    seqlen_q: int = 0        # 0 = unknown/any; a real value once the tuner uses it
    seqlen_k: int = 0
    num_heads: int = 0
    batch: int = 0


# The compiled tile ladder, from fmha_tuning_gfx1201._BLOCK_DMODEL_LADDER. A literal,
# not an import: the generator parses this module and has no flydsl. `flyc_build`
# (build time) does import flydsl, and `resolve_knobs` re-validates the tile, so a
# drift here fails loudly at build rather than silently emitting the wrong kernel.
FLYC_HEAD_DIMS = frozenset({16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 384, 512})

# gfx950's ladder, from fmha_tuning_gfx950.LADDER -- a literal for the same reason
# as FLYC_HEAD_DIMS above. Narrower than gfx1201's: no 16/48/80 (PV_MFMA_N is 32,
# so only multiples of 32 are compiled). 96 IS
# included, matching upstream's LADDER verbatim: a comment right above that tuple
# in fmha_tuning_gfx950.py claims 96 "computes the wrong answer", but that is
# stale prose left behind when the bug it described was fixed the same day
# (upstream `98493cc7`, an ancestor of the vendored commit; it added 96 to the
# tuple in the same change). Do not re-derive this by excluding 96 "to be safe":
# that would silently give up a rung upstream measured at 932 TF against 579 for
# the 128-tile fallback, on the strength of a comment its own repository had
# already superseded.
FLYC_GFX950_HEAD_DIMS = frozenset({32, 64, 96, 128, 160, 192, 224, 256, 384, 512})

# Per-arch ladder lookup, keyed the same way the builder branch below is. It is
# also the statement of which architectures this backend serves at all, and the
# table the backend's disable predicate keys on.
_FLYC_FWD_LADDERS = {
    'gfx1201': FLYC_HEAD_DIMS,
    'gfx950': FLYC_GFX950_HEAD_DIMS,
}


@ati.start
#
# --- the kernarg ABI, in `flash_attn_func_aiw_kernel` order ------------------
#
# The kernel, NOT the Python launcher: the launcher is host code that the C++ shim
# replaces (it computes the grid and marshals arguments), while the @flyc.kernel
# def is what fixes the kernarg layout. The two lists differ — the launcher carries
# a `stream` and a `batch_size` that the kernel does not have. `batch_size` left
# the kernarg in FlyDSL 67a3ace0 and survives only on the host side, where the
# launcher folds it into the grid; AOTriton computes its own grid, so nothing
# here declares it. `FlycAttnFwdContext::grid_calculator()` still needs a batch
# count and still derives one — see modules/flash/csrc/flyc_attn_fwd.cc.
#
# The order is frozen and load-bearing: fmha_common_gfx1201.py:207-209 records that
# switching these pointers to fx.Tensor would grow the kernarg segment from 268 to
# 428 bytes and shift the offset of every argument after the first.
#
# TODO(fx.Tensor): these declarations are also what the AOT compiler will read to
# synthesise operand descriptors with the right dtype and rank, which is what would
# let a future launcher take `fx.Tensor` instead of `fx.Pointer`. Not supported yet
# and deliberately so -- the tensor kernarg ABI is not pinned down (each fx.Tensor
# adds a 40-byte by-value memref descriptor interleaved after its pointer, and what
# those bytes contain is unverified. Raw pointers plus explicit strides stay the
# position while FlyDSL's own tensor ABI settles.
#
# `rank=4` with only THREE strides is the declaration that the last dimension is
# unit-stride and has no argument. FlyDSL requires stride(3) == 1 and reads no
# D-axis stride at all, so unlike the triton kernel (where `stride_qk` exists as a
# parameter and `contiguous=-1` names it) there is nothing here to point at.
# Deriving it from the shortfall — trailing `rank - len(strides)` dims are
# implicitly unit, constexpr 1, not passed — needs no new keyword and makes the
# rank annotation carry its own justification. `contiguous=-1` would be actively
# WRONG here: it indexes the matched stride list, which has three entries, so it
# would mark `stride_q_seq` as the unit stride.
@ati.tensor('Q',   'T_io', rank=4, strides='stride_q_*', wires_to='Q')
@ati.tensor('K',   'T_io', rank=4, strides='stride_k_*', wires_to='K')
@ati.tensor('V',   'T_io', rank=4, strides='stride_v_*', wires_to='V')
# B sits between V and O, and its three strides are batch, head and QUERY ROW --
# the bias is (B, H, Sq, Sk), so the axis the KV tile walks is the contiguous
# one and has no argument, exactly as D does for Q/K/V/O above.
@ati.tensor('B',   'T_io', rank=4, strides='stride_b_*', wires_to='B')
@ati.tensor('O',   'T_io', rank=4, strides='stride_o_*', wires_to='Out')
# LSE is always compact: the kernel derives both pitches from LSE_LAYOUT,
# num_head_q and the token count, so it has no stride arguments by design.
@ati.tensor('LSE', '*fp32:16', rank=2, wires_to='L')
# varlen seqinfo. These were once a rename: FlyDSL splits each side into a
# (length source, position source) pair, where AOTriton spelled the same two
# tensors cu_seqlens_?/seq_strides_?. Upstream's varlen_bits port adopted the
# role-based naming, so the wiring is now identity -- kept explicit to match the
# operand wires above.
@ati.tensor('seqinfo_q0', '*i32:16', rank=1, wires_to='seqinfo_q0')
@ati.tensor('seqinfo_q1', '*i32:16', rank=1, wires_to='seqinfo_q1')
@ati.tensor('seqinfo_k0', '*i32:16', rank=1, wires_to='seqinfo_k0')
@ati.tensor('seqinfo_k1', '*i32:16', rank=1, wires_to='seqinfo_k1')
#
# --- gfx950-only, folded to constexpr -----------------------------------------
#
# `Workspace`, `BlockTable` and `block_table_stride` are real, named parameters
# of `flash_attn_func_gfx950_kernel` -- unlike BLOCK_DMODEL/PADDED_HEAD below,
# they DO appear in that kernel's AST-parsed parameter order, at
# their own real positions (Workspace/BlockTable sit right after LSE;
# block_table_stride is the very last parameter, after every stride). They do
# NOT exist at all in gfx1201's `flash_attn_func_aiw_kernel` signature, so on
# gfx1201 `real_param_order` never contains these three names and this
# declaration is simply never consulted -- one line serves both arches.
#
# `options=[0]` is what makes them constexpr: our builds pin
# `paged=False, num_kv_splits=1` everywhere on gfx950 (asserted below), which is
# the configuration under which FlyDSL's own `_WS_ANN`/`_BT_ANN`/`_BTS_ANN`
# annotations collapse these three to Constexpr in the compiled ABI --
# `kdesc.pp_arg_doc` reads this declaration to fold their pp_args entries
# rather than re-deriving that fact from FlyDSL's own annotations (see that
# method's docstring for why the two are deliberately independent).
#
# One combined declaration, not three, and its position in this decorator
# stack does not matter: `iter_launch_arguments` walks the REAL kernel's
# parameter order (arch-specific) and looks each name up in `self.scalars` by
# name, not by where in this file the name was declared -- so the two
# non-adjacent real positions (7-8 and the very last) are found correctly
# regardless.
@ati.scalar(['Workspace', 'BlockTable', 'block_table_stride'], options=[0])
#
# --- the layout word, and the count derived from it ---------------------------
#
# `varlen_bits` IS AOTriton's `Varlen_bits`, unchanged: FlyDSL and Triton share
# one varlen ABI, so the same u32 word describes the layout to both and this is
# a plain rename like the ones below it.
@ati.scalar('varlen_bits', 'i32', wires_to='Varlen_bits')
# `num_seqlens` has no operand to wire to, and that is the whole reason it is
# declared apart from the renames. It is COMPUTED, host side, from the decoded
# `varlen_bits` and the Q tensor's own extents: FlyDSL wants the number of
# sequences packed into a 1THD tensor, and zero when the Q side is not packed
# at all. AOTriton has no field carrying that -- only the layout word it is
# derived from.
#
# The `batch_size` it used to travel with left the kernel argument list
# upstream (FlyDSL 67a3ace0) and now exists only host side, where the grid's z
# extent is `num_seqlens != 0 ? num_seqlens : batch_size`. Both numbers come
# out of one derivation, and the derivation is also where an underivable
# sequence count is rejected -- a stacked side whose token axis is not a whole
# multiple of `max_seqlen`. Getting either wrong fails SILENTLY, in FlyDSL's own
# words: "it launches N programs over a tensor whose batch axis is 1, and every
# one of them addresses a plausible row."
@ati.scalar('num_seqlens', 'i32')
#
# --- plain renames ------------------------------------------------------------
@ati.scalar('max_seqlen_q', 'i32', wires_to='Max_seqlen_q')
@ati.scalar('max_seqlen_k', 'i32', wires_to='Max_seqlen_k')
@ati.scalar('window_left',  'i32', wires_to='Window_left')
@ati.scalar('window_right', 'i32', wires_to='Window_right')
# PRNG: 1:1 with the triton kernel as of FlyDSL 971dce48 ("the philox seed becomes a
# pointer, and the forward reports what it drew") + 53334317 ("the philox offset
# splits into a pointer and an immediate"). These were the two transient mismatches
# in the previous revision of this file; both are gone, and neither needs a helper.
# Declared one per line, in kernel order, NOT grouped. Grouping the four *u64
# pointers into one @ati.tensor([...]) reads better but silently reorders:
# philox_offset2 is a scalar sitting BETWEEN them in the kernel signature, so the
# grouped form pushed it to the end and the declared kernarg order was wrong.
@ati.tensor('philox_seed_ptr', 'T_u64', rank=0)
@ati.tensor('philox_offset1', 'T_u64', rank=0)
@ati.scalar('philox_offset2', 'u64')
@ati.tensor('philox_seed_output', 'T_u64', rank=0)
@ati.tensor('philox_offset_output', 'T_u64', rank=0)
#
# --- the two dropout arguments that are not a rename --------------------------
@ati.scalar('idropout_p',    'i32')
@ati.scalar('dropout_scale', 'fp32')
#
# --- plain renames, continued -------------------------------------------------
@ati.scalar('num_head_q', 'i32',  wires_to='Num_head_q')
@ati.scalar('num_head_k', 'i32',  wires_to='Num_head_k')
@ati.scalar('hdim_qk',    'i32',  wires_to='Hdim_qk')
@ati.scalar('hdim_vo',    'i32',  wires_to='Hdim_vo')
# Last scalar before the stride block, where the backward kernels also put it
# (FlyDSL 1b58fb93 made one kernarg convention across all four).
@ati.scalar('sm_scale',   'fp32', wires_to='Sm_scale')
#
# --- functional-axis markers, NOT kernel arguments ----------------------------
#
# `BLOCK_DMODEL`/`PADDED_HEAD` are the operator's own axes, inherited by this
# backend and narrowed by it -- NOT part of `flash_attn_func_aiw_kernel`'s
# argument list declared above, so these two lines are deliberately kept out of
# that ordered block. `options=` and an explicit type are mutually exclusive, so
# a marker can never be read as a real launch argument; it is only ever found by
# axis name.
#
# Why either axis needs a marker at all: dispatch bins the caller's head
# dimension to a `BLOCK_DMODEL` rung on the OPERATOR's ladder before a backend
# is chosen, and this backend's compiled ladder is a strict subset of it, so
# the digit has to be redirected to the rung this backend actually built.
# `PADDED_HEAD` must follow that decision -- a kernel re-rounded to a wider
# rung with `PADDED_HEAD` left false is a silent wrong answer, not a build
# error.
#
# The `options=` lists restate each ladder as free documentation, not as a
# second source of truth: the digit's real range comes from the OPERATOR's
# axis, and the backend's own disable predicate is what enforces the subset.
# One shared marker declaration serves both architectures (like the
# Workspace/BlockTable declaration above, this decorator stack does not vary by
# arch), so `options=` is the UNION of both ladders rather than either alone --
# neither is a subset of the other, so one arch's list would under-document the
# other's compiled rungs.
@ati.scalar('BLOCK_DMODEL', options=sorted(FLYC_HEAD_DIMS | FLYC_GFX950_HEAD_DIMS))
@ati.scalar('PADDED_HEAD', options=[False, True])
@ati.flyc.hints(FlycFwdHints)
@ati.flyc.kernel()
def flyc_attn_fwd(arch, choices, hints):
    """Build one hsaco for the functional described by `choices`, optimized
    for `hints`.

    Executed by `aotriton.flyc_compile` at build time, in a venv that has flydsl —
    never by the generator, which only reads the decorators above. Returns
    `(built, knobs)`: `built` is whatever the builder returns (the driver drives
    it to a code object), and `knobs` is a JSON-serialisable dict of whatever
    this description wants recorded alongside the hsaco. Here that is
    `asdict(knobs)` — `resolve_knobs` is the only place `block_m` (and everything
    else the driver records alongside the hsaco, bar `block_size`, which it
    recovers itself from the compiled IR's `known_block_size`) is known, and it
    would otherwise go out of scope on return: `built` is the `_launch` closure,
    which exposes only `compile` and the `varlen_*` helpers, not `knobs`.

    TWO objects, because they are two kinds of fact:

      choices  what the kernel must SUPPORT: a `ChoiceView` (`ir/choices.py`)
               over the compile-time identity. Two call sites hand this
               function two different backings, and the function reads neither
               one directly -- only the interface: the generator has a linked
               `ir.Functional` and passes the real thing, `f.choices`
               (`FunctionalChoiceView`); the driver (`flyc_compile.py`) has
               only `--signature` text in a separate process with no linked
               IR, and passes its own `MappingChoiceView` over the parsed dict
               -- a `Functional` cannot be rebuilt from that text, which is
               exactly why the interface exists. `choices.NAME` reads a choice variable
               by attribute; `choices.arg('Q')` reads a real argument name
               that is not one (`T_io` is the variable governing `Q`).
      hints    what the kernel should be OPTIMIZED FOR. Declared by
               `@ati.flyc.hints` above. Not axes, and deliberately so —
               `seqlen_q` is a tune BINNING dimension
               (`@ati.tune.binning(Max_seqlen_q=...)`), and promoting it to an
               axis would multiply the functional space and the godel
               numbering for every backend in order to serve one.

    Deliberate asymmetry with a disable predicate, which takes a real
    `ir.Functional` and reads `f.arch`: disable predicates run GENERATOR-side,
    where the linked IR exists; this function runs DRIVER-side, where only
    `--signature` text exists. `choices` is not a `Functional` and
    must not grow into one -- if a build body ever needs arch, it arrives as an
    explicit FIRST parameter (from `f.arch` / `--target`), not smuggled into
    `choices`.

    Today `resolve_knobs` reads no field of `hints` — FlyDSL's tuner is currently
    seqlen-independent, so the count stays at one per functional and every field sits
    at its default. The parameter exists so that stops being an API change — and the
    packaging is already N-capable, so neither is the artifact layout.

    `resolve_knobs`/`fmha_knobs(...).resolve`, NOT `plan`: `plan()` is the JIT
    entry point, which takes a *real* head_dim and rounds it up the ladder,
    deriving `padded_head` on the way. AOT already knows the tile — it IS
    `BLOCK_DMODEL` — and `PADDED_HEAD` is its own functional axis, so `plan()`
    would silently re-derive an axis the operator has already fixed. The
    builder's own keyword front end draws the same distinction on both arches.

    ONE INVARIANT TO KEEP: `meta` here is a pure function of
    `choices` alone on both arches — every field either comes straight off a
    `choices.NAME` read or is a fixed pin (`num_heads=1`). That must stay true:
    `knobs.build_traits(meta)` is re-run from `meta` at build time, in a
    different process (`aotriton.flyc_compile`) than the one that computed
    `asdict(knobs)` for the psel here, and nothing downstream would notice the
    two falling out of step — the psel is what the C++ side reads for the grid.
    """
    if arch == 'gfx1201':
        # ONLY the flydsl-free tuning module at call time. The FlyDSL-bearing
        # import lives inside `build()` below, so the code generator -- which
        # calls this function but never the callable -- never imports flydsl.
        from fmha_tuning_gfx1201 import FmhaInputMetadata, FmhaKnobs, resolve_knobs

        tile = choices.BLOCK_DMODEL
        meta = FmhaInputMetadata(
            # `num_heads` reaches the emitted kernel ONLY through STRIDE_TOKEN,
            # which is read exclusively under STRIDES_CONSTEXPR — a dense-only
            # diagnostic arm the AOT path never selects (see the knob below).
            # Pinning it to 1 keeps it out of the functional space. Asserted,
            # not assumed: this is a property of today's kernel, not a
            # contract it owes us.
            num_heads=1,
            head_dim=tile,
            # FlyDSL's causal_type IS AOTriton's CAUSAL_TYPE (0 none / 1 top-left /
            # 2 bottom-right / 3 window), and the kernel only ever emits {0, 3} — the
            # same pair the operator's CAUSAL_TYPE axis offers. 1:1, no mapping.
            causal=choices.CAUSAL_TYPE != 0,
            causal_type=choices.CAUSAL_TYPE,
            dtype_str='bf16' if '*bf16' in choices.arg('Q') else 'f16',
            bias=bool(choices.BIAS_TYPE),
            dropout=bool(choices.ENABLE_DROPOUT),
        )
        # Supply FmhaKnobs to resolve_knobs to make sure knobs.block_dmodel align with choices.BLOCK_DMODEL
        knobs = resolve_knobs(meta, FmhaKnobs(
            block_dmodel=tile,
            padded_head=choices.PADDED_HEAD,
            # AOT cannot bake strides: one binary must serve every layout. This is also
            # what makes `num_heads` above irrelevant to the emitted code.
            strides_constexpr=False,
        ))
        assert not knobs.strides_constexpr, \
            'num_heads=1 is only safe while STRIDE_TOKEN stays behind strides_constexpr'

        def build(knobs=knobs):
            """Deferred: constructs the FlyDSL module. Imports flydsl transitively,
            so ONLY `aotriton.flyc_compile` (run by ninja) may call this.

            `knobs` is bound here as a default rather than captured: the name is
            rebound to the JSON knob dict below, and a live closure would follow
            it and hand a `dict` to a function expecting the knob dataclass. The
            driver calls this with no arguments."""
            from flash_attn_func_gfx1201_aiw import build_flash_attn_func_aiw_module_primary
            return build_flash_attn_func_aiw_module_primary(meta, knobs)

        # Two plain strings: the vendored file, relative to
        # modules/flash/flyc/, and the `@flyc.kernel` def's own name inside it.
        # Read by codegen/flytune.py and flyc_compile.py off the `build`
        # closure WITHOUT ever calling it.
        build.flyc_source = 'flash_attn_func_gfx1201_aiw.py'
        build.flyc_kernel_name = 'flash_attn_func_aiw_kernel'

        # gfx1201's own knob class has no GRID_AXIS_ORDER field: the grid has
        # always walked (head, q_tile, seq) here, i.e. HEAD_FASTEST. Supply the
        # key by hand so both arches' knob dicts carry it uniformly.
        knobs = asdict(knobs)  # past here `knobs` is the dict, not the dataclass
        knobs['GRID_AXIS_ORDER'] = 0  # HEAD_FASTEST; fmha_tuning_gfx950.GRID_AXIS_HEAD_FASTEST
        return build, knobs

    elif arch == 'gfx950':
        # Same flydsl-free-at-call-time rule as the gfx1201 branch above.
        from fmha_tuning_gfx950 import FmhaInputMetadata as Gfx950InputMetadata, fmha_knobs

        tile = choices.BLOCK_DMODEL
        meta = Gfx950InputMetadata(
            # See the gfx1201 branch: not a functional axis, unread by
            # resolve() for the same reason (STRIDE_TOKEN sits behind
            # strides_constexpr, pinned False below).
            #
            # STRIDE_TOKEN is not the only reader on this arch, though.
            # Upstream's `_store_lse_row` also sizes LSE's per-batch slice with
            # `traits.NUM_HEADS_Q`, because upstream compiles a kernel per
            # shape and AOT cannot. That one is answered in the kernel, which
            # takes the count off the `num_head_q` kernarg instead
            # (`ParityStoreHelper._store_lse_row_unguarded`, a local edit to
            # the vendored kernel). Left as the trait it silently dropped every
            # head but `h == 0`, which only became visible once `return_lse`
            # below was pinned on.
            num_heads=1,
            head_dim=tile,
            # gfx950's FmhaInputMetadata has no `causal_type` field — only
            # `causal`/`window` (window requires causal: "a left bound *on
            # top of* the causal one"). The kernel only ever compiles
            # CAUSAL_TYPE in {0, 3}, which is also all the operator's own
            # axis offers here, so this is 1:1, no mapping, exactly like
            # the gfx1201 branch's causal_type line.
            causal=choices.CAUSAL_TYPE != 0,
            window=choices.CAUSAL_TYPE != 0,
            dtype_str='bf16' if '*bf16' in choices.arg('Q') else 'f16',
            bias=bool(choices.BIAS_TYPE),
            dropout=bool(choices.ENABLE_DROPOUT),
        )
        knobs = fmha_knobs(
            arch,
            block_dmodel=tile,
            padded_head=choices.PADDED_HEAD,
            # AOT cannot bake strides: one binary must serve every layout.
            strides_constexpr=False,
            # Pinned, not left to the policy: this pin is what makes
            # the Workspace/BlockTable/block_table_stride `options=[0]`
            # declaration above true. Asserted below, not just assumed.
            paged=False,
            num_kv_splits=1,
            # LSE **is** optional for AOTriton -- `attn_fwd_params::L` is
            # declared "Can be T2::get_null_tensor()", and an inference caller
            # passes exactly that -- but it is optional at RUNTIME, decided per
            # launch by a null-pointer test, the way the Triton kernel's
            # `L_not_null` and gfx1201's `_l_valid` decide it. `return_lse` is
            # not that switch: it is a compile-time knob that deletes the store
            # from the binary, and one AOT binary has to serve both kinds of
            # caller. So it is pinned on, and the null case is handled in the
            # kernel (`ParityStoreHelper._store_lse_row`, which a local edit
            # to the vendored kernel added the guard to).
            #
            # Pinned rather than left alone because upstream's default is
            # `return_lse=False` -- inference builds do not want the store --
            # and `_GFX950_FALLBACK` supplies that default for every field the
            # caller leaves unset. Left unpinned, `fmha_wide_gfx950`'s
            # `if const_expr(traits.RETURN_LSE)` compiles the store away and
            # the kernel returns without ever touching a non-null LSE: the
            # tensor keeps whatever the caller allocated, which the test
            # harness fills with NaN on purpose, and every case dies on
            # "L tensor has NaN" with a launch that reported success. gfx1201
            # has no such knob -- it always emits the store, and guards it --
            # which is why this is pinned in this branch and not the one above.
            return_lse=True,
        ).resolve(meta)
        assert knobs.block_dmodel == tile, (
            f'resolve() returned block_dmodel={knobs.block_dmodel} for '
            f'BLOCK_DMODEL={tile}; the compiled tile must be the operator axis')
        assert not knobs.strides_constexpr, \
            'num_heads=1 is only safe while STRIDE_TOKEN stays behind ' \
            'strides_constexpr (the LSE descriptor, the other NUM_HEADS_Q ' \
            'reader on this arch, is handled in the kernel)'
        assert knobs.return_lse, \
            'the LSE store must exist in every AOT binary (null L is a runtime ' \
            'test, not a build variant); resolve() must not have cleared it'
        assert not knobs.paged and knobs.num_kv_splits == 1, (
            'AOT gfx950 only ever pins paged=False, num_kv_splits=1 -- that pin '
            'is what makes the Workspace/BlockTable/block_table_stride constexpr '
            'fold (options=[0]) true; resolve() must not have overridden it')

        def build():
            """Deferred: constructs the FlyDSL module. Imports flydsl transitively,
            so ONLY `aotriton.flyc_compile` (run by ninja) may call this."""
            from flash_attn_func_gfx950 import build_flash_attn_func_gfx950_module_primary
            return build_flash_attn_func_gfx950_module_primary(meta, knobs)

        build.flyc_source = 'flash_attn_func_gfx950.py'
        build.flyc_kernel_name = 'flash_attn_func_gfx950_kernel'

        # Gfx950Knobs.GRID_AXIS_ORDER is a flat resolved field already (FlyDSL
        # 70b2dbc5 made the class POD) -- no mirroring needed, unlike the
        # gfx1201 branch above.
        return build, asdict(knobs)

    else:
        # Unreachable: this backend serves exactly the architectures in
        # _FLYC_FWD_LADDERS. Fail loudly rather than silently building nothing
        # if that ever stops being true.
        assert False, f'flyc_attn_fwd: no builder branch for arch {arch!r}'
