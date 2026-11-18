# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI description of the flash attn_fwd FlyDSL backend (gfx1201 and gfx950).

Registered as backend 2 of `op_attn_fwd`, through `metro_fwd_flyc`.
`python -m aotriton.flyc_compile` on this file also produces an hsaco for
either architecture with no GPU; see `docs/FlyDSL.md`.

A FlyDSL backend is a THIRD kind, and it takes one half from each of the two ATI
already has:

    triton (@ati.source)   compiled during the build, ATI owns the perf space,
                           1:1 argument names, functional axes ARE kernel params
    aiter  (@ati.affine.*) prebuilt .co, no perf space, no axes of its own —
                           inherits the operator's and filters them
    flyc   (@ati.flyc.*)   compiled during the build (like triton), inherits and
                           filters the operator's axes (like aiter), and — unique
                           to it — dispatches an hsaco whose kernarg list is NOT
                           the operator's

Three consequences, each handled below:

1. **The tuning table is programmatic.** `gfx1201.tuning_fwd.resolve_knobs()` is
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
   them and narrows with `@ati.disable`, exactly as `aiter_fwd.py` does.
   `flyc_attn_fwd` then reads `choices.NAME` (a `ChoiceView`)
   — the OPERATOR's choices, parsed from `--signature` text by the driver — and
   maps them to builder knobs. This is also why the axes cannot be re-declared
   as a PLAIN `@ati.scalar`: they are not arguments of the flyc kernel at all,
   they are build-time Python values.

   `BLOCK_DMODEL`/`PADDED_HEAD` ARE declared below, near the bottom of the
   stack — but as MARKERS (`options=...` plus
   `wires_to=ati.context_helper(...)`, and no type), a different, narrower shape
   from every other `@ati.scalar` in this file. A marker never becomes a kernel
   argument, since `options=` and an explicit type are mutually exclusive, so it
   can never claim a kernel-argument slot the way `varlen_bits`'s `'i32'` does.
   It exists purely so that `godel_number()` can be redirected to read the
   rounded value a context helper computes, rather than the raw, off-ladder
   choice: this backend's ladder is a strict subset of the declared axis. See
   `ir/flyc/kdesc.py`'s `context_helper_for_functional` for the mechanism, and
   for why it must not be confused with the three real, explicit-type helpers
   above it.

3. **The kernarg list is declared, because nothing else can supply it.** aiter hands
   the params struct to a C++ cookie and never names a kernarg; triton gets its list
   from the kernel signature. flyc dispatches the hsaco directly, and the hsaco's
   layout comes from `flash_attn_func_aiw_kernel`, so the wiring below is the
   description's real payload.
"""

from dataclasses import dataclass, asdict

import aotriton.template_instantiation as ati
from ._common import flash_disabled, check_value
from ._flyc_common import flyc_input_metadata


@dataclass
class FlycFwdHints:
    """Tuning inputs the builder may read that are NOT functional axes.

    Registered with `@ati.flyc.hints` — namespaced with the rest of the FlyDSL-specific
    surface rather than added to `ati.tune.*`, which is the SHARED tuning vocabulary
    (schema / configs / binning / fallback all feed the LUT and the tuning DB; this
    feeds one builder and nothing else). `ati.affine.*` is the same pattern for aiter.

    FlyDSL would call these part of the **problem**, opposite its **schedule**
    (`gfx1201/tuning_fwd.py`: *"a caller states a problem, the tuning policy
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


# The compiled tile ladder, from gfx1201.tuning_fwd._BLOCK_DMODEL_LADDER. A literal,
# not an import: the generator parses this module and has no flydsl. `flyc_build`
# (build time) does import flydsl, and `resolve_knobs` re-validates the tile, so a
# drift here fails loudly at build rather than silently emitting the wrong kernel.
FLYC_HEAD_DIMS = frozenset({16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 384, 512})

# gfx950's ladder, from gfx950.tuning_fwd.LADDER -- a literal for the same reason
# as FLYC_HEAD_DIMS above. Narrower than gfx1201's: no 16/48/80 (PV_MFMA_N is 32,
# so only multiples of 32 are compiled). 96 IS
# included, matching upstream's LADDER verbatim: a comment right above that tuple
# in gfx950/tuning_fwd.py claims 96 "computes the wrong answer", but that is
# stale prose left behind when the bug it described was fixed the same day
# (upstream `98493cc7`, an ancestor of the vendored commit; it added 96 to the
# tuple in the same change). Do not re-derive this by excluding 96 "to be safe":
# that would silently give up a rung upstream measured at 932 TF against 579 for
# the 128-tile fallback, on the strength of a comment its own repository had
# already superseded.
FLYC_GFX950_HEAD_DIMS = frozenset({32, 64, 96, 128, 160, 192, 224, 256, 384, 512})

# Per-arch ladder lookup, keyed the same way the disable predicate and the
# builder branch below are. Also what `_flyc_fwd_disabled` uses to decide which
# arches this backend serves at all -- `f.arch in _FLYC_FWD_LADDERS`.
_FLYC_FWD_LADDERS = {
    'gfx1201': FLYC_HEAD_DIMS,
    'gfx950': FLYC_GFX950_HEAD_DIMS,
}


def _flyc_fwd_disabled(f):
    """Everything this backend cannot serve, in one predicate.

    Arch lives here rather than in a `@ati.flyc.arch([...])` declaration: it is one
    more exclusion among several, and splitting it across two mechanisms means two
    places to look when a functional unexpectedly has no flyc kernel. (`aiter_fwd.py`
    uses @ati.affine.arch; that predates `f.arch` being available to predicates.)

    `_FLYC_FWD_LADDERS` maps each served arch to its own compiled ladder;
    everything else about the predicate -- the WMMA dtype restriction, the cited
    flash_disabled() call -- is arch-independent.
    """
    if f.arch not in _FLYC_FWD_LADDERS:
        return True
    if flash_disabled(f):
        return True
    # f16/bf16 WMMA only.
    if '*fp32' in check_value(f, ['Q']):
        return True
    # Off-ladder head dims are rejected outright by `resolve_knobs`/`resolve()`,
    # not rounded: rounding is the *interface*'s job, because it also has to
    # arrange the runtime extent and the padded-head contract that make the
    # rounding safe.
    if check_value(f, ['BLOCK_DMODEL']) not in _FLYC_FWD_LADDERS[f.arch]:
        return True
    return False


@ati.start
# Overriding drops nothing: the cited disable is the WEAKER of the two, despite
# the kwarg's name. It is flash_disabled(f, gfx950_bad_hdims={16}); this
# predicate calls flash_disabled(f) unconditionally and only narrows further
# (arch membership, the WMMA dtype restriction, the per-arch ladder), so every
# functional the cited predicate disables is disabled here too -- gfx950 head_dim
# 16 in particular is excluded via _FLYC_FWD_LADDERS['gfx950'] (see the ladder's
# own comment), which is a strict superset of the cited predicate's {16} on that
# axis alone. Measured while gfx1201 was the only served arch: of 576
# functionals the cited predicate disables 144, and all 144 were disabled here
# too.
@ati.disable(when=_flyc_fwd_disabled,
             I_understand_this_overrides_cited_disable=True)
# `cite` fills the GAPS: any argument below that this description does not fully
# claim (dtype variables, strideless operands) is cloned from the triton kernel by
# apparel name, so the two backends cannot drift on a shared operand's type.
@ati.cite('op_attn_fwd.triton.attn_fwd')
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
# The order is frozen and load-bearing: gfx1201/utils.py's pointer_to_llvm_ptr records that
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
#
# That rejection is worth naming here, because it moves with this declaration:
# `flyc_classify_varlen()` throws on an underivable count, and it is
# `flyc_num_seqlens()` that calls it. Still every launch -- all three
# descriptions declare `num_seqlens` with this helper, and helpers are evaluated
# once per `lookup_optimal`.
@ati.scalar('num_seqlens', 'i32', wires_to=ati.context_helper('flyc_num_seqlens'))
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
@ati.scalar('idropout_p',    'i32',  wires_to=ati.context_helper('flyc_idropout_p'))
@ati.scalar('dropout_scale', 'fp32', wires_to=ati.context_helper('flyc_dropout_scale'))
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
# --- why `ati.context_helper` and not an inline expression --------------------
#
# Three arguments above are not renames of an operator operand. They need
# host-side code, and the place that code already goes is a member function on
# the generated CONTEXT class, hand-implemented in modules/flash/csrc/:
#
#     wires_to=ati.context_helper('flyc_num_seqlens')
#       -> declares  int32_t flyc_num_seqlens() const;  on FlycAttnFwdContext
#       -> author implements it in modules/flash/csrc/flyc_attn_fwd.cc
#
# This is not a new concept. `dim3 grid_calculator() const;` is declared in the
# generated context struct (codegen/template/shim.h) and implemented by hand as
# `AttnFwdContext::grid_calculator()` in modules/flash/csrc/attn_fwd.cc -- same
# class, same namespace (AOTRITON_NS::v3::flash), same split. `context_helper`
# only lets a description declare MORE of them instead of the set being
# hardwired.
#
# NO ARGUMENTS, because the context already carries everything: `params` (the
# whole operator params struct), the dispatching `current_gpu`, and the selected
# knobs are members. grid_calculator demonstrates the range -- it reads
# params->Varlen_bits, params->Max_seqlen_q, params->Q->size(1), params->Batch,
# this->BLOCK_M and this->PERSISTENT_TYPE, and takes nothing.
#
# The return type is not declared twice: it comes from the @ati.scalar type on
# the same line, so `'i32'` fixes the signature as `int32_t`.
#
# Rejected alternative -- `wires_to=ati.expr('<C++ expression>')`, i.e. a C++
# string in the Python description:
#   * a typo is caught only when the generated .cc compiles, and the error points
#     at generated code; a missing context helper is a LINK error naming the
#     symbol
#   * a string cannot be unit-tested, stepped through, or given a comment
#     explaining the varlen decoding at the point it happens
#   * anything past one expression does not fit
#   * two mechanisms for host-side code -- grid_calculator in csrc, expressions
#     in the description -- means two places to look
# The cost is that trivial cases (dropout_scale is one divide) also become
# functions. Uniformity is worth it: one mechanism, one file, one place a
# reviewer looks.
#
# THE LAUNCH GRID lands in the same place: `grid_calculator()` is already the
# hook, and FlycAttnFwdContext implements it in modules/flash/csrc/. The knobs
# it needs -- `block_m`, and the axis order -- are not perf fields on the
# context, because flyc declares no perf axes; they arrive in the selected
# image's `#P` section and are read back through `perf()`.
#
# --- functional-axis markers, NOT kernel arguments ----------------------------
#
# `BLOCK_DMODEL`/`PADDED_HEAD` are the operator's own axes, inherited via
# `@ati.cite` above and narrowed by `_flyc_fwd_disabled`'s ladder check -- NOT
# part of `flash_attn_func_aiw_kernel`'s argument list declared above, so these
# two lines are deliberately kept out of that ordered block. `options=` and an
# explicit type are mutually exclusive, so a marker can never be read by
# `iter_launch_arguments` as a real launch argument; it is only ever found by
# axis name, via `context_helper_for_functional` (ir/flyc/kdesc.py).
#
# Why either axis needs a marker at all: dispatch bins the caller's head
# dimension to a `BLOCK_DMODEL` rung on the OPERATOR's ladder before a backend
# is chosen, and this backend's compiled ladder is a strict subset of it. Wiring
# the axis to a context helper lets `godel_number()` read the ROUNDED value the
# helper computes (modules/flash/csrc/flyc_attn_fwd.cc) instead of the raw,
# potentially off-ladder choice. `PADDED_HEAD` must follow that decision -- a
# kernel re-rounded to a wider rung with `PADDED_HEAD` left false is a silent
# wrong answer, not a build error.
#
# The `options=` lists restate each ladder as free documentation, not as a
# second source of truth: the digit's real range comes from the OPERATOR's
# axis, and the backend's own disable predicate is what enforces the subset.
# One shared marker declaration serves both architectures (like the
# Workspace/BlockTable declaration above, this decorator stack does not vary by
# arch), so `options=` is the UNION of both ladders rather than either alone --
# neither is a subset of the other, so one arch's list would under-document the
# other's compiled rungs.
@ati.scalar('BLOCK_DMODEL', options=sorted(FLYC_HEAD_DIMS | FLYC_GFX950_HEAD_DIMS),
            wires_to=ati.context_helper('flyc_block_dmodel'))
@ati.scalar('PADDED_HEAD', options=[False, True],
            wires_to=ati.context_helper('flyc_padded_head'))
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

    Deliberate asymmetry with `_flyc_fwd_disabled` above, which takes a real
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

    `resolve_knobs`, NOT an arch's `plan`: `plan()` is the JIT
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
    from tuning_fwd import resolve_knobs

    meta = flyc_input_metadata(choices)
    knobs = resolve_knobs(arch, meta)

    def build():
        """Deferred: imports flydsl, so only `aotriton.flyc_compile` calls it."""
        from tuning_fwd import builder
        return builder(arch, meta, knobs)

    return build, asdict(knobs)
