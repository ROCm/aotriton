# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
KernelDescription — the codegen-facing IR for a flyc (FlyDSL-compiled) kernel.

Mirrors `ir/affine/kdesc.py`'s shape: a minimal `Interface` implementer that owns
no functional space of its own — a flyc kernel inherits and filters the functional
space of the operator that lists it as an `@ati.backend`. Built by the linker
(`codegen/linker.py:_build_flycs`) from a `FlycDecl`; `ir/ops/infer.py` then binds
the Operator in as `SHARED_IFACE`, which is the same object as
`functionals_source`.

`_axes_overrides` / `axes_multi` delegate to `functionals_source` so
`Interface.gen_functionals` enumerates exactly the axes the operator does, but
every yielded `Functional` still carries `meta_object=self` (this flyc kdesc, not
the operator). That is what gives filepack paths, hsaco entry names and the
`Fly.compile` KERNEL_NAME column this kernel's own NAME/FAMILY rather than the
operator's — a flyc kernel gets its own `.aks2`, keyed by the description's name,
rather than sharing the operator's.
"""

from ..interface import Interface
from ..context_helper import ContextHelper
from ..choices import ChoiceVarAbsent
from aotriton.template_instantiation.builder import DescriptionError


# Elemental type string (as authored on an @ati.scalar) -> C scratch-member
# type. Only the types actually used by today's context helpers
# (flyc_batch_size/flyc_num_seqlens/flyc_idropout_p: 'i32';
# flyc_dropout_scale: 'fp32') are exercised, but the table covers the full
# elemental vocabulary a future helper could plausibly use. Deliberately NOT
# typed_choice.ELEMENTAL_TYPE_MAP, which maps to TypedChoice factory classes,
# not C type-name strings.
_CONTEXT_HELPER_CTYPE = {
    'i8': 'int8_t', 'i16': 'int16_t', 'i32': 'int32_t', 'i64': 'int64_t',
    'u8': 'uint8_t', 'u16': 'uint16_t', 'u32': 'uint32_t', 'u64': 'uint64_t',
    'fp16': 'float', 'bf16': 'float', 'fp32': 'float',
}


class KernelDescription(Interface):
    """A flyc kernel built from the @ati.flyc.* stacked form. ATI-native, like
    AffineKernel: subclasses the ATI Interface base (identity surface) directly,
    no functional space of its own (inherits the operator's via
    `functionals_source`), and no perf space of its own — every functional
    resolves to exactly one hsaco, whose knob set is carried verbatim in the
    entry name's `#P` section rather than chosen among candidates
    (`ir/flyc/ksignature.py`)."""

    CODEGEN_MODULE = 'flyc'
    TUNE_NAME = 'flytune'
    FILE_PFX = 'flyc'
    ENUM_PREFIX = 'kFlyc_'
    is_tunable = False

    def __init__(self, built, *, family, source_path,
                functionals_source=None, tensors=None, scalars=None,
                builder_fn=None, hints_cls=None):
        # `built` is the BuiltKernel build_kernel() lowered this kernel's
        # cite-resolved FlycDecl clone into: it answers
        # "what are this kernel's arguments?" (order, disables) -- NOT "which
        # variants exist?", which stays functionals_source's job below (the
        # design caveat: a BuiltKernel giving flyc its own axes would make it
        # enumerate a second, wrong functional space).
        self._built = built
        self.NAME = built.name
        self.FAMILY = family
        # `source_path` spelled exactly as ir/triton/kdesc.py spells it: one
        # name for "the file this kernel's code lives in", across languages.
        self.source_path = source_path
        # The Operator that lists this kernel as an @ati.backend. None only
        # transiently, between construction and `ir/ops/infer.py` binding it --
        # `codegen/linker.py:_check_flycs_bound` fails linking if it is still
        # None afterwards, so no kdesc the generator sees has it unset.
        self._functionals_source = functionals_source
        self.desc_path = None                          # set by the linker; Fly.compile's DESC column
        # The @ati.tensor/@ati.scalar decorator specs stacked on the description
        # (FlycDecl.tensors/.scalars), and the builder placeholder function itself
        # plus its hints dataclass (FlycDecl.fn/.hints_cls) -- all threaded through
        # by the linker's _build_flycs. These are what
        # iter_launch_arguments()/iter_context_helpers() walk, and what
        # FlycTuneCodeGenerator (python/codegen/flytune.py) calls directly
        # (builder_fn(arch, choices, hints), plus hints() for the 3rd
        # argument) to obtain the knobs dict at generate time -- WITHOUT ever
        # calling the `build` callable that same call returns, which is what
        # would import flydsl.
        self.tensors = list(tensors) if tensors else []
        self.scalars = list(scalars) if scalars else []
        self.builder_fn = builder_fn
        self.hints_cls = hints_cls
        # The real, AST-parsed flyc kernel parameter list, per arch. It cannot
        # be known at link time: the vendored file and def name are set by the
        # description's builder once `arch` is resolved, and no `Functional`
        # exists yet to call `builder_fn` with. So it is resolved lazily, once
        # per arch, by `ensure_stub_resolved` (called from codegen/flytune.py's
        # `_gen_signatures` right after it calls `builder_fn(arch, choices,
        # hints)` for a real functional). Cached here rather than recomputed
        # per functional: today every functional of one arch shares the same
        # vendored file and def name, and re-parsing that file for every one of
        # an arch's functionals would be needless AST work even once that stops
        # being true.
        self._real_params_by_arch = {}

    @property
    def gpu_symbol_name(self):
        """The HIP kernel symbol the hsaco actually exports: `self.NAME`, the
        flyc DESCRIPTION's own identity (e.g. 'flyc_attn_fwd').

        Deliberately NOT FlyDSL's own naming for the `@flyc.kernel` def plus its
        internal kernel id ('flash_attn_func_aiw_kernel_0'). That would need the
        AST-located stub, which is not resolved this early, and borrowing
        FlyDSL's internal name is fragile regardless -- a per-arch vendored file
        could rename the def, or emit a non-zero kernel id, without this shim
        noticing. Naming the symbol after the description makes it a fact ATI
        itself controls:
        `python/flyc_compile.py` sets the `KernelFunction`'s `_name` to
        exactly this before tracing, so the ELF should export it verbatim.
        `flyc_compile.py`'s `_verify_elf` checks that the symbol it actually
        finds in the ELF matches this name and fails the build if they
        diverge -- a convention change should stop the build, not produce
        kernels that cannot be looked up."""
        return self.NAME

    @property
    def perf_cfields(self):
        return []

    def _axes_overrides(self):
        """(axes, overrides) to enumerate over -- the operator's own, since a flyc
        kernel has no axes of its own. `Interface.gen_functionals` uses these but
        still stamps `meta_object=self` on every yielded Functional (see the
        module docstring)."""
        return self._require_functionals_source()._axes_overrides()

    @property
    def axes_multi(self):
        """Delegated to functionals_source: `Functional.compact_choices` /
        `.unified_signature` (ir/functional.py) read `meta_object.axes_multi`, and
        meta_object is THIS kdesc once `_axes_overrides` is wired in (see above)."""
        return self._require_functionals_source().axes_multi

    def _require_functionals_source(self):
        assert self._functionals_source is not None, (
            f'flyc kernel {self.NAME!r} has no functionals_source: nothing '
            f'bound it to the operator that lists it as an @ati.backend')
        return self._functionals_source

    def list_functional_params(self):
        return self._require_functionals_source().list_functional_params()

    @property
    def func_cfields(self):
        # A flyc kernel's kernarg ABI is NOT the operator's params struct, so
        # it contributes no field to the union that struct is built from: every
        # operand it reads is already contributed by the operator's other
        # backends, and its own arguments are reached through wires_to instead.
        return []

    # --- identity: borrow the operator's params/context/call_options surface.
    # Mirrors ir/triton/kdesc.py's SHARED_IFACE-aware param_class_name, except
    # for flyc SHARED_IFACE is always the referenced operator (never None) --
    # a flyc kernel never owns its own params struct.

    @property
    def SHARED_IFACE(self):
        """The Operator this kernel borrows everything from -- the same object as
        `functionals_source`, because for flyc they ARE the same relationship:
        the operator owns the params struct AND the functional space, and this
        kernel has neither of its own.

        Readable before binding (returns None) and settable, so
        `ir/ops/infer.py`'s `infer_shared_iface` can bind it by the same
        `sub.SHARED_IFACE = op` it uses for every other kernel that borrows an
        operator's surface. Without that, flyc would need a second, bespoke
        binding pass for what is the same relationship under a different name."""
        return self._functionals_source

    @SHARED_IFACE.setter
    def SHARED_IFACE(self, op):
        self._functionals_source = op

    @property
    def param_class_name(self):
        return self.SHARED_IFACE.param_class_name

    @property
    def godel_number(self):
        return self._require_functionals_source().godel_number

    @property
    def axes_all_ordered(self):
        return self._require_functionals_source().axes_all_ordered

    def axis_of_arg(self, aname):
        return self._require_functionals_source().axis_of_arg(aname)

    def axis_by_var(self, var_name):
        return self._require_functionals_source().axis_by_var(var_name)

    def override_for(self, aname):
        return self._require_functionals_source().override_for(aname)

    def apparel_of(self, real_arg):
        return self._require_functionals_source().apparel_of(real_arg)

    def real_of(self, apparel_arg):
        return self._require_functionals_source().real_of(apparel_arg)

    def is_functional_disabled(self, functional):
        # Mirrors ir/triton/kdesc.py's is_functional_disabled exactly:
        # self._built.disables is the cite-resolved list (build_kernel /
        # resolve_cites already applied the rule that a local @ati.disable
        # replaces the cited one), so there is no separate "self._disable"
        # concept left here.
        for d in self._built.disables:
            try:
                if d.holds(functional):
                    return True
            except ChoiceVarAbsent as e:
                pred = getattr(d.when, '__name__', d.when)
                raise DescriptionError(
                    f"kernel {self.NAME!r}: the @ati.disable predicate {pred!r} reads "
                    f"a choice variable this kernel does not have ({e}). It is likely "
                    f"inherited via @ati.cite from a kernel with a different choice "
                    f"space. Declare a local @ati.disable on {self.NAME!r} that uses "
                    f"only its own choice variables (a local disable replaces the "
                    f"cited one).") from e
        return False

    # --- builder invocation: the code generator (FlycTuneCodeGenerator,
    # python/codegen/flytune.py) calls `self.builder_fn(arch, choices, hints)`
    # directly, keeps the `knobs` half of its `(build, knobs)` return, and
    # passes `build` to `ensure_stub_resolved` (above) for its two attribute
    # reads -- see that module for the sys.path setup.
    # There is deliberately no `build()` method on this class: the generator
    # must never invoke the FlyDSL compiler, and a wrapper method here would
    # just be one more place that could grow a `build()` call by mistake. Only
    # `python/flyc_compile.py`, run by ninja at build time, may call the
    # deferred `build` callable a description's fn(arch, choices, hints)
    # returns. ---

    def hints(self):
        """The hints dataclass instance passed as the builder's 2nd argument, or
        None if the description declared no @ati.flyc.hints. flytune.py has no
        `--hints` CLI override at generate time (flyc_compile.py's is a Phase-1,
        stand-alone-driver-only concept), so defaults are always used."""
        return self.hints_cls() if self.hints_cls is not None else None

    # --- launch-argument vector ---

    def ensure_stub_resolved(self, arch, build):
        """Populate the `arch` entry of the real-kernel-parameter cache from
        `build` -- the closure a description's `builder_fn(arch, choices,
        hints)` just returned. Reads only `build.flyc_source` /
        `build.flyc_kernel_name` (two plain strings); NEVER calls `build`
        itself (that would invoke the FlyDSL compiler from the generator,
        which must not happen -- see the note above `hints()`).

        Idempotent per arch (a no-op once cached), so `codegen/flytune.py`
        can call this once per functional without re-parsing the vendored
        file for every one of an arch's many functionals."""
        if arch in self._real_params_by_arch:
            return
        from pathlib import Path
        from ...specs.flyc import _flyc_kernel_stub

        module_path = Path(self.source_path) / build.flyc_source
        stub = _flyc_kernel_stub(module_path, build.flyc_kernel_name)
        # `stub.params` is already the plain parameter-name list
        # (ast_params.collect_params returns `[p.arg for p in ...]`, not
        # ParamSpec objects -- KernelStub is shared with @ati.source, whose
        # own params are always bare strings too).
        self._real_params_by_arch[arch] = list(stub.params)

    def iter_launch_arguments(self, arch):
        """Yield the C++ launch-argument vector entries in the REAL kernel's
        signature order (see codegen.common.LaunchArg). Unlike triton's
        equivalent, there is no `aux` (no global_scratch/profile_scratch: the
        flyc kernel's 44 parameters don't include Triton's two trailing scratch
        pointers) and a 4th LaunchArg.kind, 'context_helper', is possible.

        `arch`-keyed: the real parameter order cannot come
        from `self._built.arguments` (the single, arch-independent list
        `build_kernel` computed at link time from a synthesized stand-in, see
        specs/flyc.py's `_synth_param_order`) -- it must be the real, AST-
        resolved signature, which `ensure_stub_resolved` must already have
        cached for `arch` (codegen/flytune.py's `_gen_signatures` does this
        for every functional before this method is ever called)."""
        # Lazy: same aotriton.codegen -> template_instantiation cycle as
        # ir/triton/kdesc.py's iter_launch_arguments.
        from aotriton.codegen.common import LaunchArg

        real_param_order = self._real_params_by_arch.get(arch)
        assert real_param_order is not None, (
            f'flyc kernel {self.NAME!r}: no resolved kernel stub for arch '
            f'{arch!r} -- ensure_stub_resolved(arch, build) must run before '
            f'iter_launch_arguments (see codegen/flytune.py _gen_signatures)')

        tensor_ptr_lookup = {}
        for t in self.tensors:
            for name in t.arg_names:
                tensor_ptr_lookup[name] = t
        scalar_lookup = {}
        for s in self.scalars:
            for name in s.arg_names:
                scalar_lookup[name] = s
        stride_owner = {}
        for t in self.tensors:
            for dim, sname in enumerate(t.match_strides(real_param_order)):
                stride_owner[sname] = (t, dim)

        def _apparel(spec, name):
            """The operand `name` is dressed as: its wiring if it has one, else
            itself.

            `name`, NOT `spec.arg_name`. A spec can bind SEVERAL argument names
            under one choice dimension (`@ati.scalar(['Q_descale','K_descale'],
            ...)`), and `arg_name` is only the representative -- the FIRST -- of
            them, so keying on it would emit `params.Q_descale` for K_descale
            too. `wires_to=` is rejected by both decorators for a name list, so
            reading it off the spec stays correct: a wired spec always binds
            exactly one name, and that name is `name`."""
            return spec.wires_to if isinstance(spec.wires_to, str) else name

        for name in real_param_order:
            if name in stride_owner:
                t, dim = stride_owner[name]
                # A strided tensor is always singly-named (`@ati.tensor` refuses
                # strides= on a name list), so the owner's own name is the one to
                # dress -- `name` here is the STRIDE argument, not the tensor.
                yield LaunchArg(aname=name, kind='tensor_stride',
                                 expr=f'params.{_apparel(t, t.arg_name)}->kparam_stride({dim})')
            elif name in tensor_ptr_lookup:
                t = tensor_ptr_lookup[name]
                yield LaunchArg(aname=name, kind='tensor_ptr',
                                 expr=f'params.{_apparel(t, name)}->kparam_data_ptr()')
            elif name in scalar_lookup:
                s = scalar_lookup[name]
                if isinstance(s.wires_to, ContextHelper):
                    yield LaunchArg(aname=name, kind='context_helper',
                                     expr=f'CAST(&context.scratch_params.{s.wires_to.name})')
                else:
                    yield LaunchArg(aname=name, kind='scalar',
                                     expr=f'CAST(&params.{_apparel(s, name)})')
            else:
                assert False, (
                    f'flyc kernel {self.NAME!r}: undeclared kernel parameter '
                    f'{name!r} (not bound by any @ati.tensor/@ati.scalar and not '
                    f'a resolved stride argument)')

    def pp_arg_doc(self, aname):
        """(is_constexpr, comment_value) for one REAL kernel argument's
        pp_args entry -- flyc's counterpart to `ir/functional.py`'s
        `Functional.pp_arg_doc`, read by `codegen/flytune.py`'s
        `codegen_deduplicated_pp_args_function_index` the same way
        `codegen/autotune.py` reads the Triton version.

        Deliberately does NOT delegate to `Functional.resolved` the way the
        Triton version does. `aname` here can be a name like `Workspace`,
        `BlockTable` or `block_table_stride` -- real parameters of the gfx950
        kernel signature (`real_param_order`, above) that are not, and never
        will be, axes of the shared operator this kdesc borrows its
        functional space from (module docstring: a flyc kdesc adds no axes
        of its own). `functionals_source.resolved` simply has no entry for
        them, so this cannot ask the question the Triton version asks.

        Instead, constexpr-ness comes straight off THIS kernel's own
        `@ati.scalar([...], options=[...])` declaration for `aname` (e.g.
        `flyc_attn_fwd.py`'s `@ati.scalar(['Workspace', 'BlockTable',
        'block_table_stride'], options=[0])`) -- deliberately independent of
        whatever FlyDSL's own `_WS_ANN`/`_BT_ANN`/`_BTS_ANN` resolve to in the
        vendored kernel source. That independence is what makes it safe: our
        builds pin `paged=False, num_kv_splits=1` everywhere on gfx950, so
        the annotation these three collapse to in the real kernel is the
        same for every functional, uniformly -- there is no per-functional
        divergence for a shared, deduplicated pp_args registration to
        desync against. That is NOT true in general -- see the long docstring
        on `codegen_deduplicated_pp_args_function_index` for why an axis whose
        constexpr-ness genuinely varies by functional must NOT be folded here
        -- it holds only because this declaration is a per-description,
        build-wide fact rather than a per-functional derived one.

        An axis-MARKER scalar (`BLOCK_DMODEL`/`PADDED_HEAD`, also
        `options=`-declared) never collides with this: it is never a real kernel argument, so it
        never appears in `real_param_order` and this method is never asked
        about it via `iter_launch_arguments`'s yielded `aname`s."""
        for s in self.scalars:
            if aname not in s.arg_names:
                continue
            if s.options is None:
                return False, None
            if len(s.options) == 1:
                return True, str(s.options[0])
            return True, '/'.join(str(o) for o in s.options)
        return False, None

    def context_helper_for_functional(self, aname):
        """The context-helper member-function name that should stand in for
        functional axis `aname` in godel_number(), or
        None for the default `args.<aname>` read (see Interface's base
        default). A functional axis MARKER is helper-wired the same way any
        other argument is -- a local `@ati.scalar(aname, options=...,
        wires_to=ati.context_helper(...))` -- even though (unlike a real
        kernel argument) `aname` here never appears in `real_param_order`, so
        `iter_launch_arguments` never yields it as a launch argument. It only
        needs to be *findable by axis name*, which is exactly what
        `self.scalars` + `arg_names` already gives for free.

        Deliberately `s.type_ is None` (the MARKER shape, i.e.
        BLOCK_DMODEL/PADDED_HEAD), not just any context_helper-wired scalar:
        the pre-existing explicit-type helpers (`flyc_num_seqlens` on
        `num_seqlens`, `flyc_idropout_p`, `flyc_dropout_scale`) also match a
        functional axis's `aname` by
        argument name, but their helper computes a TRANSFORMED value for the
        launch argument (e.g. `flyc_num_seqlens`'s `nseq_idx = num_seqlens !=
        0 ? num_seqlens : batch_size`), not a stand-in for that axis's own
        pinned choice -- redirecting godel_number() to read it would compare
        the transformed value against the axis's untransformed choice list
        and get the wrong digit. Caught by generating flyc_bwd_dq for gfx950:
        `num_seqlens` (an explicit-type helper, `type_='i32'`) was being
        matched here before this restriction, and sub-step (d)'s independent
        recompute assertion (codegen/flyc.py) then found zero functionals
        that were both 'not disabled' and had a `num_seqlens` choice equal to
        one of `flyc_num_seqlens()`'s actual (unrelated) return values."""
        for s in self.scalars:
            if s.type_ is not None:
                continue
            if aname in s.arg_names and isinstance(s.wires_to, ContextHelper):
                return s.wires_to.name
        return None

    def iter_context_helpers(self):
        """Yield (helper_name, c_type) once per distinct `ati.context_helper`
        referenced by this kernel's @ati.scalar specs, in first-seen order.
        Drives both the context struct's declares/scratch-members (flyc.h) and
        the evaluation block in lookup_optimal() that populates them.

        Two shapes of context_helper scalar exist:
          * an explicit-type real kernel argument (`@ati.scalar('varlen_bits',
            'i32', wires_to=...)`) -- `s.type_` names the elemental type, and
            `_CONTEXT_HELPER_CTYPE` maps it to a C type string;
          * a functional-axis MARKER (`@ati.scalar('BLOCK_DMODEL',
            options=[...], wires_to=...)`) that is never a real kernel
            argument at all (see context_helper_for_functional) -- `options=`
            and an explicit type are mutually exclusive
            (decorators/scalar.py), so `s.type_` is always None here. The
            elemental type instead comes from the AXIS's own TypedChoice
            (`axis_of_arg(s.arg_name).repr_typed_choice.itype` already IS a
            full C type string, e.g. 'bool' for PADDED_HEAD's
            `options=[False, True]` -- constexpr.bool_t.itype -- or 'int16_t'
            for BLOCK_DMODEL's `options=[16, 32, ...]` -- GuessInt's
            constexpr.int16_t.itype), so this path reads it straight off the
            axis rather than adding a second, parallel elemental-type-string
            table that would need a 'bool' entry for exactly one caller.
        """
        seen = {}
        for s in self.scalars:
            if not isinstance(s.wires_to, ContextHelper):
                continue
            name = s.wires_to.name
            if s.type_ is not None:
                assert s.type_ in _CONTEXT_HELPER_CTYPE, (
                    f'flyc kernel {self.NAME!r}: context_helper scalar {s.arg_name!r} '
                    f'has unrecognised elemental type {s.type_!r}')
                ctype = _CONTEXT_HELPER_CTYPE[s.type_]
            else:
                axis = self.axis_of_arg(s.arg_name)
                assert axis is not None, (
                    f'flyc kernel {self.NAME!r}: context_helper scalar {s.arg_name!r} '
                    f'has no explicit elemental type and is not a known functional '
                    f'axis either (dtype ChoiceVar not supported for context helpers)')
                ctype = axis.repr_typed_choice.itype
            if name in seen:
                assert seen[name] == ctype, (
                    f'flyc kernel {self.NAME!r}: context_helper {name!r} is '
                    f'referenced with inconsistent C types {seen[name]!r} vs '
                    f'{ctype!r}')
                continue
            seen[name] = ctype
            yield name, ctype
