# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
KernelDescription adapter over the ATI IR (executive plan Step 4.1).

Goal: drive the EXISTING code generator from the new Axis/Override/Functional IR,
without going through the legacy TYPE_CHOICES / FEAT_CHOICES / PERF_CHOICES /
ARGUMENTS dictionaries. Those dicts exist only to power the old conditional
type-inference and gen_functionals; the new enumeration (enumerate_functionals)
replaces both, so the adapter sources everything the generator reads directly
from the BuiltKernel.

Per ati+newbinds_rev1.md §6.2 the params-struct field type comes from the axis
(the ABI), and overrides change only per-functional values, never the struct.

This step (4.1) provides the enumeration + identity surface and proves
count/godel parity with the legacy description. The remaining codegen-facing
surface (func_cfields, translate_dataframe, the Functional signature helpers) is
filled in Step 4.2 while running the real generator, where the exact requirements
surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .axis import assign_godel
from .interface import Interface

if TYPE_CHECKING:
    from collections.abc import Iterator
    from .axis import Axis
    from .typed_choice import TypedChoice


def _binning_class(selector):
    """Map an ati.tune.binning selector (.le/.gt/.eq) to the concrete
    aotriton.autotune Binning class the DB-translation expects."""
    from aotriton.autotune import BinningLessOrEqual, BinningExact
    key = getattr(selector, 'key', selector)
    if key == 'le':
        return BinningLessOrEqual
    if key == 'eq':
        return BinningExact
    raise NotImplementedError(
        f'binning selector {key!r} has no concrete class yet '
        f'(only le/eq are implemented; gt is parity-only)')


@dataclass(slots=True)
class TemplateParam:
    """The per-kernel view of one functional Axis for the compiled-in feature tables
    (codegen.kernel get_<name>_choices). Named after the legacy `TemplateParameter`
    (TP) it descends from — the codegen loops still call its instances `tp`. It is NOT
    a re-shaping of Axis — it pairs the axis with this kernel's wiring/override state
    that Axis cannot hold:
      * `repr_name` / `all_names` — the APPAREL-mapped names (operator operands) for the
        C++ surface, which depend on the kernel's wires_to map, not on the axis.
      * `overridden_to_constexpr` — whether an override bakes the axis's representative
        REAL argument to a constexpr (an argument property; a dtype variable shared by
        several tensors is never baked even if one member tensor is). A baked axis is
        excluded from the feature tables.
    Intrinsic axis data (radix / godel_stride / choices / repr_typed_choice) is read
    straight off the axis."""

    axis: Axis
    repr_name: str                  # apparel-mapped getter name (get_<repr_name>_choices)
    all_names: list[str]            # apparel-mapped member arg names
    overridden_to_constexpr: bool

    @property
    def radix(self) -> int:
        return self.axis.radix

    @property
    def godel_stride(self) -> int:
        # Trivial (single-choice) axes have no stride assigned; they contribute 0.
        return self.axis.godel_stride or 0

    @property
    def choices(self):
        return list(self.axis.choices)

    @property
    def repr_typed_choice(self) -> TypedChoice:
        return self.axis.repr_typed_choice

    @property
    def emit_feature_table(self) -> bool:
        """Whether this axis gets a compiled-in get_<name>_choices() table. A baked
        (override->constexpr) axis is excluded; stride axes never reach here."""
        return not self.overridden_to_constexpr


class KernelDescription(Interface):
    """KernelDescription-compatible facade backed by a BuiltKernel."""

    CODEGEN_MODULE = 'triton'
    TUNE_NAME = 'autotune'
    FILE_PFX = 'shim'
    ENUM_PREFIX = 'kShim_'

    def __init__(self, built, *, family, source_path=None, triton_kernel_name=None):
        self._built = built
        self.NAME = built.name
        self.FAMILY = family
        self._source_path = source_path
        self._triton_kernel_name = triton_kernel_name or built.name
        # Argument wiring (real -> apparel). A kernel's REAL arguments are its own
        # Triton-signature parameters (e.g. debug's `R`); an APPAREL argument is the
        # operator-side operand it is dressed as at the params-struct / launch site
        # (e.g. `encoded_softmax`). Declared per-argument via `wires_to=` (rev0 §4.3)
        # and stored HERE, on the kdesc, because a kernel can be launched bare (the
        # ME backend calls debug directly, with no metro) and its shim must still
        # address the apparel field. The apparel value is a plain operand name for
        # now; the representation is kept opaque so it can later carry a tuple of
        # operator params or an expression (`kernel(X=params.A+params.B)`).
        self._arg_wiring = dict(getattr(built, 'wiring', None) or {})
        # Canonical (anchor-ordered) axes; assign godel strides to multi-choice.
        self._axes_all = sorted(built.axes, key=lambda a: a.anchor)
        self._axes_multi = [a for a in self._axes_all if not a.is_trivial]
        assign_godel(self._axes_multi)
        self._godel_number = 1
        for a in self._axes_multi:
            self._godel_number *= a.radix
        self._arg_index = {a: i for i, a in enumerate(built.arguments)}
        # arg name -> owning axis (excludes nothing; every arg belongs to one axis)
        self._axis_of_arg = {arg: ax for ax in self._axes_all for arg in ax.arg_names}
        # Arguments an override bakes into a constexpr. This is an ARGUMENT-level
        # property (B, dropout_p, ...), not an axis/type-variable one: a dtype
        # variable is never baked even when a member tensor is overridden.
        self._baked_args = {t for ov in built.overrides for t in ov.targets}
        # The synthesized perf struct (the tune schema), in field order. A canonical
        # empty struct stands in when the kernel has no @ati.tune.schema, so the perf
        # paths need no None-guards.
        from ..specs.tune import EMPTY_PERF_STRUCT
        ts = built.tune
        self._perf_struct = ts.schema if (ts and ts.schema) else EMPTY_PERF_STRUCT
        # Autotune keys from @ati.tune.binning, paired with their Binning class.
        # Keys that are not kernel arguments are skipped (a binning key only matters
        # when it names a real argument the LUT is indexed by).
        self._autotune_keys = []
        if ts is not None:
            argset = set(built.arguments)
            for key, sel in ts.binning.items():
                if key in argset:
                    self._autotune_keys.append((key, _binning_class(sel)))

    # --- perf params + tuning metadata (legacy translate_* contract) ---

    @property
    def autotune_keys(self):
        return self._autotune_keys

    def gen_performance_params(self):
        """The perf-param NAMES, in schema order (consumed by codegen_perf_assignment)."""
        yield from self._perf_struct.param_names()

    @property
    def perf_cfields(self):
        # The struct owns the cfield layout (type + nbits); we supply the kernel arg
        # index and keep the nbits-descending sort (codegen struct-packing policy).
        out = self._perf_struct.cfields(index_of=lambda n: self._arg_index.get(n, -1))
        out.sort(key=lambda cf: cf.nbits, reverse=True)
        return out

    # --- DB / LUT translation (reuses legacy bodies via exposed accessors) ---

    @property
    def triton_source_path(self):
        from pathlib import Path
        return Path(self._source_path)

    @property
    def triton_kernel_name(self):
        return self._triton_kernel_name

    @property
    def is_tunable(self):
        return self._built.tune is not None and self._built.tune.is_tunable

    def perf_value(self, name, f):
        """The value of perf param `name` for functional f: the @dataclass field
        default, then any perf-channel @ati.derives that fires (last wins), e.g.
        PERSISTENT_TYPE -> 2 when CAUSAL_TYPE!=0, NUM_XCDS -> 8 when arch in
        {gfx942,gfx950}. Replaces the legacy PERF_CHOICES default +
        PROGRAMMATIC_PERFS."""
        from .override import VarRef, ValueFn
        value = self._perf_struct.default_value(name)
        for ov in self._built.perf_overrides:
            if name in ov.targets and ov.fires(f):
                if isinstance(ov.value, VarRef):
                    value = f.choice[ov.value.var_name].triton_compile_signature
                elif isinstance(ov.value, ValueFn):
                    value = ov.value(f)
                else:
                    value = ov.value
        return value

    def translate_dataframe(self, f, df):
        """Build the (lut_tensor, signatures, binning) triple for functional f from
        its tuning dataframe. Ported from the legacy KernelDescription; reads the
        adapter's autotune_keys + perf params."""
        import numpy as np
        from .ksignature import KernelSignature, COMPILER_OPTIONS
        # Inject perf params that are NOT tuned DB columns: their value is the
        # @dataclass default plus any perf-channel @ati.derives (perf_value), the
        # role the legacy PROGRAMMATIC_PERFS used to fill, before reading
        # tuned_kernel$<name>.
        for name in self._perf_struct.param_names():
            col = f'tuned_kernel${name}'
            if col not in df.columns:
                df[col] = self.perf_value(name, f)
        sparse_keys = [f'inputs${key}' for key, _ in self.autotune_keys]
        nkeys = len(sparse_keys)
        def sorted_unique_key(key):
            return np.unique(df[key].to_numpy()).tolist()
        sparse_key_possible_values = {key: sorted_unique_key(key) for key in sparse_keys}
        binning_dict = {key: algo(sparse_key_possible_values[spk])
                        for spk, (key, algo) in zip(sparse_keys, self.autotune_keys)}
        lut_shape = [f.noptimized_for] + [len(sparse_key_possible_values[key]) for key in sparse_keys]
        lut_tensor = np.full(lut_shape, -1, dtype=np.int32)
        perf_names = self._perf_struct.param_names()
        perf_keys = [f'tuned_kernel${n}' for n in perf_names]
        copt_keys = [f'compiler_options${key}' for key in COMPILER_OPTIONS]
        # Deduplicate (perf + copt) rows -> signatures.
        np_sigs, revind = np.unique(df[perf_keys + copt_keys].to_numpy(), axis=0,
                                    return_inverse=True)
        df['$$sig_num'] = revind
        def perf_bind(nprow):
            # one perf bind row: a struct instance with each field a settled choice.
            return self._perf_struct(**{n: self._perf_struct.choice_for(n, value)
                                        for n, value in zip(perf_names, nprow)})
        nperfs = len(perf_keys)
        def create_sig(nprow):
            return KernelSignature(f, perf_bind(nprow), nprow[nperfs:].tolist())
        sigs = [create_sig(nprow) for nprow in np_sigs]
        # Bucket autotune indices and fill the LUT (df's gpu column comes from the DB,
        # so iterate database_gpus).
        for i, ind_key in enumerate(sparse_keys):
            bucket = sparse_key_possible_values[ind_key]
            def discretization(v, bucket=bucket):
                return bucket.index(v)
            df[f'$$ind_{i}'] = df[ind_key].apply(discretization)
        for i, gpu in enumerate(f.database_gpus):
            if i > 0:
                lut_tensor[i] = lut_tensor[0]
            df_i = df[df['gpu'] == gpu]
            inds = tuple([df_i[f'$$ind_{j}'] for j in range(nkeys)])
            lut_tensor[i][inds] = df_i['$$sig_num']
        # Downcast the LUT dtype (int8 usually suffices).
        nsigs = len(sigs)
        for dtype in [np.int8, np.int16, np.int32]:
            if nsigs < np.iinfo(dtype).max:
                break
        lut_tensor = lut_tensor.astype(dtype)
        return lut_tensor, sigs, binning_dict

    def translate_empty_dataframe(self, f):
        import numpy as np
        from .ksignature import KernelSignature, DEFAULT_COPT
        lut_tensor = np.zeros([f.noptimized_for, 1], dtype=np.int8)
        defaults = self._perf_struct(
            **{n: self._perf_struct.choice_for(n, self.perf_value(n, f))
               for n in self._perf_struct.param_names()})
        sigs = [KernelSignature(f, defaults, DEFAULT_COPT)]
        return lut_tensor, sigs, None

    def gen_signatures_for_tuning(self, f):
        """Yield a KernelSignature per autotune config (the tuning-build path).
        Ported from the legacy KernelDescription."""
        from .ksignature import KernelSignature, COMPILER_OPTIONS, DEFAULT_COPT
        def perf_bind(cfg):
            # one perf bind row from an autotune config: struct instance of settled choices.
            return self._perf_struct(
                **{n: self._perf_struct.choice_for(n, cfg.kwargs[n])
                   for n in self._perf_struct.param_names()})
        def gen_copts(cfg):
            for copt, defopt in zip(COMPILER_OPTIONS, DEFAULT_COPT):
                yield getattr(cfg, copt, defopt)
        for cfg in self.gen_autotune_configs(f):
            yield KernelSignature(f, perf_bind(cfg), list(gen_copts(cfg)))

    def gen_autotune_configs(self, f):
        cfg = self._built.tune.configs
        return cfg(f)

    # Flash-family LUT shape constants (used by the reused sancheck body).
    # TODO: move to a per-family adapter mixin when a second family is ported.
    LUT_FULL_SEQLEN_Q = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    LUT_FULL_SEQLEN_K = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    LUT_FULL_SEQLEN_NAVI = [16, 32, 64, 128, 256, 512, 1024, 2048]

    def sancheck_lut_tensor(self, f, lut_tensor):
        from aotriton.codegen.parser import load_family_aot
        FlashKernel = load_family_aot('flash')._common.FlashKernel
        return FlashKernel.sancheck_lut_tensor(self, f, lut_tensor)

    def _gen_missing_entries(self, *args, **kwargs):
        from aotriton.codegen.parser import load_family_aot
        FlashKernel = load_family_aot('flash')._common.FlashKernel
        return FlashKernel._gen_missing_entries(self, *args, **kwargs)

    # --- axis views (used by AtiFunctional signatures) ---

    @property
    def axes_all_ordered(self):
        """All axes (incl. strides), canonical anchor order."""
        return self._axes_all

    @property
    def axes_multi(self):
        """Multi-choice axes in anchor order — the compact-signature dimensions.
        Each axis's representative name (arg_names[0], the first kernel argument
        of the group, NOT the dtype variable) is the legacy repr_name."""
        return self._axes_multi

    @property
    def partially_tuned_functionals(self) -> dict:
        ts = self._built.tune
        return dict(ts.fallback) if ts is not None else {}

    def axis_of_arg(self, aname):
        return self._axis_of_arg.get(aname)

    def axis_by_var(self, var_name):
        for ax in self._axes_all:
            if ax.var_name == var_name:
                return ax
        return None

    def override_for(self, aname):
        """The override that rewrites `aname`, or None. (Last one wins, matching
        the declared-order apply in enumerate_functionals.)"""
        found = None
        for ov in self._built.overrides:
            if aname in ov.targets:
                found = ov
        return found

    # --- argument wiring (real -> apparel) ---

    def set_arg_wiring(self, wiring: dict):
        """Install the real->apparel wiring for this kernel (real argument name ->
        operator operand it is dressed as). Called when the kernel is placed into an
        operator/metro; the wiring then applies to EVERY launch of this kernel,
        including a bare (metro-less) launch by an alternative backend."""
        self._arg_wiring = dict(wiring)

    def apparel_of(self, real_arg: str) -> str:
        """The apparel argument `real_arg` is dressed as (its operator operand), or
        `real_arg` itself when it is not wired. This is the name the params struct /
        launch site addresses."""
        return self._arg_wiring.get(real_arg, real_arg)

    def real_of(self, apparel_arg: str) -> str:
        """Inverse of apparel_of: the REAL Triton argument behind an apparel name,
        or the name itself when not wired. Used to map an apparel name emitted on
        the codegen surface back to the IR's real-keyed tables (resolved[], etc.).
        Wiring is 1:1 today, so the inverse is well-defined."""
        for real, apparel in self._arg_wiring.items():
            if apparel == apparel_arg:
                return real
        return apparel_arg

    # --- identity ---

    @property
    def ARGUMENTS(self):
        return self._built.arguments

    @property
    def godel_number(self):
        return self._godel_number

    # identity surface (unique_path / class_name_base / context_class_name /
    # metadata_class_name / enum_name) comes from the ATI Interface base. Only the
    # SHARED_IFACE-borrow of param_class_name is kernel-specific:
    @property
    def param_class_name(self):
        # When SHARED_IFACE is set (the kernel borrows an operator's param struct,
        # like legacy attn_fwd -> OpAttnFwdParams), the struct name comes from the
        # shared interface, not this kernel.
        if self.SHARED_IFACE is not None:
            return self.SHARED_IFACE.param_class_name
        return self.class_name_base + 'Params'

    # --- struct cfields (params struct ABI) ---

    @property
    def func_cfields(self):
        """One cfield per non-stride, non-perf argument, in signature order, with
        the axis's representative (rank-specialized) C itype. Overrides never
        change the struct — the ABI is owned by the axis (ati+newbinds §6.2)."""
        from .cfield import cfield
        out = []
        for ax in self._axes_all:
            if ax.is_stride:
                continue          # strides are hidden (supplied via TensorView)
            for arg in ax.arg_names:
                tc = ax.choice_for_arg(0, arg)
                # The struct field is the APPAREL name (the operator operand); the
                # IR (index, axis) stays keyed on the real argument.
                out.append(cfield(ctype=tc.itype, aname=self.apparel_of(arg),
                                  index=self._arg_index[arg], nbits=tc.NBITS or 0))
        out.sort(key=lambda cf: cf.index)
        return out

    # --- launch-argument vector (replaces KERNEL_DATA_ARGUMENTS + strides) ---

    def iter_launch_arguments(self):
        """Yield the C++ launch-argument vector entries in signature order
        (see codegen.common.LaunchArg). Data args only: tensors (ptr + each
        stride dim), scalars by-ref; constexpr features and perf are not launch
        data. Strides are emitted right after their tensor, matching legacy."""
        from aotriton.codegen.common import LaunchArg
        # Build per-arg access in signature order. Stride axes carry stride_of. All
        # `params.<X>` access goes through the APPAREL name (the operator operand /
        # struct field); the LaunchArg.aname is the apparel name too (it drives the
        # struct-field reference and the comment). The IR lookup in pp_arg_doc maps
        # the apparel name back to the real argument.
        access = {}
        for ax in self._axes_all:
            if ax.kind == 'tensor':
                for arg in ax.arg_names:
                    a = self.apparel_of(arg)
                    access[arg] = ('tensor_ptr',
                                   f'params.{a}->kparam_data_ptr()')
            elif ax.is_stride:
                tensor_arg, dim = ax.stride_of
                # The stride references its tensor's APPAREL name, but the stride
                # argument itself keeps its real name (matches golden:
                # params.encoded_softmax->kparam_stride(0), // stride_rz).
                access[ax.arg_names[0]] = (
                    'tensor_stride',
                    f'params.{self.apparel_of(tensor_arg)}->kparam_stride({dim})')
        for arg in self._built.arguments:
            ax = self._axis_of_arg.get(arg)
            if ax is None:
                continue
            if not self._is_launch_data(arg, ax):
                continue
            # Strides keep their real arg name (the comment shows stride_rz, not an
            # apparel name); tensors/scalars surface as their apparel name.
            emit_name = arg if ax.is_stride else self.apparel_of(arg)
            kind, expr = access.get(arg, ('scalar', f'CAST(&params.{self.apparel_of(arg)})'))
            yield LaunchArg(aname=emit_name, kind=kind, expr=expr)

    def _is_launch_data(self, arg, ax):
        """A launch data argument: a tensor, a non-unit stride, or a runtime
        scalar. Excludes constexpr feature scalars, unit strides, and perf."""
        if ax.kind == 'tensor':
            return True
        if ax.kind == 'stride':
            return True
        if ax.kind == 'stride_unit':
            return False
        # scalar axis: runtime if its choice is not a constexpr
        return not ax.choices[0].is_constexpr

    # gen_functionals is inherited from ir.Interface (its default _axes_overrides
    # reads self._built.{axes,overrides}); each yielded Functional has
    # meta_object = this kdesc.

    # --- compiled-in feature tables ---

    def list_functional_params(self) -> 'Iterator[TemplateParam]':
        """The template-param view for the generator's compiled-in feature tables
        (get_<name>_choices). Yields one TemplateParam per non-stride axis. An axis
        is a free feature unless its representative argument is baked to a constexpr by
        an override. A grouped dtype variable like T_io stays a feature even if a
        member tensor (B) is individually baked — baking is an argument property,
        not a type-variable one."""
        for ax in self._axes_all:
            if ax.is_stride:
                continue
            # Baking is a REAL-argument property (override targets are real args).
            baked = ax.repr_arg in self._baked_args
            # The C++ getter name (get_<name>_choices) is the axis's SIGNATURE_NAME
            # — the persisted/human-readable label (e.g. T_io -> 'Q'), not the
            # repr_arg used for value lookup. The member list is the apparel of the
            # real arg names (operator operands).
            yield TemplateParam(
                axis=ax, overridden_to_constexpr=baked,
                repr_name=self.apparel_of(ax.signature_name),
                all_names=[self.apparel_of(a) for a in ax.arg_names])

    # --- tuning passthrough ---

    @property
    def tune(self):
        return self._built.tune

    def is_functional_disabled(self, functional):
        # Driven by the @ati.disable predicates in the kernel description (any
        # firing disables). Only meaningful for tunable kernels — an untunable
        # kernel has no functionals to exclude from autotuning.
        if not self.is_tunable:
            return False
        return any(d.holds(functional) for d in self._built.disables)
