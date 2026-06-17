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

from ..builder import build_kernel
from ..describe import get_kernel_spec
from .axis import assign_godel
from .interface import Interface


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


class _PerfBind:
    """A perf parameter bound to a concrete value, implementing exactly the
    interface KernelSignature consumes (name, value, iteration, get_typed_value,
    settle_unresolved) — without the legacy Bind. Perf params are always plain
    non-conditional constexprs, so settle_unresolved is a no-op."""

    __slots__ = ('_name', '_tc')

    def __init__(self, name, tc):
        self._name = name
        self._tc = tc

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        return self._tc

    def __iter__(self):
        yield self._name, self._tc

    def get_typed_value(self, aname):
        return self._tc

    def settle_unresolved(self, tc_dict):
        pass


class _PerfParamShim:
    """Compat view of a perf-schema field as a legacy performance TemplateParameter
    for translate_dataframe / gen_signatures_for_tuning. Exposes repr_name,
    all_names, choices, create_direct, and get_cfields."""

    __slots__ = ('_pp',)

    def __init__(self, perf_param):
        self._pp = perf_param           # tune.schema.PerfParam

    @property
    def repr_name(self):
        return self._pp.name

    @property
    def all_names(self):
        return [self._pp.name]

    def create_direct(self, value):
        return _PerfBind(self._pp.name, self._pp.tcc(value))

    def get_cfields(self):
        from .cfield import cfield
        tc = self._pp.tcc(0)
        return [cfield(ctype=tc.itype, aname=self._pp.name, index=-1,
                       nbits=tc.NBITS or 0)]


class _AxisParamShim:
    """Compat view of an Axis as a legacy `tp` for the compiled-in feature tables
    (codegen.kernel get_<name>_choices). Exposes the members that code reads:
    repr_name, choices, repr_typed_choice, and `emit_feature_table` (whether this
    axis gets a compiled-in get_<name>_choices() table; false when the axis's
    representative argument is baked to a constexpr by an override)."""

    __slots__ = ('_axis', '_overridden', '_repr_name', '_all_names')

    def __init__(self, axis, overridden_to_constexpr,
                 repr_name=None, all_names=None):
        self._axis = axis
        self._overridden = overridden_to_constexpr
        # Apparel-mapped names (the operator operands) for the C++ surface; default
        # to the axis's own (real) names when not wired. The getter name derives
        # from the representative REAL argument (repr_arg), NOT signature_name (a
        # persisted-artifact label, unrelated to the C++ surface).
        self._repr_name = repr_name if repr_name is not None else axis.repr_arg
        self._all_names = (list(all_names) if all_names is not None
                           else list(axis.arg_names))

    @property
    def repr_name(self):
        # The representative ARGUMENT name (explicit on the choice variable, or
        # the single arg), NOT the dtype-variable name — the C++
        # get_<name>_choices() function name derives from it. Apparel-mapped.
        return self._repr_name

    @property
    def all_names(self):
        return list(self._all_names)

    @property
    def nchoices(self):
        return self._axis.radix

    @property
    def godel_number(self):
        # The axis's godel stride (legacy TP.godel_number == its stride). Trivial
        # (single-choice) axes have no stride assigned; they contribute 0.
        return self._axis.godel_stride or 0

    @property
    def choices(self):
        # legacy code reads tc.infotext off each; expose the underlying TypedChoice
        return [c.tc for c in self._axis.choices]

    @property
    def repr_typed_choice(self):
        return self._axis.choices[0].tc

    @property
    def overridden_to_constexpr(self) -> bool:
        """True when an override bakes this axis's representative argument into a
        constexpr (a derived/dependent value), so it is not a free feature and is
        excluded from the compiled-in feature tables. Note this is an ARGUMENT
        property: a dtype variable shared by several tensors is never baked, even
        if one member tensor (e.g. B) is individually overridden."""
        return self._overridden

    @property
    def emit_feature_table(self) -> bool:
        """Whether this axis gets a compiled-in get_<name>_choices() table. A
        baked (override→constexpr) axis is excluded; stride axes never reach here
        (filtered before the shim is built)."""
        return not self._overridden


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
        # Perf params (shims over the tune schema), in schema order.
        ts = built.tune
        self._perf_params = ([_PerfParamShim(pp) for pp in ts.schema.params]
                             if (ts and ts.schema) else [])
        # AUTOTUNE_KEYS validated against the arguments (legacy contract).
        self._autotune_keys_validated = []
        if ts is not None:
            argset = set(built.arguments)
            for key, sel in ts.binning.items():
                if key in argset:
                    self._autotune_keys_validated.append((key, _binning_class(sel)))

    # --- perf params + tuning metadata (legacy translate_* contract) ---

    @property
    def AUTOTUNE_KEYS_VALIDATED(self):
        return self._autotune_keys_validated

    def gen_performance_params(self):
        yield from self._perf_params

    @property
    def perf_cfields(self):
        ts = self._built.tune
        if ts is None or ts.schema is None:
            return []
        from .cfield import cfield
        out = []
        for pp in ts.schema.params:
            tc = pp.tcc(0)
            out.append(cfield(ctype=tc.itype, aname=pp.name,
                              index=self._arg_index.get(pp.name, -1),
                              nbits=tc.NBITS or 0))
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

    def perf_value(self, perf_param, f):
        """The value of a perf param for functional f: the @dataclass field
        default, then any perf-channel @ati.derives that fires (last wins), e.g.
        PERSISTENT_TYPE -> 2 when CAUSAL_TYPE!=0, NUM_XCDS -> 8 when arch in
        {gfx942,gfx950}. Replaces the legacy PERF_CHOICES default +
        PROGRAMMATIC_PERFS."""
        from .override import VarRef, ValueFn
        value = perf_param.default
        for ov in self._built.perf_overrides:
            if perf_param.name in ov.targets and ov.fires(f):
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
        adapter's AUTOTUNE_KEYS_VALIDATED + perf params."""
        import numpy as np
        from ..ksignature import KernelSignature, COMPILER_OPTIONS
        # Inject perf params that are NOT tuned DB columns: their value is the
        # @dataclass default plus any perf-channel @ati.derives (perf_value), the
        # role the legacy PROGRAMMATIC_PERFS used to fill, before reading
        # tuned_kernel$<name>.
        for pp in self._built.tune.schema.params:
            col = f'tuned_kernel${pp.name}'
            if col not in df.columns:
                df[col] = self.perf_value(pp, f)
        sparse_keys = [f'inputs${key}' for key, _ in self.AUTOTUNE_KEYS_VALIDATED]
        nkeys = len(sparse_keys)
        def sorted_unique_key(key):
            return np.unique(df[key].to_numpy()).tolist()
        sparse_key_possible_values = {key: sorted_unique_key(key) for key in sparse_keys}
        binning_dict = {key: algo(sparse_key_possible_values[spk])
                        for spk, (key, algo) in zip(sparse_keys, self.AUTOTUNE_KEYS_VALIDATED)}
        lut_shape = [f.noptimized_for] + [len(sparse_key_possible_values[key]) for key in sparse_keys]
        lut_tensor = np.full(lut_shape, -1, dtype=np.int32)
        perf_keys = [f'tuned_kernel${meta.repr_name}' for meta in self._perf_params]
        copt_keys = [f'compiler_options${key}' for key in COMPILER_OPTIONS]
        # Deduplicate (perf + copt) rows -> signatures.
        np_sigs, revind = np.unique(df[perf_keys + copt_keys].to_numpy(), axis=0,
                                    return_inverse=True)
        df['$$sig_num'] = revind
        def perf_bind(nprow):
            return [meta.create_direct(value)
                    for meta, value in zip(self._perf_params, nprow)]
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
        from ..ksignature import KernelSignature, DEFAULT_COPT
        lut_tensor = np.zeros([f.noptimized_for, 1], dtype=np.int8)
        defaults = []
        for shim in self._perf_params:
            value = self.perf_value(shim._pp, f)
            defaults.append(shim.create_direct(value))
        sigs = [KernelSignature(f, defaults, DEFAULT_COPT)]
        return lut_tensor, sigs, None

    def gen_signatures_for_tuning(self, f):
        """Yield a KernelSignature per autotune config (the tuning-build path).
        Ported from the legacy KernelDescription."""
        from ..ksignature import KernelSignature, COMPILER_OPTIONS, DEFAULT_COPT
        def gen_perfs(cfg):
            for meta in self._perf_params:
                yield meta.create_direct(cfg.kwargs[meta.repr_name])
        def gen_copts(cfg):
            for copt, defopt in zip(COMPILER_OPTIONS, DEFAULT_COPT):
                yield getattr(cfg, copt, defopt)
        for cfg in self.gen_autotune_configs(f):
            yield KernelSignature(f, list(gen_perfs(cfg)), list(gen_copts(cfg)))

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
                tc = ax.choice_for_arg(0, arg).tc
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

    def list_functional_params(self):
        """Compat view for the generator's compiled-in feature tables
        (get_<name>_choices). Yields one shim per non-stride axis. An axis is a
        free feature unless its representative argument is baked to a constexpr by
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
            yield _AxisParamShim(ax, baked,
                                 repr_name=self.apparel_of(ax.signature_name),
                                 all_names=[self.apparel_of(a) for a in ax.arg_names])

    # --- tuning passthrough ---

    @property
    def tune(self):
        return self._built.tune

    @property
    def is_tunable(self):
        return self._built.tune is not None and self._built.tune.is_tunable

    def is_functional_disabled(self, functional):
        # Driven by the @ati.disable predicates in the kernel description (any
        # firing disables). Only meaningful for tunable kernels — an untunable
        # kernel has no functionals to exclude from autotuning.
        if not self.is_tunable:
            return False
        return any(d.holds(functional) for d in self._built.disables)


def build_kernel_description(kernel, *, family, source_path=None,
                             triton_kernel_name=None, register=True):
    """Build an KernelDescription from a kernel already described via
    ati.describe() / the stacked-@ form.

    Before lowering, @ati.cite gaps are filled from kernels built earlier (the flat
    per-family registry). After building, the kdesc registers itself so later
    kernels can cite it. `register=False` skips registration (test isolation)."""
    from .ops.cite import resolve_cites
    from .. import registry as _registry
    spec = get_kernel_spec(kernel)
    assert spec is not None, (
        f'kernel {getattr(kernel, "__name__", kernel)!r} has no ATI description; '
        f'call ati.describe(...) or use the stacked-@ form first')
    resolve_cites(spec, family=family)        # fill cited gaps before lowering
    built = build_kernel(spec)
    adapter = KernelDescription(built, family=family, source_path=source_path,
                                   triton_kernel_name=triton_kernel_name)
    adapter.kernel_spec = spec      # source KernelSpec (for --sancheck)
    if register:
        _registry.register_kernel(adapter)
    return adapter
