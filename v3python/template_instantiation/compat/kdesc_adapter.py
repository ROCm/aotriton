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
from ..ir import assign_godel, enumerate_functionals


class _AxisParamShim:
    """Compat view of an Axis as a legacy `tp` for the compiled-in feature tables
    (codegen.kernel get_<name>_choices). Exposes the members that code reads:
    repr_name, choices, repr_typed_choice, and `maybe_conditional` (a legacy-name
    alias; in the new IR it means 'this argument is baked to a constexpr by an
    override' — see overridden_to_constexpr)."""

    __slots__ = ('_axis', '_overridden')

    def __init__(self, axis, overridden_to_constexpr):
        self._axis = axis
        self._overridden = overridden_to_constexpr

    @property
    def repr_name(self):
        # The representative ARGUMENT name (explicit on the choice variable, or
        # the single arg), NOT the dtype-variable name — the C++
        # get_<name>_choices() function name derives from it.
        return self._axis.signature_name

    @property
    def all_names(self):
        return list(self._axis.arg_names)

    @property
    def nchoices(self):
        return self._axis.radix

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

    # Legacy-name alias for the current generator, which still reads
    # `tp.maybe_conditional`. Refactored away in Step 4.2.4.
    @property
    def maybe_conditional(self) -> bool:
        return self._overridden


class AtiFunctional:
    """A functional produced by the adapter. Wraps an IR Functional and carries
    the identity the generator keys on. Codegen-facing signature/name helpers are
    added in Step 4.2."""

    __slots__ = ('_ir', '_kdesc', '_optimized_for')

    def __init__(self, ir_functional, kdesc, optimized_for):
        self._ir = ir_functional
        self._kdesc = kdesc
        self._optimized_for = list(optimized_for)

    @property
    def meta_object(self):
        return self._kdesc

    @property
    def arch(self):
        return self._ir.arch

    @property
    def arch_number(self):
        return self._ir.arch_number

    @property
    def godel_number(self):
        return self._ir.godel_number

    @property
    def choices(self):
        return self._ir.choices

    @property
    def resolved(self):
        return self._ir.resolved

    @property
    def optimized_for(self):
        return self._optimized_for

    @property
    def noptimized_for(self):
        return len(self._optimized_for)

    @property
    def family(self):
        return self._kdesc.FAMILY

    @property
    def FAMILY(self):
        return self._kdesc.FAMILY

    @property
    def name(self):
        return self._kdesc.NAME

    @property
    def NAME(self):
        return self._kdesc.NAME

    @property
    def database_gpus(self):
        from v3python.gpu_targets import AOTRITON_TUNING_DATABASE_REUSE
        return [AOTRITON_TUNING_DATABASE_REUSE.get(g, g) for g in self._optimized_for]

    # --- compact / fallback choices (multi-choice axes only) ---

    @property
    def compact_choices(self) -> dict:
        """repr_name -> resolved TypedChoice for each multi-choice axis (the
        legacy compact_dict)."""
        d = {}
        for ax in self._kdesc.axes_multi:
            name = ax.signature_name
            d[name] = self._ir.resolved[name].tc
        return d

    @property
    def fallback_choices(self) -> dict:
        fb = self._kdesc.partially_tuned_functionals
        out = {}
        for k, tc in self.compact_choices.items():
            out[k] = fb.get(k, tc)
        return out

    # --- triton-compile-signature dicts (resolved per arg) ---

    def build_complete_tc_dict(self):
        """arg_name -> resolved TypedChoice for every argument (post-override)."""
        return {a: c.tc for a, c in self._ir.resolved.items()}

    def build_tc_dict(self):
        """repr_name -> resolved TypedChoice for multi-choice axes."""
        return self.compact_choices

    # --- core signatures (must match legacy bytes) ---

    @property
    def unified_signature(self) -> str:
        parts = []
        for ax in self._kdesc.axes_multi:
            name = ax.signature_name
            tc = self._ir.resolved[name].tc
            parts.append(f'{name}={tc.testrun_entry_signature}')
        return ';'.join(parts)

    @property
    def signature_in_func_name(self) -> str:
        parts = []
        for ax in self._kdesc.axes_multi:
            name = ax.signature_name
            s = str(self._ir.resolved[name].tc.triton_compile_signature)
            s = s.replace('*', '＊').replace(':', '@') \
                 .replace('True', 'T').replace('False', 'F')
            parts.append(s)
        return '_'.join(parts)

    @property
    def compact_signature_noarch(self) -> str:
        return 'F__' + self.signature_in_func_name

    @property
    def human_readable_signature(self) -> str:
        # Best-effort: a C++ comment, not byte-load-bearing (per review). One line
        # per non-stride / non-hidden axis, representative arg + resolved value.
        lines = []
        for ax in self._kdesc.axes_all_ordered:
            if ax.is_stride and ax.kind == 'stride':
                # hidden u64 strides are omitted; baked (constexpr) strides shown
                if not self._ir.resolved[ax.arg_names[0]].is_constexpr:
                    continue
            if ax.kind == 'stride_unit':
                continue
            name = ax.signature_name
            lines.append(f'{name} = {self._ir.resolved[name].tc}')
        return 'Human-readable Signature \n// ' + '\n// '.join(lines)

    # --- file packing paths ---

    @property
    def filepack_ondisk_path(self):
        import hashlib
        from pathlib import Path
        from v3python.gpu_targets import AOTRITON_ARCH_TO_DIRECTORY
        digest = hashlib.sha256(self.unified_signature.encode()).hexdigest()
        return (Path(AOTRITON_ARCH_TO_DIRECTORY[self.arch]) / self._kdesc.FAMILY
                / self._kdesc.NAME / digest)

    @property
    def filepack_inzip_name(self) -> str:
        return self.unified_signature

    @property
    def full_flatzip_path(self):
        from pathlib import Path
        from v3python.gpu_targets import AOTRITON_ARCH_TO_DIRECTORY
        return (Path(AOTRITON_ARCH_TO_DIRECTORY[self.arch]) / self._kdesc.FAMILY
                / (self._kdesc.NAME + '.zip'))

    @property
    def full_filepack_dir(self):
        from pathlib import Path
        from v3python.gpu_targets import AOTRITON_ARCH_TO_DIRECTORY
        return (Path(AOTRITON_ARCH_TO_DIRECTORY[self.arch]) / self._kdesc.FAMILY
                / self._kdesc.NAME)

    @property
    def tunecc_signature(self) -> str:
        return '#F;' + self.unified_signature + f';;arch={self.arch}'

    def __repr__(self):
        return (f'AtiFunctional({self._kdesc.NAME!r}, arch={self.arch!r}, '
                f'godel={self.godel_number})')


class AtiKernelDescription:
    """KernelDescription-compatible facade backed by a BuiltKernel."""

    CODEGEN_MODULE = 'triton'
    TUNE_NAME = 'autotune'
    FILE_PFX = 'shim'
    SHARED_IFACE = None

    def __init__(self, built, *, family, source_path=None, triton_kernel_name=None):
        self._built = built
        self.NAME = built.name
        self.FAMILY = family
        self._source_path = source_path
        self._triton_kernel_name = triton_kernel_name or built.name
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

    # --- identity ---

    @property
    def ARGUMENTS(self):
        return self._built.arguments

    @property
    def godel_number(self):
        return self._godel_number

    @property
    def unique_path(self):
        from pathlib import Path
        return Path(self.FAMILY) / self.CODEGEN_MODULE / self.NAME

    # --- class names (legacy Interface naming rules) ---

    @property
    def class_name_base(self):
        return ''.join(x.capitalize() for x in self.NAME.lower().split('_'))

    @property
    def param_class_name(self):
        return self.class_name_base + 'Params'

    @property
    def context_class_name(self):
        return self.class_name_base + 'Context'

    @property
    def metadata_class_name(self):
        return self.class_name_base + 'Metadata'

    @property
    def enum_name(self):
        return f'kShim_{self.class_name_base}'

    # --- struct cfields (params struct ABI) ---

    @property
    def func_cfields(self):
        """One cfield per non-stride, non-perf argument, in signature order, with
        the axis's representative (rank-specialized) C itype. Overrides never
        change the struct — the ABI is owned by the axis (ati+newbinds §6.2)."""
        from v3python.base.cfield import cfield
        out = []
        for ax in self._axes_all:
            if ax.is_stride:
                continue          # strides are hidden (supplied via TensorView)
            for arg in ax.arg_names:
                tc = ax.choice_for_arg(0, arg).tc
                out.append(cfield(ctype=tc.itype, aname=arg,
                                  index=self._arg_index[arg], nbits=tc.NBITS or 0))
        out.sort(key=lambda cf: cf.index)
        return out

    @property
    def perf_cfields(self):
        """Perf-struct fields from the schema, sorted nbits-descending for compact
        packing (matching legacy)."""
        from v3python.base.cfield import cfield
        ts = self._built.tune
        if ts is None or ts.schema is None:
            return []
        out = []
        for pp in ts.schema.params:
            tc = pp.tcc(0)
            out.append(cfield(ctype=tc.itype, aname=pp.name,
                              index=self._arg_index.get(pp.name, -1),
                              nbits=tc.NBITS or 0))
        out.sort(key=lambda cf: cf.nbits, reverse=True)
        return out

    # --- launch-argument vector (replaces KERNEL_DATA_ARGUMENTS + strides) ---

    def iter_launch_arguments(self):
        """Yield the C++ launch-argument vector entries in signature order
        (see codegen.common.LaunchArg). Data args only: tensors (ptr + each
        stride dim), scalars by-ref; constexpr features and perf are not launch
        data. Strides are emitted right after their tensor, matching legacy."""
        from v3python.codegen.common import LaunchArg
        # Build per-arg access in signature order. Stride axes carry stride_of.
        access = {}
        for ax in self._axes_all:
            if ax.kind == 'tensor':
                for arg in ax.arg_names:
                    access[arg] = ('tensor_ptr',
                                   f'params.{arg}->kparam_data_ptr()')
            elif ax.is_stride:
                tensor_arg, dim = ax.stride_of
                access[ax.arg_names[0]] = (
                    'tensor_stride',
                    f'params.{tensor_arg}->kparam_stride({dim})')
        for arg in self._built.arguments:
            ax = self._axis_of_arg.get(arg)
            if ax is None:
                continue
            if not self._is_launch_data(arg, ax):
                continue
            kind, expr = access.get(arg, ('scalar', f'CAST(&params.{arg})'))
            yield LaunchArg(aname=arg, kind=kind, expr=expr)

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

    # --- enumeration (replaces legacy gen_functionals) ---

    def gen_functionals(self, target_arch):
        for ir_f in enumerate_functionals(self._built.axes, self._built.overrides,
                                          target_arch):
            yield AtiFunctional(ir_f, self,
                                optimized_for=target_arch[ir_f.arch])

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
            baked = ax.signature_name in self._baked_args
            yield _AxisParamShim(ax, baked)

    # --- tuning passthrough ---

    @property
    def tune(self):
        return self._built.tune

    @property
    def is_tunable(self):
        return self._built.tune is not None and self._built.tune.is_tunable

    def is_functional_disabled(self, functional):
        return False


def build_kernel_description(kernel, *, family, source_path=None,
                             triton_kernel_name=None):
    """Build an AtiKernelDescription from a kernel already described via
    ati.describe() / the stacked-@ form."""
    spec = get_kernel_spec(kernel)
    assert spec is not None, (
        f'kernel {getattr(kernel, "__name__", kernel)!r} has no ATI description; '
        f'call ati.describe(...) or use the stacked-@ form first')
    built = build_kernel(spec)
    return AtiKernelDescription(built, family=family, source_path=source_path,
                                triton_kernel_name=triton_kernel_name)
