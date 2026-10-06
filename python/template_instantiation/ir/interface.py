# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI Interface base — the shared codegen-identity surface.

Every ATI description the generator consumes (kernel / operator / metro / affine /
conditional) carries the same identity & C++-naming surface: FAMILY / NAME /
CODEGEN_MODULE, the `unique_path` selective key, and the `<Base>{Params,Context,
Metadata}` / enum-name rules. This base factors that surface so each concrete class
declares only its distinct behavior (functional/tuning surface, launcher methods,
co_gen, branch refs) plus a few class attrs.

NOTE: this is NOT the former legacy base Interface. That one also
carried the pre-ATI functional-generation machinery (TYPE_CHOICES / FEAT_CHOICES /
ARGUMENTS dicts, `__init__`/`_late_init`) which the ATI enumeration replaced. The ATI
base is identity-only — no functional `__init__`.
"""

import itertools
from abc import ABC, abstractmethod
from pathlib import Path

from .axis import assign_godel, godel_of
from .functional import Functional, _resolve


class FallbackAxisRead(RuntimeError):
    """Tuning candidates depend on an @ati.tune.fallback axis. One tuning entry
    serves every value of such an axis, so they must not."""


def _fallback_choice(axis, value):
    """The choice of `axis` that an @ati.tune.fallback value names."""
    for tc in axis.choices:
        sig = tc.triton_compile_signature
        if sig == value and isinstance(sig, bool) == isinstance(value, bool):
            return tc
    raise ValueError(f'fallback {axis.signature_name}={value!r} is not one of '
                     f'{[tc.triton_compile_signature for tc in axis.choices]}')


class Interface(ABC):
    # --- identity / codegen wiring (subclasses set these) -------------------
    FAMILY = None              # e.g. 'flash'
    NAME = None                # e.g. 'attn_fwd' (often per-instance, set in __init__)
    CODEGEN_MODULE = None      # 'triton' | 'op' | 'affine'  (selective-path segment)
    TUNE_NAME = None           # 'autotune' | 'optune' | None
    FILE_PFX = 'iface'         # 'shim' (kernel) | 'iface' (operator) | 'affine'
    ENUM_PREFIX = None         # 'kShim_' | 'kOp_' | 'kMetro_' | 'kSlimAffine_' | 'kConditional_'
    SHARED_IFACE = None        # the operator whose params struct this borrows (or None)
    HEADER_EXTRA_INCLUDES = []
    SOURCE_EXTRA_INCLUDES = []

    # --- C++ name helpers --------------------------------------------------

    @staticmethod
    def _name_to_base(name):
        return ''.join(x.capitalize() for x in name.lower().split('_'))

    @property
    def class_name_base(self):
        return self._name_to_base(self.NAME)

    @property
    def enum_name(self):
        return f'{self.ENUM_PREFIX}{self.class_name_base}'

    @property
    def param_class_name(self):
        # Default: this description owns its params struct. Subclasses that BORROW an
        # operator's struct (the triton kernel, the affine kernel) override this to
        # SHARED_IFACE.param_class_name.
        return self.class_name_base + 'Params'

    @property
    def context_class_name(self):
        return self.class_name_base + 'Context'

    @property
    def metadata_class_name(self):
        return self.class_name_base + 'Metadata'

    # --- selective-build key ----------------------------------------------

    @property
    def unique_path(self) -> Path:
        return Path(self.FAMILY) / self.CODEGEN_MODULE / self.NAME

    # --- tuning fallback (operator-level partial tune; default identity) ---

    @property
    def partially_tuned_functionals(self) -> dict:
        return {}

    def tuning_representative(self, f: Functional) -> Functional:
        """The functional whose tuning entry also serves `f`: `f` with every
        partially-tuned (fallback) axis pinned to its fallback value.

        Functionals that differ only in fallback axes share one database row and
        one tuning entry, so anything that defines that entry's candidates must
        be computed from this representative, never from `f` itself."""
        axes_all, axes_multi, overrides = self._ordered_axes()
        fallback = self.partially_tuned_functionals
        by_label = {ax.signature_name: ax for ax in axes_all}
        unknown = set(fallback) - set(by_label)
        if unknown:
            raise ValueError(f'{self.NAME}: @ati.tune.fallback names no axis: {sorted(unknown)}')
        picked = dict(f.choice)
        for label, value in fallback.items():
            picked[by_label[label].var_name] = _fallback_choice(by_label[label], value)
        if picked == f.choice:
            return f
        godel = godel_of(axes_multi, [ax.choices.index(picked[ax.var_name]) for ax in axes_multi])
        return self._functional(axes_all, overrides, picked, godel,
                                f.arch, f.arch_number, f.optimized_for)

    # --- functional enumeration (classical: the Interface yields its functionals) -

    def _axes_overrides(self):
        """(axes, overrides) this Interface enumerates over. Default reads the
        lowered BuiltKernel (`self._built`); the operator overrides this to source
        its DEFAULT backend's axes while keeping meta_object = the operator."""
        return self._built.axes, self._built.overrides

    def _ordered_axes(self):
        """(axes_all, axes_multi, overrides): every axis in canonical anchor
        order, and the multi-choice ones (the godel digits, strides assigned)."""
        axes, overrides = self._axes_overrides()
        axes_all = sorted(axes, key=lambda a: a.anchor)
        axes_multi = [a for a in axes_all if not a.is_trivial]
        assign_godel(axes_multi)
        return axes_all, axes_multi, overrides

    def _functional(self, axes_all, overrides, picked, godel, arch, arch_number, gpus):
        return Functional(meta_object=self, arch=arch,
                          arch_number=arch_number, godel_number=godel,
                          choice=picked,
                          resolved=_resolve(axes_all, overrides, picked, arch),
                          optimized_for=gpus)

    def gen_functionals(self, target_arch):
        """Yield every Functional of this Interface (the classical enumeration:
        product over the multi-choice axes (godel), fan each variable's choice onto
        its arguments, apply overrides in declared order). `target_arch` is an
        ordered {arch -> gpus}; arch_number is its enumeration index."""
        axes_all, axes_multi, overrides = self._ordered_axes()
        trivial_pick = {a.var_name: a.choices[0] for a in axes_all if a.is_trivial}
        for arch_number, (arch, gpus) in enumerate(target_arch.items()):
            for sel in itertools.product(*[range(a.radix) for a in axes_multi]):
                godel = godel_of(axes_multi, sel)
                picked = dict(trivial_pick)
                picked.update({a.var_name: a.choices[i]
                               for i, a in zip(sel, axes_multi)})
                yield self._functional(axes_all, overrides, picked, godel,
                                       arch, arch_number, gpus)

    # --- functional surface (the contract the code generator consumes) -----
    #
    # Every Interface must answer these for the generator's params-struct ABI and
    # compiled-in feature tables. KernelDescription / Operator / AffineKernel compute
    # them from their axes; metro launchers (MetroKernel / ConditionalKernel) own no
    # functional space and answer with the empty surface.

    @property
    @abstractmethod
    def func_cfields(self):
        """The params-struct C fields (list[cfield]), one per non-stride/non-perf
        functional argument in signature order."""
        raise NotImplementedError

    @abstractmethod
    def list_functional_params(self):
        """The per-axis TemplateParam views for the compiled-in feature tables
        (get_<name>_choices). Iterable; empty for launchers with no functional space."""
        raise NotImplementedError

    # --- tuning surface (safe defaults; tunable kernels override) -----------

    is_tunable = False           # only KernelDescription with a tune schema is tunable

    @property
    def perf_cfields(self):
        """The perf-struct C fields (list[cfield]); empty unless the kernel has a
        @ati.tune.schema."""
        return []

    def is_functional_disabled(self, functional) -> bool:
        """Whether `functional` is excluded from generation (@ati.disable). Default
        False — only kernels/affines carrying a disable predicate override this."""
        return False

    # --- selective execution (overridden by kernels that support it) -------

    def iter_kernel_slot_names(self):
        yield from ()

    # --- context-helper-wired functional axes ------------------------------

    def context_helper_for_functional(self, aname: str) -> str | None:
        """The context-helper member-function name that stands in for functional
        axis `aname` when computing `godel_number()`, instead of the default
        `params->aname` read -- or None, meaning "read it off params like every
        other axis".

        None is the base default and is correct for everyone but flyc:
        `ir/flyc/kdesc.py`'s KernelDescription is the only override, for its own
        head-dimension-ladder wiring. The Triton KernelDescription inherits this
        unchanged -- it has no context-helper mechanism at all -- so
        `codegen_godel_number_calculation` (codegen/interface.py) keeps its
        single `args.` path there rather than needing a second one."""
        return None
