# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Operator adapter over the ATI IR (executive plan Step 5.2b).

An operator dispatches among interchangeable BACKENDS (the triton metro vs an
aiter ASM kernel, ...). It owns the params struct, built from its DEFAULT backend
(the feature superset): the union (union_params) of the default backend's metro
sub-kernel arguments. The operator's *functional* space is that same default
backend's axes — so AtiOperator reuses the default backend's BuiltKernel for the
func_cfields / gen_functionals / godel surface, and adds the operator-only bits:
the backend list + enums, CALL_OPTIONS_NAME, and optune (OPTUNE_KEYS +
translate_dataframe, reused from the legacy Operator body).
"""

from .kdesc_adapter import AtiFunctional, _binning_class


class AtiOperator:
    """Operator-compatible facade backed by a default-backend BuiltKernel."""

    CODEGEN_MODULE = 'op'
    TUNE_NAME = 'optune'
    FILE_PFX = 'iface'
    SHARED_IFACE = None
    HEADER_EXTRA_INCLUDES = []
    SOURCE_EXTRA_INCLUDES = []

    def __init__(self, name, *, family, default_kdesc, backends, optune_keys,
                 call_options_name):
        self.NAME = name
        self.FAMILY = family
        self._default = default_kdesc      # AtiKernelDescription (param-struct owner)
        self._backends = list(backends)
        self._optune_keys = dict(optune_keys)      # arg name -> BinningSelector
        self.CALL_OPTIONS_NAME = call_options_name
        self._backend_dict = {b.enum_name: b for b in self._backends}

    # --- identity ---

    @property
    def unique_path(self):
        from pathlib import Path
        return Path(self.FAMILY) / self.CODEGEN_MODULE / self.NAME

    @staticmethod
    def _name_to_base(name):
        return ''.join(x.capitalize() for x in name.lower().split('_'))

    @property
    def class_name_base(self):
        return self._name_to_base(self.NAME)

    @property
    def enum_name(self):
        return f'kOp_{self.class_name_base}'

    @property
    def param_class_name(self):
        return self.class_name_base + 'Params'

    @property
    def context_class_name(self):
        return self.class_name_base + 'Context'

    @property
    def metadata_class_name(self):
        return self.class_name_base + 'Metadata'

    # --- backends ---

    def list_backends(self):
        return self._backends

    def get_backend(self, name):
        """The backend whose NAME is `name`, for @ati.cite resolution
        (ops[op].get_backend(metro)). Raises KeyError if absent."""
        for b in self._backends:
            if getattr(b, 'NAME', None) == name:
                return b
        raise KeyError(
            f'operator {self.NAME!r} has no backend named {name!r}; '
            f'backends: {[getattr(b, "NAME", None) for b in self._backends]}')

    @property
    def fallback_backend(self):
        return self._backends[0]

    @property
    def nbackends(self):
        return len(self._backends)

    # --- functional space (reuses the default backend's axes) ---

    @property
    def godel_number(self):
        return self._default.godel_number

    @property
    def func_cfields(self):
        return self._default.func_cfields

    def list_functional_params(self):
        return self._default.list_functional_params()

    # Axis views the AtiFunctional reads (delegated to the default backend, which
    # owns the functional axes).
    @property
    def axes_multi(self):
        return self._default.axes_multi

    @property
    def axes_all_ordered(self):
        return self._default.axes_all_ordered

    @property
    def partially_tuned_functionals(self):
        return self._default.partially_tuned_functionals

    def axis_of_arg(self, aname):
        return self._default.axis_of_arg(aname)

    def axis_by_var(self, var_name):
        return self._default.axis_by_var(var_name)

    def override_for(self, aname):
        return self._default.override_for(aname)

    def apparel_of(self, real_arg):
        # The operator's params come from its default backend; reuse its wiring so
        # AtiFunctional (shared between kdesc and operator meta_object) resolves the
        # same apparel names regardless of which it is keyed on.
        return self._default.apparel_of(real_arg)

    def real_of(self, apparel_arg):
        return self._default.real_of(apparel_arg)

    def gen_functionals(self, target_arch):
        # Same functional product as the default backend, but meta_object is THIS
        # operator (codegen keys on functional.meta_object).
        from ..ir import enumerate_functionals
        built = self._default._built
        for ir_f in enumerate_functionals(built.axes, built.overrides, target_arch):
            yield AtiFunctional(ir_f, self, optimized_for=target_arch[ir_f.arch])

    # --- optune (operator-level: pick the backend) ---

    @property
    def OPTUNE_KEYS(self):
        return {k: _binning_class(sel) for k, sel in self._optune_keys.items()}

    def translate_dataframe(self, f, df):
        from aotriton.op.operator import Operator
        return Operator.translate_dataframe(self, f, df)

    def translate_empty_dataframe(self, f):
        from aotriton.op.operator import Operator
        return Operator.translate_empty_dataframe(self, f)

    def fallback_compact_dict(self, compact_dict):
        # No PARTIALLY_TUNED_FUNCTIONALS at the operator level (matches legacy).
        return dict(compact_dict)
