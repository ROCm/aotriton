# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Operator adapter over the ATI IR.

An operator dispatches among interchangeable BACKENDS (the triton metro vs an
aiter ASM kernel, ...). It owns the params struct, built from its DEFAULT backend
(the feature superset): the union (union_params) of the default backend's metro
sub-kernel arguments. The operator's *functional* space is that same default
backend's axes — so Operator reuses the default backend's BuiltKernel for the
func_cfields / gen_functionals / godel surface, and adds the operator-only bits:
the backend list + enums, CALL_OPTIONS_NAME, and optune (OPTUNE_KEYS +
translate_dataframe, reused from the legacy Operator body).
"""

from .kdesc import _binning_class
from .interface import Interface


class Operator(Interface):
    """Operator-compatible facade backed by a default-backend BuiltKernel."""

    CODEGEN_MODULE = 'op'
    TUNE_NAME = 'optune'
    FILE_PFX = 'iface'
    ENUM_PREFIX = 'kOp_'

    def __init__(self, name, *, family, default_kdesc, backends, optune_keys,
                 call_options_name, struct_cfields=None,
                 partially_tuned_functionals=None):
        self.NAME = name
        self.FAMILY = family
        # default_kdesc owns the FUNCTIONAL axes (godel/gen_functionals/axis lookups).
        # By default it also owns the params STRUCT (the feature-superset kernel, e.g.
        # attn_fwd for op_attn_fwd). When the struct is a union across sub-kernels
        # with no single superset (op_attn_bwd), struct_cfields supplies it while the
        # axes still come from default_kdesc (a representative sub-kernel, dk_dv).
        self._default = default_kdesc
        self._struct_cfields = struct_cfields
        self._backends = list(backends)
        self._optune_keys = dict(optune_keys)      # arg name -> BinningSelector
        self.CALL_OPTIONS_NAME = call_options_name
        self._backend_dict = {b.enum_name: b for b in self._backends}
        # Operator-level partial tuning. EXPLICIT — never inherited from default_kdesc:
        # a kernel's @ati.tune.fallback is a per-perf-row downgrade that would fight
        # the operator's backend-selection tuning. Default {} (the legacy operators'
        # effective value).
        self._partially_tuned = dict(partially_tuned_functionals or {})

    # --- identity ---

    # identity surface (unique_path / class_name_base / enum_name (kOp_) /
    # param_class_name / context_class_name / metadata_class_name) comes from the ATI
    # Interface base.

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
        if self._struct_cfields is not None:
            return self._struct_cfields
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
        # The operator's OWN partial-tune, NOT the representative kernel's. Inheriting
        # default_kdesc's @ati.tune.fallback here would wrongly fold a kernel-level
        # perf downgrade into the operator's backend-selection LUT.
        return dict(self._partially_tuned)

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

    def _axes_overrides(self):
        # The operator's functional space is its DEFAULT backend's; meta_object on the
        # yielded functionals stays THIS operator (Interface.gen_functionals sets it),
        # which is what codegen keys on.
        built = self._default._built
        return built.axes, built.overrides

    # --- optune (operator-level: pick the backend) ---

    @property
    def OPTUNE_KEYS(self):
        return {k: _binning_class(sel) for k, sel in self._optune_keys.items()}

    def translate_dataframe(self, f, df):
        """Build the operator's backend-selection LUT from its optune dataframe.
        Ported from the legacy Operator; the LUT stores backend enum names."""
        import numpy as np
        sparse_keys = [f'inputs${key}' for key in self.OPTUNE_KEYS.keys()]
        nkeys = len(sparse_keys)
        def sorted_unique_key(key):
            return np.unique(df[key].to_numpy()).tolist()
        sparse_key_possible_values = {key: sorted_unique_key(key) for key in sparse_keys}
        binning_dict = {key: algo(sparse_key_possible_values[f'inputs${key}'])
                        for key, algo in self.OPTUNE_KEYS.items()}
        lut_shape = [f.noptimized_for] + [len(sparse_key_possible_values[key]) for key in sparse_keys]
        lut_tensor = np.full(lut_shape, -1, dtype=np.int32)
        backend_key = 'op$backend'
        for i, ind_key in enumerate(sparse_keys):
            bucket = sparse_key_possible_values[ind_key]
            def discretization(v, bucket=bucket):
                return bucket.index(v)
            df[f'$$ind_{i}'] = df[ind_key].apply(discretization)
        for i, (target_gpu, fallback_gpu) in enumerate(f.database_gpus):
            if i > 0:
                lut_tensor[i] = lut_tensor[0]
            # Try target GPU first, fall back to fallback_gpu
            df_i = df[df['gpu'] == target_gpu]
            if df_i.empty:
                df_i = df[df['gpu'] == fallback_gpu]
            inds = tuple([df_i[f'$$ind_{j}'] for j in range(nkeys)])
            lut_tensor[i][inds] = df_i[backend_key]
        backend_inds = np.unique(lut_tensor).tolist()
        return lut_tensor, [self._backends[ind].enum_name for ind in backend_inds], binning_dict

    def translate_empty_dataframe(self, f):
        import numpy as np
        lut_tensor = np.zeros([f.noptimized_for, 1], dtype=np.int8)
        return lut_tensor, [self.fallback_backend.enum_name], None

