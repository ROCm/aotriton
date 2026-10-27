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
translate_tensor, ported from the legacy Operator body).
"""

from .triton.kdesc import _binning_class
from .interface import Interface


class Operator(Interface):
    """Operator-compatible facade backed by a default-backend BuiltKernel."""

    CODEGEN_MODULE = 'op'
    TUNE_NAME = 'optune'
    FILE_PFX = 'iface'
    ENUM_PREFIX = 'kOp_'

    def __init__(self, name, *, family, default_kdesc, backends, optune_keys,
                 call_options_name, struct_cfields=None,
                 partially_tuned_functionals=None, backend_names=None):
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
        # The names @ati.backend DECLARED, index-aligned with _backends. This is
        # the user-facing vocabulary -- 'triton', 'aiter', 'flyc' -- and is what
        # the published backend constants are built from. Deliberately NOT
        # b.enum_name: that carries the kMetro_/kShim_/kSlimAffine_ prefix, which
        # says how the backend is assembled. That is a fact about the generator,
        # and it has no business in an interface a caller selects through.
        self._backend_names = list(backend_names) if backend_names else [
            b.NAME for b in self._backends]
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
    def backend_names(self):
        """The @ati.backend-declared names, index-aligned with list_backends()."""
        return list(self._backend_names)

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

    def translate_tensor(self, f, view):
        """Build the operator's backend-selection LUT from its tuning tensors
        (database.csvdb.TuningView): one slab per gpu f is optimized for, each
        cell a backend precedence. The LUT stores backend indices."""
        import numpy as np
        key_names = list(self.OPTUNE_KEYS.keys())
        nkeys = len(key_names)
        # One row per tuned cell: (gpu index, tuning key values, best backend,
        # runner-up). The precedence names backends by their @ati.backend name;
        # the runner-up repeats the best when there is none.
        rows = []
        for i, slab in enumerate(view.slabs):
            if slab is None:
                continue
            assert slab.tuning_keys == key_names, \
                f'{slab.path}: tuning keys {slab.tuning_keys}, expected {key_names}'
            for index in zip(*np.nonzero(slab.cells)):
                ranked = [self.backend_names.index(name) for name in view.backends(slab.cells[index])]
                keys = [slab.ladders[d][index[d]] for d in range(nkeys)]
                rows.append((i, keys, ranked[0], ranked[min(1, len(ranked) - 1)]))
        sparse_key_possible_values = [sorted({keys[d] for _, keys, _, _ in rows}) for d in range(nkeys)]
        binning_dict = {key: algo(sparse_key_possible_values[d])
                        for d, (key, algo) in enumerate(self.OPTUNE_KEYS.items())}
        lut_shape = [f.noptimized_for] + [len(v) for v in sparse_key_possible_values]
        lut_tensor = np.full(lut_shape, -1, dtype=np.int32)
        if True:  # FIXME: Flyc duct tape
            FLYC_ARCHS = ('gfx950', 'gfx1201')
        buckets = [{v: j for j, v in enumerate(values)} for values in sparse_key_possible_values]
        for i in range(f.noptimized_for):
            if i > 0:
                lut_tensor[i] = lut_tensor[0]
            for gpu, keys, best, runner_up in rows:
                if gpu != i:
                    continue
                chosen = best
                if True:  # FIXME: Flyc duct tape
                    # An arch can share another arch's tensors through a symlink
                    # (e.g. gfx1200 -> gfx1201), so gfx1200 inherits gfx1201's
                    # picks -- flyc among them, which it has no images for. That
                    # lands as a valid-looking index the runtime will not rescue:
                    # op.cc only substitutes fallback_backend for a NEGATIVE
                    # entry, and only retries on hipErrorPeerAccessUnsupported,
                    # while flyc answers hipErrorNotSupported. Swap flyc for the
                    # runner-up here.
                    names = self.backend_names
                    if f.arch not in FLYC_ARCHS and 'flyc' in names:
                        flyc_ind = names.index('flyc')
                        if chosen == flyc_ind:
                            chosen = runner_up
                        # The runner-up REPEATS the best when flyc was the only
                        # backend to clear the accuracy gate -- see
                        # compute_best_results.py's `alt[0] if alt else won`
                        # -- so the substitution above can leave flyc standing
                        # and hand gfx1200 the very index this exists to
                        # remove. Backend 0 is triton / triton_split, which
                        # every arch has images for, so it is the one answer
                        # that is always launchable.
                        if chosen == flyc_ind:
                            chosen = 0
                lut_tensor[(i, *[buckets[d][k] for d, k in enumerate(keys)])] = chosen
        backend_inds = np.unique(lut_tensor).tolist()
        return lut_tensor, [self._backends[ind].enum_name for ind in backend_inds], binning_dict

    def translate_empty(self, f):
        import numpy as np
        lut_tensor = np.zeros([f.noptimized_for, 1], dtype=np.int8)
        return lut_tensor, [self.fallback_backend.enum_name], None

