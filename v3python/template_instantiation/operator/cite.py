# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
@ati.cite resolution — kernel level (executive plan
agent-plans/ati_aux-kernel-xref_exec0.md Step 4).

A citing kernel declares only what is unique to it (wiring, perf, private args) and
@ati.cite("<op>.<metro>.<kernel>")s another kernel for the rest. This pass fills
the citing kernel's gaps at the SPEC level, BEFORE build_kernel lowers it to axes:

  * a GAP ARGUMENT — a signature parameter the citing kernel does not claim locally
    — is given a cloned tensor/scalar spec from the cited kernel, matched by
    APPAREL name. (Stride-bearing tensors must be declared locally per rev0 §5, so
    gap args are scalars / strideless tensors; a gap mapping to a strided cited
    tensor is an error.)
  * a STRING DTYPE reference that names neither a same-kernel @ati.tensor_dtype nor
    a literal ATI type is resolved against the cited kernel's dtype-variable table
    (Step 1 resolution path 3), cloning the named dtype-var locally.

Precedence is `local > cited > error`: anything the citing kernel declares wins; a
gap or dtype-var the cite cannot supply raises AtiDescriptionError.

Kernel-level citing keys on the LAST cite segment (the Triton kernel name) looked
up in the flat per-family registry; the `<op>.<metro>` prefix is validated and kept
for the later metro/operator-level resolution.
"""

from ..builder import AtiDescriptionError, _is_ati_type_string
from ..decorators import TensorSpec, ScalarSpec, ChoiceVar
from .. import registry


def _spec_apparel_names(spec):
    """{apparel_name -> (source_spec, real_arg)} for one cited KernelSpec's tensor
    and scalar bindings. A single binding's apparel is its wires_to or its own arg
    name; multi-arg (list) bindings cannot wire, so apparel == each arg name."""
    out = {}
    for t in spec.tensors:
        if len(t.arg_names) == 1:
            apparel = t.wires_to or t.arg_names[0]
            out[apparel] = (t, t.arg_names[0])
        else:
            for a in t.arg_names:
                out[a] = (t, a)
    for s in spec.scalars:
        if len(s.arg_names) == 1:
            apparel = s.wires_to or s.arg_names[0]
            out[apparel] = (s, s.arg_names[0])
        else:
            for a in s.arg_names:
                out[a] = (s, a)
    return out


def _locally_claimed(spec, param_names):
    claimed = set()
    for t in spec.tensors:
        claimed.update(t.arg_names)
        claimed.update(t.match_strides(param_names))
    for s in spec.scalars:
        claimed.update(s.arg_names)
    if spec.tune is not None and spec.tune.schema is not None:
        claimed.update(pp.name for pp in spec.tune.schema.params)
    return claimed


def _clone_for_gap(name, src_spec, gap_arg):
    """A new single-argument spec for `gap_arg`, copying the cited binding's
    type/dtype/options/rank. Strided cited tensors cannot be cloned into a gap (the
    citing kernel must declare them locally with its own stride glob)."""
    if isinstance(src_spec, TensorSpec):
        if src_spec.strides_pattern is not None:
            raise AtiDescriptionError(
                f"kernel {name!r}: cited gap tensor {gap_arg!r} is stride-bearing; "
                f"declare it locally with its own strides= glob (rev0 §5), not via "
                f"@ati.cite")
        return TensorSpec(gap_arg, src_spec.dtype, strides=None,
                          rank=src_spec.rank, contiguous=None)
    # ScalarSpec: reconstruct from whichever slot the cited binding used.
    if src_spec.dtype is not None:
        return ScalarSpec(gap_arg, src_spec.dtype)
    if src_spec.options is not None:
        return ScalarSpec(gap_arg, options=src_spec.options)
    return ScalarSpec(gap_arg, src_spec.type_)


def _string_dtype_names(spec):
    """The string dtype references on tensor/scalar specs that are NOT literal ATI
    types — candidate dtype-variable names (same-kernel or cited)."""
    names = set()
    for t in spec.tensors:
        d = t.dtype
        if isinstance(d, str) and not _is_ati_type_string(d):
            names.add(d)
    for s in spec.scalars:
        d = s.type_
        if isinstance(d, str) and not _is_ati_type_string(d):
            names.add(d)
    return names


def resolve_cites(spec, *, family, lookup=None):
    """Augment a citing KernelSpec in place from its @ati.cite targets, BEFORE
    build_kernel. `lookup(family, kernel_name) -> cited AtiKernelDescription` (its
    spec is read via `.kernel_spec`); defaults to the flat kernel registry."""
    if not spec.cites:
        return spec
    if lookup is None:
        lookup = registry.get_kernel
    name = getattr(spec.kernel, '__name__', 'kernel')
    param_names = [p.name for p in spec.params]

    # Cited specs (kernel-level: key on the last cite segment).
    cited_specs = []
    for c in spec.cites:
        cited_kdesc = lookup(family, c.kernel_name)
        if cited_kdesc is None:
            raise AtiDescriptionError(
                f"kernel {name!r}: @ati.cite({c.target!r}) names kernel "
                f"{c.kernel_name!r}, which is not a built {family!r} kernel "
                f"(declare/import it before the citing kernel)")
        cs = getattr(cited_kdesc, 'kernel_spec', None)
        if cs is None:
            raise AtiDescriptionError(
                f"kernel {name!r}: cited kernel {c.kernel_name!r} has no ATI spec")
        cited_specs.append(cs)

    # Merge cited apparel/dtype tables (earlier cites win on a tie — first found).
    cited_apparel = {}
    cited_dtype_vars = {}
    for cs in cited_specs:
        for apparel, pair in _spec_apparel_names(cs).items():
            cited_apparel.setdefault(apparel, pair)
        for dv in cs.dtype_vars:
            cited_dtype_vars.setdefault(dv.name, dv)
        # ChoiceVars threaded by object on cited specs also count as named vars.
        for src in (*cs.tensors, *cs.scalars):
            d = getattr(src, 'dtype', None)
            if d is not None and not isinstance(d, str):
                cited_dtype_vars.setdefault(d.name, d)

    # 0) Perf is citeable (rev0 §4.3). When the citing kernel declares NO perf at
    # all, inherit the cited kernel's whole `tune` (schema + configs + binning +
    # ...). A kernel that wants its OWN perf — e.g. a schema-only untunable aux
    # kernel (schema, no configs) — declares it and is left untouched. Done BEFORE
    # the gap-fill so an inherited schema's perf-param names count as claimed.
    if spec.tune is None:
        for cs in cited_specs:
            if cs.tune is not None:
                spec.tune = cs.tune
                break

    # 1) Gap arguments -> cloned specs (apparel == gap name; unclaimed args unwired).
    claimed = _locally_claimed(spec, param_names)
    for arg in param_names:
        if arg in claimed:
            continue
        pair = cited_apparel.get(arg)
        if pair is None:
            raise AtiDescriptionError(
                f"kernel {name!r}: parameter {arg!r} is neither declared locally "
                f"nor supplied by any @ati.cite ({[c.target for c in spec.cites]}); "
                f"declare it or cite a kernel that defines it")
        src_spec, _real = pair
        clone = _clone_for_gap(name, src_spec, arg)
        if isinstance(clone, TensorSpec):
            spec.tensors.append(clone)
        else:
            spec.scalars.append(clone)

    # 2) String dtype refs unresolved locally -> pull the named dtype-var from a
    # cite. (Same-kernel dtype-vars are left for the builder; literals are skipped.)
    local_dv = {dv.name for dv in spec.dtype_vars}
    for dname in _string_dtype_names(spec):
        if dname in local_dv:
            continue
        dv = cited_dtype_vars.get(dname)
        if dv is None:
            raise AtiDescriptionError(
                f"kernel {name!r}: dtype variable {dname!r} is neither an "
                f"@ati.tensor_dtype on this kernel nor reachable through "
                f"@ati.cite ({[c.target for c in spec.cites]})")
        # Clone WITHOUT the cited signature_name: it names a cited argument (e.g.
        # 'Q') absent from this kernel. The builder re-derives signature_name from
        # the citing kernel's own args (the wired/cloned ones using this dtype).
        spec.dtype_vars.append(
            ChoiceVar(dv.name, dv.choices, kind=dv.kind, signature_name=None))
        local_dv.add(dname)

    # 3) @ati.disable is citeable (rev0 §4.5): no local disable -> inherit the
    # cited target's; a local disable REPLACES it. A bare-callable local disable
    # shadowing a cited one (it cannot super()) is a FATAL error unless explicitly
    # affirmed — a warning would be lost in the generator's output, silently
    # dropping the cited correctness exclusion.
    cited_disables = [d for cs in cited_specs for d in cs.disables]
    if not spec.disables:
        spec.disables = list(cited_disables)        # inherit verbatim
    elif cited_disables:
        for d in spec.disables:
            if not d.is_callable_class and not d.override_ack:
                raise AtiDescriptionError(
                    f"kernel {name!r}: a local @ati.disable (a bare "
                    f"lambda/function) would replace the cited disable from "
                    f"{[c.target for c in spec.cites]}, silently dropping the cited "
                    f"correctness exclusion. To EXTEND it, make the disable a "
                    f"callable class that calls super().__call__(f); to override "
                    f"intentionally, pass "
                    f"I_understand_this_overrides_cited_disable=True to "
                    f"ati.disable(...).")
    return spec
