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
gap or dtype-var the cite cannot supply raises DescriptionError.

Kernel-level citing keys on the LAST cite segment (the Triton kernel name) looked
up in the flat per-family registry; the `<op>.<metro>` prefix is validated and kept
for the later metro/operator-level resolution.
"""

import fnmatch

from ...builder import DescriptionError, _is_ati_type_string
from ...decorators import TensorSpec, ScalarSpec, ChoiceVar
from ... import registry


def _spec_apparel_names(spec):
    """{apparel_name -> (source_spec, real_arg, cited_param_names)} for one cited
    KernelSpec's tensor and scalar bindings. A single binding's apparel is its
    wires_to or its own arg name; multi-arg (list) bindings cannot wire, so
    apparel == each arg name. cited_param_names is the cited kernel's signature
    (needed to resolve a cited tensor's stride glob to exact names)."""
    cited_params = [p.name for p in spec.params]
    out = {}
    for t in spec.tensors:
        if len(t.arg_names) == 1:
            apparel = t.wires_to or t.arg_names[0]
            out[apparel] = (t, t.arg_names[0], cited_params)
        else:
            for a in t.arg_names:
                out[a] = (t, a, cited_params)
    for s in spec.scalars:
        if len(s.arg_names) == 1:
            apparel = s.wires_to or s.arg_names[0]
            out[apparel] = (s, s.arg_names[0], cited_params)
        else:
            for a in s.arg_names:
                out[a] = (s, a, cited_params)
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


def _clone_for_gap(name, src_spec, gap_args, cited_params):
    """A new spec for `gap_args` (>=1 arguments that came from the SAME cited
    binding), copying the cited binding's type/dtype/options/rank. Passing the whole
    group preserves the cited grouping — several gap args sharing one cited spec
    collapse to one axis / one feature-table getter, matching the cited kernel.

    A STRIDED cited tensor is cloned with its EXACT resolved stride names + the
    resolved contiguous stride name (rev0 §5 extension): the cited glob is resolved
    against the CITED kernel's params here, and those exact names are recorded on
    the clone (resolved_strides) so the citing builder binds the same arguments
    verbatim — never re-globbing, which could match a different set if the two
    kernels name their strides differently."""
    arg = gap_args if len(gap_args) > 1 else gap_args[0]
    if isinstance(src_spec, TensorSpec):
        resolved = src_spec.match_strides(cited_params) or None
        contiguous = (src_spec.resolve_contiguous(cited_params)
                      if resolved is not None else None)
        return TensorSpec(arg, src_spec.dtype, rank=src_spec.rank,
                          contiguous=contiguous, resolved_strides=resolved)
    # ScalarSpec: reconstruct from whichever slot the cited binding used.
    if src_spec.dtype is not None:
        return ScalarSpec(arg, src_spec.dtype)
    if src_spec.options is not None:
        return ScalarSpec(arg, options=src_spec.options)
    return ScalarSpec(arg, src_spec.type_)


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


def _kernel_spec_of(kdesc, name, target):
    cs = getattr(kdesc, 'kernel_spec', None)
    if cs is None:
        raise DescriptionError(
            f"kernel {name!r}: cite target {target!r} resolves to a kernel with "
            f"no ATI spec (a legacy kernel cannot be cited)")
    return cs


def _resolve_one_cite(c, family, name, lookup, metro_lookup=None):
    """Resolve a single @ati.cite target to a LIST of cited KernelSpecs.

    Resolution order (rev0 §4.4):
      0. metro level via `metro_lookup` (the linker's injection) — for a whole-metro
         (2-segment) target, return every sub-kernel's KernelSpec straight from the
         Pass-1 shells, WITHOUT requiring a built operator. This is the header/extern
         path that lets a sub-kernel cite the metro that contains it (a true cycle):
         the cited sub-kernels' argument surface is known from Pass 1, so the citer's
         gaps resolve even while its own implementation is still being linked. A
         3-segment target falls through to the per-kernel `lookup`.
      1. operator/metro level via the `ops` registry — `ops[op].get_backend(metro)`,
         then `.get_kernel(kernel)` for a 3-segment target, or ALL of the metro's
         sub-kernels for a 2-segment (whole-metro) target;
      2. kernel level via the flat kernel registry (`lookup`) on the last segment —
         the Step-4 path, used when the operator is not (yet) built.
    A whole-metro (2-segment) target REQUIRES the ops path or metro_lookup."""
    if metro_lookup is not None and c.kernel_name is None:
        cited = metro_lookup(family, c.op_name, c.metro_name)
        if cited is not None:
            return list(cited)
    op = registry.get_op(family, c.op_name)
    if op is not None:
        try:
            metro = op.get_backend(c.metro_name)
        except KeyError as e:
            raise DescriptionError(
                f"kernel {name!r}: @ati.cite({c.target!r}): {e}")
        if c.kernel_name is None:
            # whole metro -> every sub-kernel's spec (the merged interface)
            return [_kernel_spec_of(k, name, c.target)
                    for k in metro.iter_subkernels()]
        try:
            sub = metro.get_kernel(c.kernel_name)
        except KeyError as e:
            raise DescriptionError(
                f"kernel {name!r}: @ati.cite({c.target!r}): {e}")
        return [_kernel_spec_of(sub, name, c.target)]
    # Operator not built; fall back to the flat kernel registry (kernel-level).
    if c.kernel_name is None:
        raise DescriptionError(
            f"kernel {name!r}: @ati.cite({c.target!r}) is a whole-metro cite but "
            f"operator {c.op_name!r} is not a built {family!r} operator "
            f"(build/import it before the citing kernel)")
    kdesc = lookup(family, c.kernel_name)
    if kdesc is None:
        raise DescriptionError(
            f"kernel {name!r}: @ati.cite({c.target!r}) names kernel "
            f"{c.kernel_name!r}, which is neither a built {family!r} operator's "
            f"sub-kernel nor a built kernel (declare/import it before the citing "
            f"kernel)")
    return [_kernel_spec_of(kdesc, name, c.target)]


def resolve_cites(spec, *, family, lookup=None, metro_lookup=None):
    """Augment a citing KernelSpec in place from its @ati.cite targets, BEFORE
    build_kernel. Each target resolves (via metro_lookup, the ops registry, else the
    flat kernel registry) to one or more cited KernelSpecs whose practices fill the
    citing kernel's gaps. `lookup(family, kernel_name)` overrides the flat-registry
    lookup; `metro_lookup(family, op_name, metro_name)` resolves a whole-metro cite to
    its sub-kernels' specs directly (the linker's header/extern path)."""
    if not spec.cites:
        return spec
    if lookup is None:
        lookup = registry.get_kernel
    name = getattr(spec.kernel, '__name__', 'kernel')
    param_names = [p.name for p in spec.params]

    cited_specs = []
    for c in spec.cites:
        cited_specs.extend(
            _resolve_one_cite(c, family, name, lookup, metro_lookup=metro_lookup))

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

    # 1) Gap arguments -> cloned specs. Gaps that came from the SAME cited binding
    # are cloned together (one axis / one feature-table getter), PRESERVING the
    # cited grouping — e.g. attn_fwd groups Num_head_q/Max_seqlen_q/Max_seqlen_k
    # under one scalar spec, so debug inherits them as one group, not three. Group
    # membership is keyed on the cited source spec (identity); order follows the
    # citing kernel's signature.
    claimed = _locally_claimed(spec, param_names)
    paramset = set(param_names)
    # A strided cited tensor cloned into a gap also covers the cited tensor's
    # RESOLVED stride names (resolved against the CITED kernel's params) that also
    # appear in the citing signature — those are not standalone gaps; the clone
    # binds them. Using resolved names (not a re-glob) keeps it robust when the two
    # kernels name their strides differently.
    stride_covered = set()
    for apparel, (src_spec, _real, cited_params) in cited_apparel.items():
        if (isinstance(src_spec, TensorSpec) and src_spec.strides_pattern
                and apparel not in claimed):
            for sname in src_spec.match_strides(cited_params):
                if sname in paramset and sname != apparel:
                    stride_covered.add(sname)
    gap_groups = []          # list of (src_spec, [gap_arg,...], cited_params)
    group_index = {}         # id(src_spec) -> index into gap_groups
    for arg in param_names:
        if arg in claimed or arg in stride_covered:
            continue
        pair = cited_apparel.get(arg)
        if pair is None:
            raise DescriptionError(
                f"kernel {name!r}: parameter {arg!r} is neither declared locally "
                f"nor supplied by any @ati.cite ({[c.target for c in spec.cites]}); "
                f"declare it or cite a kernel that defines it")
        src_spec, _real, cited_params = pair
        key = id(src_spec)
        if key not in group_index:
            group_index[key] = len(gap_groups)
            gap_groups.append((src_spec, [], cited_params))
        gap_groups[group_index[key]][1].append(arg)
    for src_spec, gap_args, cited_params in gap_groups:
        clone = _clone_for_gap(name, src_spec, gap_args, cited_params)
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
            raise DescriptionError(
                f"kernel {name!r}: dtype variable {dname!r} is neither an "
                f"@ati.tensor_dtype on this kernel nor reachable through "
                f"@ati.cite ({[c.target for c in spec.cites]})")
        # Keep the cited signature_name when that argument also exists in the citing
        # kernel (so the persisted aks2/DB key matches, e.g. 'Q' shared by fwd/bwd);
        # otherwise drop it and let the builder re-derive from the citing kernel's
        # own arguments.
        sig = dv.signature_name if dv.signature_name in paramset else None
        spec.dtype_vars.append(
            ChoiceVar(dv.name, dv.choices, kind=dv.kind, signature_name=sig))
        local_dv.add(dname)

    # 2b) Inherit cited OVERRIDES (@ati.derives) for shared operands. A cited
    # kernel's conditional degradation (dropout/philox/Window -> constexpr 0 when
    # off, B -> 0 when bias off) travels with the operand: a citing kernel that
    # shares the operand should degrade it the same way (otherwise it would emit a
    # live feature table for an argument that is conditionally baked). Inherit a
    # cited override only when ALL its targets exist in the citing kernel AND the
    # citing kernel does not already override any of them (local wins). Perf-channel
    # derives are NOT inherited here (they ride on the cited tune via step 0).
    locally_overridden = {t for ov in spec.overrides for t in ov.targets}
    perf_param_names = set()
    if spec.tune is not None and spec.tune.schema is not None:
        perf_param_names = {pp.name for pp in spec.tune.schema.params}
    seen_inherited = set()
    for cs in cited_specs:
        for ov in cs.overrides:
            if any(t in perf_param_names for t in ov.targets):
                continue                      # perf derive -> via the tune, not here
            if not all(t in paramset for t in ov.targets):
                continue                      # operand absent from the citing kernel
            if any(t in locally_overridden for t in ov.targets):
                continue                      # local override wins
            # Dedup across cited kernels: dk_dv and dq carry the same
            # dropout/Window degradation, so inherit each target-set once.
            if ov.targets in seen_inherited:
                continue
            seen_inherited.add(ov.targets)
            spec.overrides.append(ov)

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
                raise DescriptionError(
                    f"kernel {name!r}: a local @ati.disable (a bare "
                    f"lambda/function) would replace the cited disable from "
                    f"{[c.target for c in spec.cites]}, silently dropping the cited "
                    f"correctness exclusion. To EXTEND it, make the disable a "
                    f"callable class that calls super().__call__(f); to override "
                    f"intentionally, pass "
                    f"I_understand_this_overrides_cited_disable=True to "
                    f"ati.disable(...).")
    return spec
