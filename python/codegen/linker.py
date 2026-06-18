# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI build — Pass 2: LINK.

Takes the Pass-1 CompiledFamily shells (parser.py) and resolves their relocations
into the final IR tree the code generators consume: it resolves @ati.cite gaps,
builds every kernel / metro / affine / operator, derives each operator's params
struct + default kernel (A1/A3 — neither is declared on the @ati.operator surface),
infers SHARED_IFACE, and verifies every kernel argument is resolved (A4: the linker's
"undefined symbol" diagnostic).

Cycles are ALLOWED and must TERMINATE (A4): a kernel's parsed ARGUMENTS are its
"header"/extern surface (known from Pass 1), and @ati.scalar/.tensor are the
"implementation". Cite inheritance fills only GAP arguments (in the header, no local
implementation) and NEVER overwrites a locally-declared implementation.

The family-scoped name lookup is a plain local dict (kernel def-name -> built kdesc);
relocation state lives on the shells, not in a side symbol table.
"""

import sys
from graphlib import TopologicalSorter, CycleError


class FamilyArtifacts:
    """The linked output for one family: the lists the codegen consumers iterate."""

    def __init__(self, family, kernels, operators, affine_kernels):
        self.family = family
        self.kernels = kernels
        self.operators = operators
        self.affine_kernels = affine_kernels


def _metro_subkernel_names(compiled, op_name, metro_name):
    """The concrete sub-kernel def-names of a whole-metro cite target, or None if the
    metro is not in this family. The op_name segment is informational (the metro name
    is the backend enum name, unique within the family)."""
    metro_shell = compiled.metros.get(metro_name)
    if metro_shell is None:
        return None
    return list(metro_shell.subkernel_names)


def _kernel_build_order(compiled):
    """Order kernel def-names so every @ati.cite target is built before its citer.
    A 3-segment cite depends on the named sub-kernel; a 2-segment (whole-metro) cite
    depends on the metro's OTHER sub-kernels — never on the citer itself, so a
    sub-kernel citing the metro that contains it (a true cycle in implementation
    terms) is acyclic in HEADER terms and terminates: the citer reads the others'
    argument surface, not its own. A genuine dependency cycle (A != B both ways) is a
    compiler error."""
    ts = TopologicalSorter()
    for name, shell in compiled.kernels.items():
        deps = set()
        for c in shell.cites:
            if c.kernel_name is not None:
                if c.kernel_name in compiled.kernels:
                    deps.add(c.kernel_name)
            else:
                for sub in (_metro_subkernel_names(
                        compiled, c.op_name, c.metro_name) or []):
                    if sub != name:                  # exclude the citer (header path)
                        deps.add(sub)
        ts.add(name, *deps)
    try:
        return list(ts.static_order())
    except CycleError as e:
        raise SystemExit(f'ATI linker: @ati.cite dependency cycle: {e.args[1]}')


def _clone_spec(spec):
    """A shallow copy of a KernelSpec with FRESH mutable lists, so the linker's
    resolve_cites (which appends gap tensors/scalars/overrides/dtype_vars and may set
    tune/disables) never mutates the module-level passive spec — making linking
    idempotent (the spec is the source of truth; the linker builds from a copy)."""
    from aotriton.template_instantiation.describe import KernelSpec
    clone = KernelSpec.__new__(KernelSpec)
    clone.kernel = spec.kernel
    clone.params = spec.params              # immutable signature; shared is fine
    clone.tensors = list(spec.tensors)
    clone.scalars = list(spec.scalars)
    clone.overrides = list(spec.overrides)
    clone.tune = spec.tune                  # replaced wholesale by resolve_cites if None
    clone.disables = list(spec.disables)
    clone.source_path = spec.source_path
    clone.dtype_vars = list(spec.dtype_vars)
    clone.cites = list(spec.cites)
    return clone


def _build_kernels(compiled):
    """Resolve cites + build every kernel shell into a KernelDescription, in cite
    dependency order. Returns {def-name -> KernelDescription}."""
    from aotriton.template_instantiation.ir.ops.cite import resolve_cites
    from aotriton.template_instantiation.builder import build_kernel
    from aotriton.template_instantiation.ir.kdesc import KernelDescription

    built = {}
    specs = {}                # def-name -> the cloned, cite-resolved spec

    def lookup(_family, kernel_name):
        kd = built.get(kernel_name)
        return kd

    def metro_lookup(_family, _op_name, metro_name, _citer):
        """A whole-metro cite donor set: the metro's sub-kernels EXCEPT the citer
        (header path — the citer inherits the others' argument surface, never its
        own), in @ati.hints.union_precedence priority order (key kernels first) so a
        colliding operand's binding comes from the canonical key kernel, not whichever
        sub-kernel happens to come first in call order. Each donor is its cite-resolved
        clone (already built, since build order puts donors before the citer)."""
        metro_shell = compiled.metros.get(metro_name)
        if metro_shell is None:
            return None
        donors = []
        for sub in metro_shell.donor_order():
            if sub == _citer:
                continue
            donor_spec = specs.get(sub)        # the cite-resolved clone KernelSpec
            if donor_spec is not None:
                donors.append(donor_spec)
        return donors

    for name in _kernel_build_order(compiled):
        shell = compiled.kernels[name]
        spec = _clone_spec(shell.spec)
        resolve_cites(spec, family=compiled.family, lookup=lookup,
                      metro_lookup=lambda f, o, m, _n=name: metro_lookup(f, o, m, _n))
        specs[name] = spec
        bk = build_kernel(spec)
        kdesc = KernelDescription(bk, family=compiled.family,
                                  source_path=shell.source_path,
                                  triton_kernel_name=name)
        kdesc.kernel_spec = spec       # the cite-resolved clone (for whole-metro cites)
        built[name] = kdesc
    return built


def _build_affines(compiled):
    """Build every AffineKernel from its parsed AffineDecl."""
    from aotriton.template_instantiation.ir.affine import AffineKernel
    out = {}
    for name, decl in compiled.affines.items():
        out[name] = AffineKernel(
            name=name, family=compiled.family, co_dir=decl.co_dir,
            cookie=decl.cookie, headers=decl.headers,
            supported_arch=decl.supported_arch, choice_filters=decl.choice_filters,
            shared_operator_name=decl.shared_operator_name,
            supplied_specs=decl.supplied_specs, disable=decl.disable,
            supplies_after=decl.supplies_after, supplies_before=decl.supplies_before)
    return out


def _build_metros(compiled, built_kernels):
    """Build every MetroKernel, binding its sub-kernels by name to built kdescs."""
    from aotriton.template_instantiation.ir.metro import build_metro
    out = {}
    for name, shell in compiled.metros.items():
        out[name] = build_metro(shell.plan, built_kernels, name,
                                family=compiled.family)
    return out


def _backend_objs(op_shell, built_kernels, metros, affines):
    """Resolve an operator shell's index-sorted backend refs to built IR objects."""
    objs = []
    for index, kind, name in op_shell.backend_refs:
        if kind == 'metro':
            objs.append(metros[name])
        elif kind == 'kernel':
            objs.append(built_kernels[name])
        else:
            objs.append(affines[name])
    return objs


def _iter_concrete(backend):
    """Concrete sub-kernels of a backend (metro -> its sub-kernels; else itself)."""
    if hasattr(backend, 'iter_subkernels'):
        yield from backend.iter_subkernels()
    else:
        yield backend


def _derive_default_kdesc(backends):
    """The operator's functional-axes owner (A1): the first TUNABLE concrete sub-kernel
    of the default (index-0) backend — fwd: attn_fwd; bwd: the metro's first key kernel
    dk_dv. Falls back to the first sub-kernel when none is tunable."""
    subs = list(_iter_concrete(backends[0]))
    return next((s for s in subs if getattr(s, 'is_tunable', False)), subs[0])


def _derive_struct_cfields(backends, default_kdesc):
    """Derive the operator params struct (A3 — no struct_cfields on the surface): the
    order-preserving UNION over all backends' concrete sub-kernels' functional fields,
    with affine supplied_operands (DQ_ACC) anchored via their union_order. When every
    contributor's fields are a SUBSET of the default kernel's (the fwd case — the
    metro's key kernel IS the feature superset), the union equals that kernel's struct,
    so return it directly (the merge over a superset can reorder shared fields and is
    unnecessary). Otherwise merge (the bwd case — DQ/DB/Out/DQ_ACC live only on some
    backends)."""
    from aotriton.template_instantiation.ir.operator import build_merged_struct_cfields
    contributors = [s for b in backends for s in _iter_concrete(b)]
    default_fields = {cf.aname for cf in default_kdesc.func_cfields}
    all_fields = set()
    for s in contributors:
        all_fields |= {cf.aname for cf in s.func_cfields}
    if all_fields <= default_fields:
        return None        # superset: the Operator uses default_kdesc.func_cfields
    return build_merged_struct_cfields(contributors)


def _build_operators(compiled, built_kernels, metros, affines):
    """Build every Operator with derived default_kdesc + struct (A1/A3)."""
    from aotriton.template_instantiation.ir.operator import Operator
    out = {}
    for name in compiled.op_order:
        shell = compiled.operators[name]
        decl = shell.decl
        indices = [i for i, _k, _n in shell.backend_refs]
        assert indices == list(range(len(indices))), (
            f'operator {name!r} backend indices must be dense 0..n-1, got {indices}')
        backends = _backend_objs(shell, built_kernels, metros, affines)
        default_kdesc = _derive_default_kdesc(backends)
        struct_cfields = _derive_struct_cfields(backends, default_kdesc)
        out[name] = Operator(
            name, family=compiled.family, default_kdesc=default_kdesc,
            struct_cfields=struct_cfields, backends=backends,
            optune_keys=dict(decl.binning),
            call_options_name=decl.opspec.call_options_name,
            partially_tuned_functionals=dict(decl.fallback))
    return out


def _check_unresolved_arguments(built_kernels):
    """A4 — the linker's "undefined symbol" check: after cite resolution + build, every
    argument in every kernel's header (parsed ARGUMENTS) must have an implementation
    (an axis or a baked override). build_kernel already raises on a truly undefined
    argument; here we assert the post-build invariant and emit a compiler-style error
    + non-zero exit if any kernel left an argument with no axis and no override."""
    errors = []
    for name, kdesc in built_kernels.items():
        built = kdesc._built
        covered = set()
        for ax in built.axes:
            covered.update(ax.arg_names)
        for ov in (*built.overrides, *built.perf_overrides):
            covered.update(ov.targets)
        # Perf-schema params are implemented by the tune schema (autotune configs),
        # not by a functional axis or an override.
        if built.tune is not None and built.tune.schema is not None:
            covered.update(pp.name for pp in built.tune.schema.params)
        for arg in built.arguments:
            if arg not in covered:
                errors.append((name, arg))
    if errors:
        print('ATI linker: unresolved argument(s) (no implementation after cite '
              'resolution):', file=sys.stderr)
        for name, arg in errors:
            print(f'  kernel {name!r}: argument {arg!r} is undefined', file=sys.stderr)
        raise SystemExit(1)


class Linker:
    """Pass 2 — LINK. Owns a Parser (and thus the `modules/` root); resolves each
    family's Pass-1 shells into the final IR tree the code generators consume.

    `module_dir` is `<root_dir>/modules`, given explicitly by the generator
    (--root_dir) — no cwd/__file__ guessing. Construct one Linker per generation."""

    def __init__(self, module_dir):
        from .parser import Parser
        self.parser = Parser(module_dir)

    def link_family(self, aot_module, family):
        """Pass 2 for one family: compile (Pass 1) then resolve + build the final
        tree. Returns FamilyArtifacts(kernels, operators, affine_kernels)."""
        from aotriton.template_instantiation.ir.ops.infer import infer_shared_iface

        compiled = self.parser.compile_family(aot_module, family)
        built_kernels = _build_kernels(compiled)
        _check_unresolved_arguments(built_kernels)
        affines = _build_affines(compiled)
        metros = _build_metros(compiled, built_kernels)
        operators = _build_operators(compiled, built_kernels, metros, affines)

        op_list = [operators[n] for n in compiled.op_order]
        infer_shared_iface(op_list)

        return FamilyArtifacts(
            family,
            kernels=list(built_kernels.values()),
            operators=op_list,
            affine_kernels=list(affines.values()))

    def link_all_families(self):
        """Discover every family under module_dir, link each, and concatenate the
        artifacts the generator consumes. Returns (kernels, operators,
        affine_kernels)."""
        kernels, operators, affine_kernels = [], [], []
        for family in self.parser.discover_families():
            aot = self.parser.load_family_aot(family)
            arts = self.link_family(aot, family)
            kernels.extend(arts.kernels)
            operators.extend(arts.operators)
            affine_kernels.extend(arts.affine_kernels)
        return kernels, operators, affine_kernels
