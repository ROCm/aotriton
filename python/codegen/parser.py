# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI build — Pass 1: COMPILE (parser).

The ATI decorators are PASSIVE: they only RECORD specs onto the `def` objects
(`fn.__ati__` / `fn.__ati_operator__` / `fn.__ati_affine__` / `fn.__ati_metro__`).
This module parses those passive records into lightweight IR SHELLS the linker
(Pass 2) builds + resolves, and also owns family discovery / module loading
(absorbed from the old `aotriton.rules` aggregator).

Compile vs link (the compiler framing the executive plan adopts):
  * COMPILE (here) parses each passive declaration and records its cross-object
    references as deferred "relocations" stored DIRECTLY ON the shell (the metro shell
    keeps its sub-kernel name list; the operator shell keeps its backend refs; the
    kernel shell keeps its un-cite-resolved spec). No build_kernel / build_operator /
    build_metro is run here — a citing kernel still has gap arguments at this point.
  * LINK (linker.py) resolves the relocations (cites, backend binding, SHARED_IFACE),
    builds every IR object, and derives each operator's struct + default kernel.

Family discovery (interim protocol, pre-`--module_dir`): each kernel family lives in
`<repo>/modules/<family>/aot`, a python package whose `__init__` exposes `operators`
(the roots). Each `aot` package is loaded by EXPLICIT file path under a synthetic
top-level name so its relative imports resolve without a `<family>` namespace pkg and
its `kernel/` sources keep importing each other by bare name.
"""

import sys
import importlib.util
from pathlib import Path


def load_family_aot(family):
    """Fetch an already-loaded family's `aot` package from the import cache.

    The Parser loads each family by path under the synthetic name
    `_aotriton_modules_<family>_aot`; this free function returns that cached module
    (or None) for the few consumers that need the loaded package but do not have a
    Parser handle — e.g. ir/kdesc.py's flash sancheck back-edge, which runs after the
    family was loaded during linking. It never loads (no modules_dir): the family
    must already be loaded by a Parser."""
    return sys.modules.get(f'_aotriton_modules_{family}_aot')


# --- Pass-1 shells (relocations stored on the shell) -------------------------
#
# Each shell is the minimal representation of ONE description node that is knowable
# at parse time, before the linker resolves cross-object references.  They are the
# compiler analogue of object-file symbol-table entries: they record the exported
# name and the list of unresolved external references ("relocations").
#
# WHY NOT merge shells into the IR classes (KernelDescription / MetroKernel / Operator)?
# Because the IR constructors require fully-resolved inputs that do not exist yet:
#   • KernelDescription needs a lowered BuiltKernel (cite gaps filled, Axis IR built,
#     Godel strides assigned). Cite resolution requires every cite target to already be
#     built — which requires a global topological sort over the whole family.
#   • MetroKernel needs live KernelDescription objects for its sub-kernels (the plan
#     only carries names-as-strings at parse time).
#   • Operator needs its default_kdesc and struct_cfields derived from all fully-built
#     backends; there is no partial result until every backend is materialized.
#
# A two-phase IR init would work, but it leaves IR objects in a broken intermediate
# state — every consumer would need guards. The shell/IR split makes incompleteness
# unrepresentable: an IR object that exists is always fully constructed.
#
# NOTE: there is no KernelDecl alongside OperatorDecl / AffineDecl. That is because
# KernelSpec *is* the kernel's passive "object file" — it plays the same role. The
# difference is that KernelSpec must be cloned and mutated during linking (cite
# resolution appends to its tensors/scalars/overrides on a per-link copy), so it
# cannot be a frozen record the way OperatorDecl and AffineDecl are. OperatorDecl /
# AffineDecl contain no unresolved cross-kernel references; their contents are fully
# known at parse time and the linker reads them verbatim.

class KernelShell:
    """A parsed triton-kernel description: its un-cite-resolved KernelSpec + identity.
    NAME / triton_kernel_name / the family-scoped key are all the def __name__ (== the
    Triton kernel symbol name, since @ati.source loads that symbol); source_path rides
    on the spec. The linker resolves @ati.cite gaps then builds the KernelDescription."""

    __slots__ = ('name', 'spec', 'source_path')

    def __init__(self, name, spec, source_path):
        self.name = name
        self.spec = spec
        self.source_path = source_path

    @property
    def cites(self):
        return self.spec.cites


class MetroShell:
    """A parsed @ati.metro_kernel backend: its MetroPlan + the backend enum-name. The
    sub-kernel NAMES (plan Call strings) are the relocation the linker binds.

    `precedence` is the optional @ati.hints.union_precedence order (highest priority
    first) used when sub-kernel bindings collide — for a whole-metro @ati.cite gap
    donor and the operator params-struct union. When absent it is the call order."""

    __slots__ = ('name', 'plan', 'subkernel_names', 'precedence')

    def __init__(self, name, plan, subkernel_names, precedence=None):
        self.name = name
        self.plan = plan
        self.subkernel_names = subkernel_names
        self.precedence = precedence

    def donor_order(self):
        """Sub-kernel names in donor priority: the union_precedence order (filtered to
        this metro's sub-kernels, then any unlisted ones in call order), else call
        order."""
        if not self.precedence:
            return list(self.subkernel_names)
        subs = set(self.subkernel_names)
        ordered = [n for n in self.precedence if n in subs]
        ordered += [n for n in self.subkernel_names if n not in self.precedence]
        return ordered


class OperatorShell:
    """A parsed @ati.operator: the passive OperatorDecl + its backend refs as
    (index, kind, name) where kind is 'metro' | 'kernel' | 'affine'. default_kdesc +
    struct are DERIVED by the linker (A1/A3); the surface declares neither."""

    __slots__ = ('name', 'decl', 'backend_refs')

    def __init__(self, name, decl, backend_refs):
        self.name = name
        self.decl = decl
        self.backend_refs = backend_refs      # index-sorted (index, kind, name)


class CompiledFamily:
    """Pass-1 output for one family: parsed shells keyed by NAME, plus declared order."""

    def __init__(self, family):
        self.family = family
        self.kernels = {}      # def-name -> KernelShell
        self.metros = {}       # backend enum-name -> MetroShell
        self.affines = {}      # affine NAME -> AffineDecl
        self.operators = {}    # op-name -> OperatorShell
        self.op_order = []     # operator NAMEs in declared order


def _plan_subkernel_names(plan):
    """Every concrete sub-kernel NAME a MetroPlan calls (descending into if/else)."""
    from aotriton.template_instantiation.specs.metro import Call, Cond
    out = []

    def walk(steps):
        for s in steps:
            if isinstance(s, Call):
                out.append(s.kernel)
            elif isinstance(s, Cond):
                walk(s.then)
                walk(s.orelse)
    walk(plan.steps)
    return out


class Parser:
    """Pass 1 — COMPILE. Owns the `modules/` root (the per-family kernel/operator
    descriptions) and turns each family's passive `@ati.*` records into IR shells.

    `module_dir` is `<root_dir>/modules`, given explicitly by the generator
    (--root_dir). `modules/` is DATA beside the package source, NOT inside the
    importable `aotriton` package (a non-editable install copies the package out of
    the checkout but never ships modules/), so its location is passed in — never
    derived from `__file__` or the cwd."""

    def __init__(self, module_dir):
        self.module_dir = Path(module_dir)

    # --- family discovery / loading (absorbed from python/rules) -------------

    def discover_families(self):
        """Family names = subdirs of module_dir that contain an `aot` package."""
        if not self.module_dir.is_dir():
            return []
        return [child.name for child in sorted(self.module_dir.iterdir())
                if (child / 'aot' / '__init__.py').is_file()]

    def load_family_aot(self, family):
        """Import <module_dir>/<family>/aot/__init__.py by path under a synthetic
        unique package name so its relative imports work without a <family> namespace
        pkg. `modules/<family>` must stay a plain directory (not a package) so its
        `kernel/` sources keep importing each other by bare name; loading `aot` by
        name would require `<family>` to be a clean namespace package, which any
        sys.path entry containing a `<family>.py` (e.g. `tritonsrc/flash.py`) would
        shadow. Loading by path sidesteps that entirely. Cached in sys.modules."""
        modname = f'_aotriton_modules_{family}_aot'
        cached = sys.modules.get(modname)
        if cached is not None:
            return cached
        aot_dir = self.module_dir / family / 'aot'
        spec = importlib.util.spec_from_file_location(
            modname, aot_dir / '__init__.py',
            submodule_search_locations=[str(aot_dir)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        return mod

    # --- compile ------------------------------------------------------------

    def _parse_kernel(self, def_obj, compiled):
        """Record a triton-kernel `def` (carrying fn.__ati__) as a KernelShell."""
        from aotriton.template_instantiation.specs.finalize import get_kernel_spec
        spec = get_kernel_spec(def_obj)
        assert spec is not None, (
            f'{getattr(def_obj, "__name__", def_obj)!r} has no @ati.* kernel spec')
        name = getattr(spec.kernel, '__name__', None)
        assert name, f'kernel {def_obj!r} has no __name__'
        if name not in compiled.kernels:
            compiled.kernels[name] = KernelShell(name, spec, spec.source_path)
        return name

    def compile_family(self, aot_module, family):
        """Pass 1 for one family: walk the `operators` roots, parse every reachable
        kernel / metro / affine into a shell, and record cross-references as
        relocations on the shells. Returns a CompiledFamily (nothing is built yet)."""
        from aotriton.template_instantiation.specs.metro import MetroPlan

        compiled = CompiledFamily(family)
        for op_def in getattr(aot_module, 'operators', []):
            decl = getattr(op_def, '__ati_operator__', None)
            assert decl is not None, (
                f'{family}: operators entry {op_def!r} has no __ati_operator__ '
                f'(not a passive @ati.operator def)')
            backend_refs = []
            for b in decl.backends:
                ref = b.obj        # the in-file def/object the @ati.backend points at
                if getattr(ref, '__ati_metro__', None) is not None:
                    plan = ref.__ati_metro__
                    assert isinstance(plan, MetroPlan)
                    sub_names = _plan_subkernel_names(plan)
                    for sub_name in sub_names:
                        sub_def = getattr(aot_module, sub_name, None)
                        assert sub_def is not None, (
                            f'{family}: metro {b.name!r} calls sub-kernel '
                            f'{sub_name!r} not found in the aot module')
                        self._parse_kernel(sub_def, compiled)
                    if b.name not in compiled.metros:
                        precedence = getattr(ref, '__ati_union_precedence__', None)
                        compiled.metros[b.name] = MetroShell(b.name, plan, sub_names,
                                                             precedence=precedence)
                    backend_refs.append((b.index, 'metro', b.name))
                elif getattr(ref, '__ati__', None) is not None:
                    kname = self._parse_kernel(ref, compiled)   # bare-kernel backend
                    backend_refs.append((b.index, 'kernel', kname))
                elif getattr(ref, '__ati_affine__', None) is not None:
                    adecl = ref.__ati_affine__
                    if adecl.name not in compiled.affines:
                        compiled.affines[adecl.name] = adecl
                    backend_refs.append((b.index, 'affine', adecl.name))
                else:
                    raise AssertionError(
                        f'{family}: backend {b.name!r} ref {ref!r} is neither a '
                        f'metro, a kernel, nor an affine description')
            backend_refs.sort(key=lambda t: t[0])
            compiled.operators[decl.name] = OperatorShell(decl.name, decl,
                                                          backend_refs)
            compiled.op_order.append(decl.name)

        return compiled
