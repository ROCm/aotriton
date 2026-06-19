# AOTriton Template Instantiation (ATI)

ATI is the declarative front-end for describing AOTriton kernels and operators.
An author writes Python decorator stacks that record *what* a kernel does; the ATI
compiler pipeline translates those descriptions into C++ shim headers, operator
dispatchers, and autotune tables without ever importing or executing Triton.

---

## Part I — User Guide: Describing Kernels and Operators

### 1. Concepts

ATI descriptions live in a **family registry** — a Python package under
`modules/<family>/aot/`. The registry is purely *declarative*: it imports existing
kernel description files and defines operator/metro structures, then exposes
`operators = [...]` as the build roots.

Every description is made with **stacked decorator blocks**, all terminated by
`@ati.start` at the top:

```
@ati.start           ← outermost; runs last; triggers finalization
@ati.tensor(...)     ← specs accumulate bottom-up ...
@ati.scalar(...)
@ati.source(...)     ← innermost; runs first; supplies the kernel object
def my_kernel():     ← placeholder def — body is IGNORED
    pass             ← use `pass` or `...`; the function is never called
```

Python applies decorators from bottom to top, so `@ati.source` runs first and
returns a stub; each spec above it accumulates onto that stub; finally `@ati.start`
reads all accumulated specs in source order and produces the passive description
record.

**The placeholder `def` name is the kernel's identity.** It must match the
symbol name in the Triton source file (or the `name=` argument to
`@ati.source`). The function body is never executed — write `pass` or `...`.
The only meaningful content on the placeholder `def` is optional **string type
annotations** on the parameters, which become `ScalarSpec`s (see
[§2.1](#21-atisourcepath-namenone)).

There are four kinds of stacked block, distinguished by the innermost decorator:

| Innermost | Block kind | Produces |
|---|---|---|
| `@ati.source(...)` | Kernel | `KernelSpec` on `fn.__ati_node__` |
| `@ati.affine.aiter_asm(...)` | Affine | `AffineDecl` on `fn.__ati_node__` |
| `@ati.operator(...)` | Operator | `OperatorDecl` on `fn.__ati_node__` |
| `@ati.metro_kernel` | Metro | `MetroPlan` on `fn.__ati_node__` |

---

### 2. Describing a Triton Kernel

```python
import aotriton.template_instantiation as ati

# Named type variable shared by several tensors.
T_io = ati.type_var('T_io', dtype=['*fp16:16', '*bf16:16', '*fp32:16'],
                    signature_name='Q')

@ati.start
@ati.disable(when=lambda f: f.choices.CAUSAL_TYPE != 0 and f.choices.BIAS_TYPE != 0)
@ati.type_var('T_seq', dtype=['*i32:16'])          # anonymous (only one tensor)
@ati.tensor('Q',   T_io, strides='stride_q?', contiguous=-1)
@ati.tensor('K',   T_io, strides='stride_k?', contiguous=-1)
@ati.tensor('V',   T_io, strides='stride_v?', contiguous=-1)
@ati.tensor('Out', T_io, strides='stride_o?', contiguous=-1)
@ati.tensor('B',   T_io, strides='stride_b?')      # zeroed by override below
@ati.scalar('Max_seqlen_q', T_seq)                 # bound to shared variable
@ati.scalar('CAUSAL_TYPE', options=[0, 3])          # enumerated constexpr
@ati.scalar('BIAS_TYPE',   options=[0, 1])
@ati.derives('B', to=0, when=ati.eq('BIAS_TYPE', 0))  # conditional override
@ati.tune.schema(AttnFwdPerf)                      # perf parameters
@ati.tune.configs(gen_autotune_configs)             # autotune config generator
@ati.tune.binning(Max_seqlen_q=ati.tune.binning.le,
                  Max_seqlen_k=ati.tune.binning.le) # LUT binning keys
@ati.source('../kernel/fwd_kernel.py')             # innermost: Triton source
def attn_fwd():
    pass
```

#### 2.1 `@ati.source(path, name=None)`

The innermost decorator. Resolves `path` relative to the description file,
AST-parses the Triton source (no import, no Triton required in the generator
environment), and returns a `KernelStub` carrying the parameter names. The
kernel symbol name defaults to the placeholder `def`'s name; pass `name=` to
override.

The placeholder `def` may also declare scalar parameters with **string type
annotations** as a terse alternative to stacked `@ati.scalar`:

```python
@ati.start
@ati.source('../kernel/fwd_kernel.py')
def attn_fwd(dropout_p: 'fp32', philox_seed: '*u64'):
    pass
```

Each string-annotated parameter becomes a `ScalarSpec`; a conflict with an
explicit `@ati.scalar` for the same parameter is an error.

#### 2.2 Choice Variables — `ati.type_var` / `ati.scalar_var`

A **choice variable** is a named, reusable set of TypedChoices. Multiple tensors
or scalars can bind to the same variable, grouping them into one enumeration
axis:

```python
T_io = ati.type_var('T_io', dtype=['*fp16:16', '*bf16:16', '*fp32:16'],
                    signature_name='Q')

@ati.tensor('Q', T_io, strides='stride_q?', contiguous=-1)
@ati.tensor('K', T_io, strides='stride_k?', contiguous=-1)
# → Q and K share one axis; they always have the same dtype.
```

`signature_name` sets the label for persisted artifacts (compact signature,
DB row keys). Multi-choice variables spanning several tensors must declare it
explicitly (it would otherwise be an arbitrary per-arg pick baked into stored
artifacts).

`ati.scalar_var` works identically for scalar enumerations.

#### 2.3 Tensor Binding — `@ati.tensor`

```python
ati.tensor(arg_name, dtype, *,
           strides=None,    # glob pattern, e.g. 'stride_q?'
           rank=None,       # explicit rank (else inferred from stride count)
           contiguous=None, # index or name of the contiguous (unit) stride
           wires_to=None)   # operand name in the operator's params struct
```

`strides` is a glob pattern matched against the kernel signature to discover
stride parameters automatically. `contiguous` marks one stride as always 1
(it becomes a `stride_unit` hidden axis — constexpr, not passed at launch).
`wires_to` maps this real argument to an operator operand (for metro wiring).

A `dtype` of `'*fp16:16'` is a literal ATI type string; a `ChoiceVar` (from
`ati.type_var`) makes the tensor polymorphic. Multiple argument names can share
one binding by passing a list: `ati.tensor(['Q', 'K'], T_io, rank=2)`.

#### 2.4 Scalar Binding — `@ati.scalar`

```python
ati.scalar(arg_name, type_or_var=None, *, options=None, wires_to=None)
```

- `ati.scalar('Sm_scale', 'fp32')` — plain runtime scalar with explicit type.
- `ati.scalar('Sm_scale')` with a `'fp32'` annotation on the def — same effect.
- `ati.scalar('CAUSAL_TYPE', options=[0, 3])` — enumerated constexpr scalar.
- `ati.scalar('Max_seqlen_q', S)` — bound to a shared `ati.scalar_var`.

`options` accepts a Python list (C width inferred from values) or a numpy array
with an explicit dtype (`np.array([0, 3], np.int16)`) to fix the struct width.

#### 2.5 Conditional Overrides — `@ati.derives`

```python
ati.derives('B', to=0, when=ati.eq('BIAS_TYPE', 0))
# → functional where BIAS_TYPE==0 sees B constexpr-zeroed.

ati.derives('B', to=ati.VarRef('T_io'), when=ati.ne('BIAS_TYPE', 0))
# → functional where BIAS_TYPE!=0 inherits B's type from T_io.
```

Predicates: `ati.eq`, `ati.ne`, `ati.lt`, `ati.gt`, `ati.le`, `ati.ge`, or a
callable `when=lambda f: f.choices.CAUSAL_TYPE != 0`.

A zeroed tensor (`to=0`) implicitly cascades to all its non-unit strides — the
description stays terse.

`@ati.overrides` is an alias for `@ati.derives`.

#### 2.6 Disable Predicates — `@ati.disable` / `@ati.no_disable`

```python
@ati.disable(when=lambda f: f.arch.startswith('gfx11') and f.choices.BLOCK_DMODEL > 256)
```

Excludes functionals where the predicate fires from code generation. The
predicate receives a `Functional` and reads `f.choices.<var>` and `f.arch`.

When a kernel `@ati.cite`s another and inherits its disable predicate, but the
predicate reads choice variables the citing kernel doesn't have, use
`@ati.no_disable()` to explicitly replace the inherited predicate with one that
never fires:

```python
@ati.start
@ati.no_disable()            # citing kernel has no relevant exclusions
@ati.cite('op_attn_fwd.triton.attn_fwd')
@ati.source('../kernel/bwd_preprocess.py')
def bwd_preprocess():
    pass
```

#### 2.7 Cross-Kernel References — `@ati.cite`

```python
@ati.cite('op_attn_fwd.triton.attn_fwd')          # 3-segment: one sub-kernel
@ati.cite('op_attn_bwd.triton_split')              # 2-segment: whole metro
```

Fills **gap arguments** — signature parameters the citing kernel doesn't claim
locally — from the cited kernel's bindings, matched by apparel name. Gap filling
happens at link time (Pass 2), so there is no order constraint between the
citing and cited kernel in the source file.

#### 2.8 Performance Tuning — `@ati.tune.*`

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class AttnFwdPerf:
    BLOCK_M:         np.int16 = 16   # every field MUST have a default
    BLOCK_N:         np.int16 = 16
    PRE_LOAD_V:      bool = False
    NUM_XCDS:        np.int8 = 1

@ati.tune.schema(AttnFwdPerf)    # declare the perf struct
@ati.tune.configs(gen_autotune_configs)   # generator yields Config objects
@ati.tune.binning(Max_seqlen_q=ati.tune.binning.le)  # DB LUT indexing
@ati.tune.fallback(PADDED_HEAD=False)    # partial-tune defaults
```

A kernel without `@ati.tune.configs` is **schema-only** (untunable) — the
default perf values are always used. Perf-field overrides via `@ati.derives`
target perf param names and apply in the perf layer, never in the functional
struct.

`Config(kw, num_warps=4, num_stages=2)` — mirrors `triton.Config` API for
drop-in compatibility.

---

### 3. Metro Kernels — Wiring Collaborating Kernels

A **metro** sequences multiple Triton kernels for one operator functional using
ordinary Python `if`/`else`. The DSL body is AST-transpiled; it is never
executed:

```python
@ati.start
@ati.hints.union_precedence([bwd_kernel_dk_dv, bwd_kernel_dq,
                             bwd_preprocess_varlen, bwd_preprocess])
@ati.metro_kernel
def metro_bwd(params):
    if params.num_seqlens > 0:
        bwd_preprocess_varlen(params)
    else:
        bwd_preprocess(params)
    bwd_kernel_dk_dv(params)
    bwd_kernel_dq(params)
```

`@ati.metro_kernel` is the innermost marker (transpiles the body into a
`MetroPlan`); `@ati.start` finalizes above it.

**Grammar**: only `kernel(params)` calls and `if params.<X>[.data_ptr()] <op>
<literal>: ...` conditionals are valid. The condition reads one `params`
attribute and compares it to a literal.

**`@ati.hints.union_precedence([...])`** declares merge priority when
sub-kernels disagree on the binding of shared operands. The cited kernel's cite
gap-fill also respects this order: the highest-priority sub-kernel's binding
wins. This is the only hint provided today.

---

### 4. Operators — Multi-Backend Dispatch

An operator dispatches among interchangeable backends (triton metro, fused
triton kernel, or affine ASM kernel):

```python
@ati.start
@ati.tune.fallback(PADDED_HEAD=False)
@ati.tune.binning(Max_seqlen_q=ati.tune.binning.le,
                  Max_seqlen_k=ati.tune.binning.le)
@ati.backend(1, aiter_fmha_v3_fwd, 'aiter')    # explicit dispatch index
@ati.backend(0, metro_fwd, 'triton')
@ati.operator(call_options_name='attn_options')  # innermost
def op_attn_fwd():
    pass
```

- `@ati.backend(index, ref, name)` — `index` is the dispatch/enum/DB order
  (load-bearing: tuning rows store this integer). `name` forms the C++ enum
  (e.g. `'triton'` → `kOp_Triton`). `ref` is the in-file def/object.
- `@ati.operator(call_options_name=...)` — the innermost marker.
- Operator-level `@ati.tune.binning` is the **OPTUNE** binning (which backend
  to pick), not kernel-level autotune.
- Operator-level `@ati.tune.fallback` provides PARTIALLY_TUNED defaults.

The linker **derives** the params struct (union over all backends' argument
surfaces) and the default backend (first tunable sub-kernel of backend 0); the
operator description declares neither.

---

### 5. Slim Affine (AITER ASM) Kernels

Affine kernels are thin C++ shims that call a 3rd-party AITER API; they carry
pre-built `.co` files and own no functional space of their own:

```python
@ati.start
@ati.disable(when=_aiter_fwd_disabled)
@ati.affine.shared_operator('op_attn_fwd')
@ati.affine.arch(['gfx942', 'gfx950'])
@ati.affine.limitations(Q=lambda dtype: 'fp16' in dtype or 'bf16' in dtype,
                        BLOCK_DMODEL=lambda x: x in [128, 192])
@ati.affine.structures(cookie='aiter::mha_fwd_args')
@ati.affine.directories(co_dir='fmha_v3_fwd',
                        headers=['aotriton/_internal/flash/aiter.h'])
@ati.affine.aiter_asm(name='aiter_fmha_v3_fwd')    # innermost
def aiter_fmha_v3_fwd():
    pass
```

| Decorator | Purpose |
|---|---|
| `@ati.affine.aiter_asm(name=...)` | Innermost marker; names the shim |
| `@ati.affine.shared_operator(op_name)` | Operator whose params struct this borrows (SHARED_IFACE) |
| `@ati.affine.arch([...])` | Supported arches |
| `@ati.affine.limitations(key=predicate, ...)` | Choice filters (excluded combinations) |
| `@ati.affine.structures(cookie='...')` | 3rd-party API struct type name |
| `@ati.affine.directories(co_dir, headers=[...])` | `.co` repository path + extra headers |
| `@ati.affine.supplies(specs, after=..., before=...)` | Extra operands contributed to the operator's params union (e.g. DQ_ACC for bwd) |

---

### 6. Family Registry Layout

```
modules/
  <family>/
    aot/
      __init__.py       ← build roots: declares operators, metros, imports
      <kernel>.py       ← one file per kernel description
      <affine>.py       ← one file per affine kernel description
      _common.py        ← shared helpers (disable predicates, value lists)
    kernel/             ← Triton kernel source files referenced by @ati.source
      <kernel>.py
```

`__init__.py` must expose `operators = [op_def, ...]` as the parser's entry
points. It should be **purely declarative** — no build calls, no registry writes.
Family name is inferred from the path: `modules/<family>/aot/`.

---

## Part II — Developer Guide: Architecture and Implementation

### 1. Overview

ATI is a **two-pass compiler** over a tree of passive description records:

```
Author writes:     stacked-@ decorator blocks
Stage 1 (decorators):  specs accumulate on defs as StackedSpec records
Stage 2 (finalize):    @ati.start consolidates specs → passive AtiNode records
Pass 1 (parser):       AtiNode records → lightweight shells (strings as refs)
Pass 2 (linker):       shells → resolved IR tree (cite gaps filled, IR built)
Codegen:               IR tree → C++ headers, operator dispatchers, autotune
```

### 2. Stage 2: Passive Description Records (specs/)

Every `@ati.start` block produces one **AtiNode** subclass instance stored as
`fn.__ati_node__`. All four record classes share this attribute name, allowing
`isinstance()` dispatch everywhere.

```
AtiNode (specs/node.py)
  ├── KernelSpec   (specs/kernel.py)   — @ati.source + all specs
  ├── AffineDecl   (specs/affine.py)  — @ati.affine.* stack
  ├── OperatorDecl (specs/operator.py) — @ati.operator stack
  └── MetroPlan    (specs/metro.py)   — @ati.metro_kernel transpiled AST
```

**`KernelSpec`** is the kernel's passive "object file". It differs from the
other three records in that it must be **cloned and mutated during linking**
(cite resolution appends gap tensors/scalars/overrides onto a per-link copy).
`OperatorDecl` and `AffineDecl` carry no cross-kernel references, so the linker
reads them verbatim.

#### 2.1 The Stacked-@ Mechanics

Every spec class inherits `StackedSpec` (specs/base.py), which provides one
`__call__`:

```python
class StackedSpec:
    __slots__ = ()
    def __call__(self, target):
        from .finalize import accumulate_spec
        return accumulate_spec(self, target)
```

`accumulate_spec(spec, kernel)` appends to `kernel.__ati_pending__` and returns
`kernel`. The pending list is cleared by `@ati.start`.

`@ati.start` (specs/finalize.py) dispatches O(1) on the **innermost spec**
(`specs[-1]` after source-order reversal — Python's bottom-up application
guarantees the innermost decorator's spec is always the kind discriminant):

```python
marker = specs[-1]
if isinstance(marker, OperatorSpec):    → _finalize_operator → OperatorDecl
elif isinstance(marker, AffineKernelSpec): → _finalize_affine → AffineDecl
elif isinstance(marker, MetroPlan):    → _finalize_metro  → MetroPlan
else:                                  → describe()        → KernelSpec
```

This means every affine stack must have `@ati.affine.aiter_asm` as the
**innermost** decorator (directly above the `def`), with all other
`@ati.affine.*` specs above it.

#### 2.2 `@ati.source` and KernelStub

`@ati.source` (decorators/source.py) AST-parses the Triton source without
importing it. It returns a `KernelStub`:

```python
class KernelStub:
    __slots__ = ('__name__', 'params', 'source_path', 'annotations',
                 '__ati_pending__', '__ati_node__')
```

`params` is the parameter-name list (from AST). `annotations` is the dict of
string type annotations the author wrote on the placeholder `def`. The finalizer
converts each annotation into a `ScalarSpec` (terse alternative to stacked
`@ati.scalar`); a conflict with an explicit spec on the same param is an error.

#### 2.3 `describe()` — Kernel Finalization

```python
def describe(kernel, *specs, _validate=True):
```

1. Introspect params via `kernel_params(kernel)` (reads `KernelStub.params`, or
   falls back to `inspect.signature` for plain callables in test fixtures).
2. Partition specs into typed buckets: tensors, scalars, overrides, tune, disables,
   dtype_vars, cites, plus placeholder-def string annotations → extra ScalarSpecs.
3. Validate completeness: every signature parameter must be claimed exactly once
   by a tensor/scalar/tune-schema/stride-glob. Unclaimed params are only allowed
   when `@ati.cite` is present (the linker fills gaps at build time).
4. Build `TuneSpec` from collected tune records.
5. Attach `KernelSpec` as `kernel.__ati_node__`.

### 3. Stage 3 (Builder): KernelSpec → BuiltKernel

`builder/kernel.py` lowers one `KernelSpec` into Axis + Override IR:

```python
def build_kernel(kernel_spec) -> BuiltKernel:
    name = ...
    param_index = ...
    _resolve_named_dtypes(kernel_spec, name)   # string dtype refs → ChoiceVar

    axes, nonunit_strides = _build_axes(kernel_spec, param_index, name)
    axes.sort(key=lambda a: a.anchor)

    wiring = _collect_wiring(kernel_spec.tensors, kernel_spec.scalars)
    functional_overrides, perf_overrides = _split_overrides(
        kernel_spec.overrides, kernel_spec.tune, name)
    functional_overrides += _synthesize_stride_overrides(
        functional_overrides, nonunit_strides)

    return BuiltKernel(name, axes, functional_overrides, ...)
```

**`_build_axes`** groups specs by choice variable, resolves choices, computes
anchors (parameter index of the first argument), and emits:
- One `Axis(kind='tensor')` per choice group of tensors
- One `Axis(kind='stride' or 'stride_unit')` per matched stride parameter —
  hidden axes that participate in the signature but not in Godel counting
- One `Axis(kind='scalar')` per choice group of scalars

**Axes are anchor-sorted** (earliest signature argument first) to ensure
reproducible Godel numbering on both the Python and C++ sides.

**Overrides split** into two channels: those targeting perf-schema fields go to
`perf_overrides` (applied only in the perf translation layer); the rest go to
`functional_overrides` (applied per-functional in `enumerate_functionals`).

**Stride override cascade**: a functional override that zeroes a tensor
(`to=0`) implicitly cascades to all its non-unit strides — the builder
synthesises stride overrides automatically so the description stays terse.

`BuiltKernel` → `KernelDescription(Interface)` wraps the built IR with the
codegen surface (godel number, axes_multi, perf struct, `list_functional_params`,
`func_cfields`, etc.).

### 4. Pass 1: COMPILE (codegen/parser.py)

The `Parser` / `FamilyCompiler` visitor walks the operator tree starting from
`aot_module.operators` and records every reachable kernel/metro/affine as a
**shell** — a lightweight struct holding the passive record plus unresolved
cross-references ("relocations") as strings:

```
CompiledFamily
  kernels  — {def-name → KernelShell(name, KernelSpec, source_path)}
  metros   — {enum-name → MetroShell(name, MetroPlan, subkernel_names, precedence)}
  affines  — {affine-name → AffineDecl}
  operators — {op-name → OperatorShell(name, OperatorDecl, backend_refs)}
  op_order — [operator names in declared order]
```

Backend dispatch uses `isinstance` on `fn.__ati_node__` (an `AtiNode`
subclass), with one `visit_<kind>` method per node type — an explicit
visitor pattern. Adding a new backend kind = adding a `visit_<kind>` method;
the dispatch loop does not change.

Metro sub-plan descent (`Call`/`Cond` tree) uses `_iter_plan_subkernels` — a
generator that recurses into `Cond.then`/`Cond.orelse`, yielding concrete
sub-kernel names. This is the Pass-1 analogue of `ir/ops/infer._iter_subkernels`
on the built IR.

### 5. Pass 2: LINK (codegen/linker.py)

The `Linker` resolves each `CompiledFamily` into final IR:

1. **Topological sort** (`_kernel_build_order`): Kernels that cite others must
   be built after their cite targets. A citing kernel that contains itself
   within a metro it cites (a true implementation cycle) is acyclic in *header*
   terms and terminates — the citer only needs the cited kernels' argument
   surfaces, not their own.

2. **Spec cloning**: Each kernel's `KernelSpec` is shallowly cloned with fresh
   mutable lists before cite resolution, keeping the module-level spec
   immutable and linking idempotent.

3. **Cite resolution** (`ir/ops/cite.py`): For each `@ati.cite` target, find
   the cited kernel's argument surface and append gap tensors/scalars/overrides
   to the citing kernel's cloned spec. A whole-metro cite donates all sub-kernels'
   argument surfaces in `union_precedence` priority order.

4. **`build_kernel`**: Lower the cite-resolved `KernelSpec` → `BuiltKernel`.

5. **`KernelDescription`**: Wrap `BuiltKernel` into the codegen-facing IR.
   Assign Godel strides; compute `_godel_number` (product of radices);
   build perf struct, arg index, baked-args set, and autotune keys.

6. **Metro build** (`builder/metro.py`): `lower_plan(MetroPlan, kernel_map,
   ...)` replaces string sub-kernel names with live `KernelDescription`
   objects. `ConditionalKernel(Interface)` wraps each `Cond` step.

7. **Operator derivation** (linker.py `_derive_*`): The linker derives:
   - `default_kdesc` (A1): first tunable concrete sub-kernel of backend 0 — the
     functional-axes owner.
   - `struct_cfields` (A3): order-preserving union of all backends' `func_cfields`
     via `build_merged_struct_cfields`. When backend 0 is already a superset,
     returns `None` and the operator reuses the default kernel's cfields.

8. **`infer_shared_iface`** (`ir/ops/infer.py`): Resolves
   `@ati.affine.shared_operator(name)` string references to actual `Operator`
   objects, setting `SHARED_IFACE` so the affine kernel's shim borrows the
   operator's params struct.

9. **`FamilyArtifacts`**: `(kernels, operators, affine_kernels)` — the three
   lists the codegen drivers consume.

### 6. IR Type Hierarchy

All concrete IR types implement `Interface(ABC)` (`ir/interface.py`):

```
Interface (ABC)
  ├── KernelDescription   CODEGEN_MODULE='triton', TUNE_NAME='autotune'
  ├── Operator            CODEGEN_MODULE='op',     TUNE_NAME='optune'
  ├── MetroKernel         CODEGEN_MODULE='op',     TUNE_NAME=None
  ├── ConditionalKernel   CODEGEN_MODULE='op'
  └── AffineKernel        CODEGEN_MODULE='affine', TUNE_NAME=None
```

`Interface` declares two abstract methods that every concrete type must
implement:

```python
@property @abstractmethod
def func_cfields(self) -> list[cfield]:
    """C struct fields for the params struct."""

@abstractmethod
def list_functional_params(self) -> Iterator[TemplateParam]:
    """Per-axis view for compiled-in feature tables."""
```

Metro/conditional launchers own no functional space; they return empty lists.
`gen_functionals(target_arch)` is implemented on `Interface` using
`_axes_overrides()` (which each concrete type overrides to supply its axes and
overrides); the operator overrides `_axes_overrides` to read from its
`default_kdesc` while setting `meta_object=self`.

**`Functional`** (`ir/functional.py`) is one fully-pinned instantiation:

```python
class Functional:
    meta_object   # owning Interface
    arch, arch_number, godel_number
    choice        # {var_name → TypedChoice} (post-pick)
    resolved      # {arg_name → TypedChoice} (post-override)
    _optimized_for  # GPU list for this functional

    @property def choices -> ChoiceView   # f.choices.T_io ergonomic accessor
```

**`TemplateParam`** (`ir/axis.py`) pairs an `Axis` with per-kernel
wiring/override state for the compiled-in feature tables:

```python
@dataclass(slots=True)
class TemplateParam:
    axis: Axis
    repr_name: str          # apparel-mapped getter name
    all_names: list[str]    # apparel-mapped member arg names
    overridden_to_constexpr: bool
```

### 7. Code Generation

The root generator (`codegen/root.py`) calls `Linker.link_all_families()` to
obtain the IR tree, then dispatches each item to the appropriate generator:

| IR type | Generator | Output files |
|---|---|---|
| `KernelDescription` | `KernelShimGenerator` | `shim.<name>.{h,cc}` |
| `Operator` | `OperatorGenerator` | `op.<name>.{h,cc}` |
| `AffineKernel` | `SlimAffineGenerator` | `affine.<name>.{h,cc}` |
| per-functional | `AutotuneCodeGenerator` | `autotune.<name>/<godel>.cc` |

Each generator inherits `InterfaceGenerator` and overrides `write_shim_header`
/ `write_shim_source`. Templates live under `codegen/template/` as Jinja-style
`.h`/`.cc` files.

**Kernel shim** (`codegen/kernel.py`): Emits the C++ struct (func fields +
perf fields), the Godel-number computation function (`godel_number_body`),
compiled-in feature tables (`get_<name>_choices()`), and the autotune LUT
(`Autotune_<name>__A<arch>__F<godel>`). Per-functional autotune entries are
emitted by `AutotuneCodeGenerator`, which reads the tuning DB via
`KernelDescription.translate_dataframe`.

**Operator dispatcher** (`codegen/operator.py`): Emits the backend enum, a
`godel_number()` function (same mixed-radix computation but over the operator's
axes), per-backend launcher stubs, and the optune LUT.

**Generation modes**:
- `--noimage_mode`: Struct/header generation only; no HSACO images produced.
- Default: Full build including HSACO packing.
- `--build_for_tuning`: Emit all functional entries (including un-tuned)
  so the tuning infrastructure can submit them.

### 8. Key Design Invariants

**PASSIVE DECLARATIONS**: Decorators only record specs; no IR is built until
the linker runs. The same family module can be imported repeatedly without
side effects.

**TWO-PHASE BUILD**: Pass 1 records all cross-references as strings; Pass 2
resolves them in topological order. This lets any sub-kernel cite any other
sub-kernel in the same family, including patterns that form implementation-level
cycles (a sub-kernel citing the metro it's inside).

**TRITON-FREE GENERATOR**: The generator never imports or executes Triton code.
`@ati.source` uses Python's `ast` module on the Triton source file; parameter
types come entirely from `@ati.tensor`/`@ati.scalar` decorators.

**SINGLE ATTRIBUTE `__ati_node__`**: Every described `def` carries exactly one
`__ati_node__` attribute holding its `AtiNode` subclass instance. Dispatch uses
`isinstance()`, not string-attribute probing.

**GODEL NUMBERING**: Multi-choice axes are assigned mixed-radix strides in
canonical anchor order; the same ordering is reproduced in the generated C++.
A functional's identity is `(arch_number, godel_number)`.

**OVERRIDE CHANNELS**: Overrides are split at build time into functional
(applied per-functional during enumeration) and perf (applied only in the
tuning translation layer, never in `resolved[]`).
