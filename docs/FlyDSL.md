# FlyDSL backends in AOTriton

FlyDSL is a third kernel language AOTriton can build a backend from, alongside
Triton (`@ati.source`) and aiter (`@ati.affine.*`). This document describes the
*mechanism* — how a FlyDSL backend is pinned, vendored, compiled, generated and
dispatched — and is deliberately kernel-family agnostic. Flash attention is used only as the worked
example, because it is the family that exists; nothing here is specific to it,
and a reader wiring up the next family should not have to read that one's head
dimension ladders to find the machinery.

## How a FlyDSL backend differs from the other two

|  | Triton (`@ati.source`) | aiter (`@ati.affine.*`) | FlyDSL (`@ati.flyc.*`) |
|---|---|---|---|
| kernel source | a `.py` in `modules/<family>/kernel/` | prebuilt, external | a `.py` in `modules/<family>/flyc/`, vendored |
| when compiled | during the AOTriton build | never (linked) | during the AOTriton build |
| functional space | the operator's, filtered | n/a | **inherits** the operator's and filters it |
| kernel arguments | the operator's params struct | n/a | **its own**, unrelated to the params struct |

The last two rows are the whole of what makes FlyDSL a separate concept rather
than a second flavour of Triton.

A flyc kernel **has no functional space of its own.** It inherits the
operator's and narrows it: an operator may offer head dimensions 16 through 512
while this backend compiles a subset, so the backend declares which rungs it
builds and the dispatcher maps a request onto the nearest one it has.

A flyc kernel's **kernel-argument list is not the operator's params struct.**
A Triton kernel is written against AOTriton's own argument names, so a launch
argument is usually the same name on both sides. A FlyDSL kernel is written
elsewhere and vendored, so its argument list is whatever upstream chose. The
description therefore declares each kernel argument explicitly, with its type
and pointer alignment, and says where the value comes from — an operator
operand under a different name, or a host-side helper that computes something
the operator has no field for.

## The pins

Three files, deliberately independent, because FlyDSL enters a build through
three channels that version separately.

| file | pins | used for |
|---|---|---|
| `third_party/flydsl-compiler.txt` | a pip requirement | the wheel: the compiler itself |
| `third_party/flydsl-kernel.txt` | a git tag | a source tree supplying FlyDSL's shared kernel libraries |
| `third_party/flydsl-llvm.txt` | a `git+<url>@<ref>` line, or nothing | the LLVM that wheel must be built against |

**The compiler is a wheel, not a submodule,** because FlyDSL bundles its own
LLVM/MLIR: its `setup.py` wants a prebuilt bundled MLIR, so
`pip install third_party/flydsl` would not be self-contained the way the
Triton submodule is. Two ways to move off the pinned wheel, for two different
situations: edit `flydsl-compiler.txt` once a build is published and every
build should use it; set `AOTRITON_USE_LOCAL_FLYDSL_WHEEL` for a wheel that
exists only on local disk.

**The kernel tree is a second pin because the wheel ships no `kernels/` at
all,** and vendored kernels are written against FlyDSL's shared kernel
libraries — buffer operations, common helpers, and whatever family-specific
utility module the kernels derive from. CMake shallow-clones the tag into the
build directory, and the CMake cache variable `AOTRITON_FLYDSL_KERNEL_ROOT`
points there by default. The *environment variable* of the same name — which is
what `flyc_bootstrap` actually reads, and what CMake passes down to each compile
rule — has no default at all: outside a CMake build you must set it yourself.
Setting it to a local checkout is supported and is how you build against
in-progress kernel changes — but note that **one root serves every
architecture**, so moving it moves them all, and a regression on one can
present as a regression on another.

**The LLVM pin is a tripwire and is normally empty.** Fill it only while
upstream LLVM is known to miscompile something the released wheel depends on.
While it is non-empty, a build that has not been given a locally built wheel
fails at configure time, naming how to produce one. That is deliberate: a wheel
built against a broken LLVM produces kernels that are *wrong* rather than
absent, and no gate short of a numerical test would catch it.

## The vendoring contract

`modules/<family>/flyc/` is a **bare directory of Python files**, modelled on
`modules/<family>/kernel/`: no `__init__.py`, not a package, flat sibling
imports between the files. Anything that imports from it puts the directory on
`sys.path` first — the same contract the Triton kernel directory already uses.

Rules that make a re-sync cheap, and that a re-syncer must keep:

* **Copy files verbatim, then reapply the local edits as a separate step,** so
  that a diff against upstream is exactly the recorded set and nothing else.
  The commits that added each architecture's files are the record of where they
  came from — repo, branch, commit and file list — and the commits after them
  are the complete list of what differs. `git log modules/<family>/flyc/`
  answers both questions.
* **Never copy FlyDSL's shared libraries into this directory** to unblock an
  import error. That is the fork the strategy exists to avoid. If a symbol is
  missing because it is branch-local and not in the released tree, add a
  fallback to the family's polyfill module; if the *path* is wrong, fix
  `python/flyc_bootstrap.py`.
* **Prefer authoring a compatibility module to rewriting import lines.** Where
  upstream has a small `sys.path` shim beside the kernels, replacing it with a
  local module of the same name and interface keeps every vendored file's
  imports verbatim, which means a re-sync diff for those files is empty.
  Rewriting the import lines instead costs the same edits on every re-sync.
* **Keep the tuning-policy modules free of any `flydsl` import.** The code
  generator calls them to resolve knobs and must never import flydsl; a flydsl
  import appearing in one breaks generation, not just the build.
* **Keep torch out of module scope.** The build virtual environment
  deliberately has no torch. A vendored file that imports it at module scope
  fails at generate time, far from the edit.
* **Leave a comment at every local edit.** `git log` is not where anyone reads
  a kernel, and a *deletion* has no line to notice.

## Cross-compiling with no GPU

`python/flyc_compile.py` compiles one vendored kernel to a `.hsaco` for a
named architecture, from a signature and a set of hints, with **no device
present and no kernel ever launched**. Two non-obvious things make that work,
both of which cost an investigation to find:

* **`ROCM_PATH` must name the directory containing `llvm/bin/ld.lld` at
  exactly that relative path.** Getting it wrong produces
  `error: lld invocation failed` and nothing else — no lld output, no mention
  of a path. `flyc_bootstrap.resolve_rocm_path()` exists to avoid rediscovering
  this, and hunts the usual locations, including the one inside an installed
  ROCm SDK wheel.
* **`flyc.compile()` cannot be used.** Its tail builds an `ExecutionEngine`,
  which needs HIP, so the documented GPU-less compile mode is unusable through
  that entry point. The driver invokes the traced `JitFunction` directly under
  `COMPILE_ONLY=1` and reads the artifact back out of it.

Nothing about this needs the target architecture to be present, or any
architecture at all — the build host is typically neither of the targets.

## Adding a FlyDSL backend for a new family

1. **Pin.** If this is the first FlyDSL backend, add the three
   `third_party/flydsl-*.txt` files. Otherwise reuse them: the pins are
   repository-wide, not per family.
2. **Vendor.** Copy the kernel sources into `modules/<family>/flyc/`, one
   commit per architecture, each recording repo, branch, commit and file list
   in its message. Do not copy interface or test modules — those are the
   language's own JIT entry points, and AOTriton's generated shim replaces
   them.
3. **Make them importable,** in a separate commit: the import rewrites, the
   compatibility module, the torch-laziness edits. This commit is the
   specification a re-syncer reapplies.
4. **Polyfill what the released pin lacks**, in `modules/<family>/flyc/`, with
   every entry preferring the installed package's own definition and falling
   back only when it is absent — so the module empties itself out as helpers
   land upstream.
5. **Describe the kernel** in `modules/<family>/aot/`: mark the builder with
   `@ati.flyc.kernel` and `@ati.flyc.hints`, declare the kernel arguments with
   their types and alignments, and have the builder return the FlyDSL module
   and the knobs it resolved.
6. **Compile one, by hand,** before wiring anything into the build:
   `AOTRITON_FLYDSL_KERNEL_ROOT=<flydsl source tree> python -m
   aotriton.flyc_compile <description> --kernel_name <name> --target <arch>
   --out_path <path without extension> --signature ... --hints ...` should
   produce a `.hsaco` whose ELF flags name the architecture you asked for. The
   environment variable is required, not optional: `flyc_bootstrap` raises
   without it and there is no default outside CMake. That check needs no GPU
   and no AOTriton build, and it is the cheapest place to find out that a
   kernel argument is misdeclared.
7. **Put development tools in `modules/<family>/flyc/devtools/`,** not in
   `modules/<family>/tests/`. The latter is reserved for tests of the built
   library; a harness that imports the vendored kernels directly and times them
   is a different thing and should not need an AOTriton build to run.

## Code generation: what the description turns into

Everything above stops at "a signature and a set of hints produce one
`.hsaco`". The build has to do that for every functional the backend serves,
package the results, and emit C++ that picks the right one at runtime. That is
the code generator's half, and it starts with one line on the operator.

### Becoming a backend

A flyc kernel reaches the generator by being an `@ati.backend` of an operator:

```python
@ati.backend(2, metro_fwd_flyc, 'flyc')
```

The index is an ABI — it is what a caller passes to force a backend — and the
name is the vocabulary a caller selects through. Both are published as generated
constants; see "Dispatch" below.

The backend object may be the flyc kernel itself, or a metro containing it. A
metro is the right answer whenever the backend needs a step FlyDSL has no
equivalent for: a metro's steps are bound by kind, so a launcher may mix
languages, and one metro can hold a FlyDSL kernel and a Triton one and run both
on one stream.

Linking then binds the operator into the flyc kernel, which is where its
functional space comes from. A flyc kernel that is not any operator's backend
fails linking with a message saying so, rather than surviving with no functional
space and failing later at whichever generated file first asks it a question.

### Narrowing the inherited space

`@ati.disable(when=...)` says which of the operator's functionals this backend
serves. Put every exclusion in one predicate — architecture included — rather
than splitting architecture into a separate declaration: when a functional
unexpectedly has no flyc kernel, one predicate is one place to look.

`@ati.cite('<op>.<backend>.<kernel>')` fills the gaps. Any argument the
description does not fully claim — a dtype variable, an operand with no strides
declared — is cloned from the cited kernel by name, so two backends reading the
same tensor cannot drift on what is in it.

A flyc kernel does not inherit the cited kernel's `tune`. It has no perf space
of its own to receive one.

### Kernel arguments that are not operands

`wires_to=` on `@ati.tensor` / `@ati.scalar` normally names an operator operand:
this kernel argument IS that operand, under another name. When the value has to
be COMPUTED instead, name a host-side function:

```python
@ati.scalar('num_seqlens', 'i32', wires_to=ati.context_helper('flyc_num_seqlens'))
```

That declares `int32_t flyc_num_seqlens() const;` on the generated context class
and leaves the body to the author, in `modules/<family>/csrc/`. It is the same
generator-declares / author-implements split `grid_calculator()` has always
used; `ati.context_helper` only lets a description declare more of them.

A helper takes no arguments, because the context already carries everything it
could need: the operator's params struct, and the GPU `lookup_optimal` was
called with. Its return type comes from the `@ati.scalar` type on the same line.

Three rules govern them:

* **A rename stays a rename.** If the argument IS an operand under another name,
  spell it `wires_to='<Operand>'`. A helper for it would be a function whose
  body is a field read.
* **A helper must be a pure function of the params struct and the current GPU,
  and helpers must be mutually independent.** They are all evaluated once, in
  declaration order, at the top of `lookup_optimal`, before anything else. A
  helper that read the selected kernel's knobs would get a default-constructed
  value; one that read another helper's result would get a zero. Neither raises.
* **The result is cached, not recomputed.** Each helper writes into a `mutable`
  scratch member on the context, and the launch-argument vector holds that
  member's address for the duration of the launch. A function's return value has
  no address, which is why the indirection exists.

### Redirecting a functional axis through a helper

A backend that compiles a subset of an operator's axis has a dispatch problem:
the caller's value is binned against the OPERATOR's ladder before a backend is
chosen, and the resulting digit can name a rung this backend never built.

Declaring the axis as a MARKER fixes it:

```python
@ati.scalar('BLOCK_DMODEL', options=[...],
            wires_to=ati.context_helper('flyc_block_dmodel'))
```

`options=` and an explicit type are mutually exclusive, so a marker can never be
mistaken for a real kernel argument; it is only ever found by axis name. What it
does is redirect `godel_number()` to read the value the helper computes instead
of the raw choice.

The generator emits, per architecture, the set of rungs this backend actually
compiled — derived from the same surviving-functional list that fills the
dispatch table, never hand-maintained — so the helper rounds against what was
built rather than against a list that can drift. An arch that compiles nothing
for this kernel gets an empty row, which is correct: the helper returns a
sentinel, the digit is rejected, and the launch fails with "unsupported" rather
than reaching a kernel that does not exist.

Any axis whose value another axis depends on must be derived from the SAME
decision, not re-derived. A "padded head" flag that disagreed with the rung
actually selected would be a silently wrong answer rather than a build error.

### What the build emits

Per flyc kernel, per functional it serves:

| artifact | what it is |
|---|---|
| `<family>/flyc.<kernel>.{h,cc}` | the C++ shim: context class, helper declarations, launch-argument builders, the dispatch table |
| `<family>/flytune.<kernel>/<functional>.cc` | one table entry, naming the single compiled image for that functional |
| a row in `Fly.compile` | the command line that compiles that image, run by ninja |
| an entry in the existing `.aks2` / flatzip rules | packaging, shared with every other backend |

`Fly.compile` is the only rule file a FlyDSL backend adds. Its images are
clustered and archived by the same rules Triton's are, so a FlyDSL backend costs
one new compile rule and no new packaging.

**Nothing in the generator ever invokes the FlyDSL compiler.** A description's
builder returns `(build, knobs)`: `knobs` is a plain dict, already resolved, and
`build` is a deferred callable that would construct the FlyDSL module. The
generator reads `knobs` and two plain strings off `build`, and discards it
without calling it. Only `flyc_compile`, run by ninja, calls it. Getting that
wrong would put every kernel compile inside `cmake` configure.

**A flyc kernel has no autotune LUT.** Every functional resolves to exactly one
image, and what distinguishes that image is the full knob set the description
chose. It travels in the entry name's `#P` section as a `k=v;k=v` string and is
read back at runtime by a small parser rather than a generated struct — so
adding a knob does not change any C++ type.

### Iterating without paying for Triton

`AOTRITON_DEBUG_SKIP_TRITON_KERNELS=1`, settable by `-D` or from the
environment, makes the generator emit the Triton C++ shims but no Triton image
rules. A configure+build then compiles only the other backends' kernels, which
turns "does this backend reach the image archive at all" from an hours-long
question into a short one.

It produces an INCOMPLETE image set and must never be shipped; the build says so
at configure time. Triton kernels that another backend's metro borrows are kept
regardless, by name, because skipping them would not give "Triton operators do
not work, everything else does" — it would give a metro with a missing step.

## Dispatch: choosing the backend at runtime

The backend index is public ABI. It is generated, per family, into
`<build>/include/aotriton/<family>/backends.h` from the same list that assigns
the internal enum, and installed alongside the hand-written headers:

```cpp
struct OpAttnFwdBackend {
  static constexpr int32_t kTriton = 0;
  static constexpr int32_t kAiter  = 1;
  static constexpr int32_t kFlyc   = 2;
  static constexpr int32_t Max     = 3;
};
```

Named from what `@ati.backend` declared, not from the internal enum name: the
internal name carries a prefix saying how the backend is assembled, which is the
generator's business and would change under a caller if a bare kernel were
re-shaped into a metro. Each struct also gets an X-macro, so a language binding
lists the two struct names and nothing else — the constant names expand from the
generator. That is deliberate: three constants in this repository have been
hand-copied into a second file and drifted.

`Max` is how many backends the LIBRARY was generated with. It is not how many
are available on the machine in hand — a backend may be architecture-restricted —
so a tool that probes backends should ask the runtime, not `Max`.

At runtime the dispatch order per launch is fixed:

1. `lookup_optimal(gpu)` captures the GPU, then evaluates every context helper
   once into the scratch members.
2. `godel_number()` runs, reading helper-wired axes from those scratch members
   and everything else off the params struct.
3. The table entry for `(arch, godel number)` selects the single compiled image,
   and the knob string is parsed once from it.
4. `launch()` builds the launch-argument vector — operands from the params
   struct, computed values from the scratch members — and invokes the kernel
   with the grid `grid_calculator()` returns.

Steps 1 and 2 are why helper independence matters: everything in step 1 runs
before anything in steps 2 to 4 exists.
