# FlyDSL backends in AOTriton

FlyDSL is a third kernel language AOTriton can build a backend from, alongside
Triton (`@ati.source`) and aiter (`@ati.affine.*`). This document describes the
*mechanism* — how a FlyDSL backend is pinned, vendored and compiled — and is
deliberately kernel-family agnostic. Flash attention is used only as the worked
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
