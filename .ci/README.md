# Basic Usage of CI Scripts

NOTE: ALL SCRIPTS REQUIRE **BASH**.

|  Script Name               |              Task                                         |
| -------------------------- | --------------------------------------------------------- |
| build-tune.sh              | Build Tuning Version of AOTriton                          |
| build-test.sh              | Build Testing Version of AOTriton                         |
| run-test.sh                | Run full tests against AOTriton built by build-test.sh    |
| run-ci-test.sh             | Run run-test.sh with `USE_ADIFFS_TXT` set                 |
| build-for-torch.sh         | Build AOTriton for PyTorch                                |
| torch-build.sh             | Build PyTorch with AOTriton built by build-for-torch.sh   |
| releasesuite-git-head.sh   | Build AOTriton release tarballs (calls build_triton_wheels.sh first) |
| build_triton_wheels.sh     | Build and cache Triton wheels from commit hashes          |
| build_llvm_tarball.sh      | Build and cache an LLVM/MLIR tarball for FlyDSL            |
| build_flydsl_wheel.sh      | Build and cache a FlyDSL compiler wheel against that LLVM  |
| triton-tester-build.sh     | (Under redesign) Build AOTriton with a Triton mainline wheel   |

## Naming Scheme

* `build-*.sh`: script that runs directly, either manually by user inputs, or
  invoked indirectly by another script.
* `run-*.sh`: run some task given a success `build-*.sh`, within the same environment.
* `*-build.sh`: build something inside a docker environment created by this
  script. The actual build task should be done by `build-*.sh` script
* `runc-*.sh`: the half of a two-file build that runs *inside* a container.
  Its partner on the host handles option parsing, caching, git mirrors and
  `docker run`; the `runc-` half does the actual work and takes plain
  positional arguments, so a caller with no Docker (`.tune`'s worker
  container) can run it directly.
* `common-*.sh`: a "library" script that could be sourced by other scripts.
* `docker-script-*`: a script prepare environment inside a docker container for
  direct scripts like `build-*.sh` or `run-*.sh`
  - It is intentionally not to call the build commands directly, to decouple
    from build command changes accross different commits.

`releasesuite-git-head.sh`, as the mono-entry to make AOTriton releases, does
not fall into any category and consequently does not follow any naming scheme above.

# Example Usages of AOTrtion Tests

## Build for Tuning

tl;dr example to build for target gfx950:
``` bash
bash .ci/build-tune.sh gfx950
```

Syntax:
```
bash .ci/build-tune.sh <target arch>
```

This script will create a build directory
`build-${aotriton_major}.${aotriton_minor}-tune-${target_arch}` under the
parent directory of `.ci`

## Build for Testing

Similar to `build-tune.sh`, but replace "tune" with "test"

tl;dr example to build for target gfx950:
``` bash
bash .ci/build-test.sh gfx950
```

This script will create a build directory
`build-${aotriton_major}.${aotriton_minor}-test-${target_arch}` under the
parent directory of `.ci`

## AOTriton Self Test

`aotriton-self-test.sh` runs two CPU-only unit-test suites: the ATI code
generator's own tests (`python/test`) and the `pytest-gpu-lease` plugin's own
tests (`python/pytest-gpu-lease/tests`). Neither needs a GPU, a ROCm
install, or a built AOTriton library, so this is the fastest pre-flight
check available and the one to run first.

tl;dr:
``` bash
bash .ci/aotriton-self-test.sh
# or
make -C .ci check-self-test
```

It creates its own disposable venv (`build-unittest-venv/` by default,
override with `AOTRITON_UNITTEST_VENV`) rather than reusing the CMake build
venv, which has `aotriton` but no `pytest`, or the user's own env, which
typically has `pytest` but no `aotriton`.

This is a different, narrower pass than `run-test.sh` below: `run-test.sh`
exercises the real flash-attention GPU kernels against a built library, needs
a GPU and ROCm, and takes much longer. `aotriton-self-test.sh` never touches
a GPU or a built library at all.

## Run Tests

tl;dr example to perform full tests before release

``` bash
bash .ci/run-test.sh 0 2 v3
```

Syntax:
```
bash .ci/run-test <Pass Number> <Test level> <Backend>
```

* Pass number is a number added to output files to avoid overwritting existing
  result when run the script manually
* Test level is a number in 0/1/2, each level adds more tests to the suite.
* Backend is a string value within split/fused/aiter/v3, to test different backends.

## Test With PyTorch

Step 1: build library for pytorch

``` bash
cd aotriton
bash .ci/build-for-torch.sh
```

Step 2: build pytoch with aotriton built in Step 1

``` bash
cd pytorch
bahs ../aotriton/.ci/torch-build.sh
```

## Triton Wheel Pre-Build

`releasesuite-git-head.sh` always builds Triton wheels before starting the
AOTriton build. The wheels are cached in `<output_dir>/.cache/wheels/` and
reused on subsequent runs (skipped if a matching wheel is already present).

A `triton-mirror` Docker volume is maintained automatically as a bare clone of
`https://github.com/ROCm/triton`. Each run fetches all remote branches to
ensure the requested commit hashes are reachable, then performs a shallow clone
per hash into a tmpfs for the actual wheel build.

The `--yaml` flag is still optional. When provided, its `.venvs` hashes are
built in addition to the embedded `third_party/triton` submodule (unless the
YAML supplies `.venvs.default`, which replaces the submodule hash).

A `.venvs` entry may give `hash`/`origin` as a `{hash, origin}` map instead of
a bare hash string, to build from a commit that lives in a different Triton
origin (e.g. a private fork). An optional third field, `pat_environ`, names
an environment variable holding a GitHub PAT. See `.ci/AltWheelExample.yaml`
for a full example of this (source) format -- CMake itself only ever reads
the resolved form (plain wheel paths) documented at `docs/AltWheelExample.yaml`.

```yaml
venvs:
  private_fork:
    hash: <sha>
    origin: https://github.com/some-org/private-triton
    pat_environ: GITHUB_TOKEN
```

Some build processes expect the token under a specific variable name (e.g.
`GITHUB_EMU_TOKEN`) rather than a fixed default; `pat_environ` lets you
specify that name and passes it through as-is into the Triton build, so the
same PAT authenticates both the git clone and any artifact downloads from
private GitHub instances during the build. If `pat_environ` is omitted, no
PAT is used.

To build wheels manually:
```bash
bash .ci/build_triton_wheels.sh \
  --wheel_output_dir <output_dir> \
  --version_suffix "+aotriton0.12" \
  <hash1> [<hash2> ...]
```

## FlyDSL Compiler Wheel From Source

### Why this exists

FlyDSL bundles its own LLVM/MLIR rather than linking a system one, which is why
`third_party/flydsl-compiler.txt` pins a *wheel* and not a submodule: its
`setup.py` needs a prebuilt bundled MLIR to build against. A FlyDSL carrying a
different LLVM therefore has to be rebuilt, not reconfigured.

`third_party/flydsl-llvm.txt` is where AOTriton says which LLVM that is. It is
a third pin, independent of the compiler and kernel pins, and **it is expected
to be empty almost always**:

| `flydsl-llvm.txt` | what a build does |
|---|---|
| empty (just its own comment line) | normal. Install the wheel `flydsl-compiler.txt` pins and proceed. This is the steady state |
| non-empty, and a local wheel is supplied | proceed with the supplied wheel |
| non-empty, and no local wheel supplied | **fail at configure time** |

"Empty" means no non-comment, non-blank line, so emptying the file at release
time still leaves the comment that explains what it is for. Emptying it is the
task that pairs with bumping `flydsl-compiler.txt` once upstream LLVM is fixed.

The third row is deliberate. A wheel built against a spill-miscompiling LLVM
produces kernels that are *wrong rather than absent*, at exactly the head
dimensions that spill, and no build-time check would notice — only numerical
test failures much later would. Failing at configure time turns a silent
miscompile into a message, and the message names `build_flydsl_wheel.sh`.

So for a release whose `flydsl-llvm.txt` is non-empty, the sequence below is
not a convenience. It is the only supported way to build.

### The two scripts

```
build_llvm_tarball.sh --tarball_output_dir <dir>
                      [--llvm_origin <url>] [--llvm_commit <ref>]
                      [--python <X.Y>] [--jobs <N>] [--pat_environ <VAR>]

build_flydsl_wheel.sh --wheel_output_dir <dir> --flydsl_commit <ref>
                      [--flydsl_origin <url>]
                      [--llvm_tarball <path> | --llvm_origin <url> --llvm_commit <ref>]
                      [--python <X.Y>] [--version_suffix <s>]
                      [--rocm <ver>] [--jobs <N>] [--pat_environ <VAR>]
```

Each prints the absolute path of its product on stdout and nothing else, so
they chain with a plain command substitution. `build_flydsl_wheel.sh` calls
`build_llvm_tarball.sh` itself when no `--llvm_tarball` is given, putting
tarballs in a `llvm-tarballs/` directory beside the wheel cache.

`--flydsl_commit` takes a **git ref** in the FlyDSL repository — a tag, branch
or SHA — not a pip requirement. `third_party/flydsl-kernel.txt` holds one
(`v0.3.1`); `third_party/flydsl-compiler.txt` holds a pip requirement
(`flydsl==0.3.1`) and is the pin for the *released wheel* path, which is the
one this script replaces.

Typical use — the wheel that `build-test.sh --flydsl_wheel` wants:

```bash
wheel=$(bash .ci/build_flydsl_wheel.sh \
  --wheel_output_dir ../flydsl-wheels \
  --flydsl_commit "$(cat third_party/flydsl-kernel.txt)" \
  --python 3.13)
bash .ci/build-test.sh gfx950 <triton wheel> --flydsl_wheel "${wheel}"
```

For a release, do not run them by hand — `releasesuite-git-head.sh` does it:

```bash
bash .ci/releasesuite-git-head.sh --image \
  --flydsl_commit v0.3.1 \
  $HOME/aotriton-gh-release/release/0.14b/hashed
```

`--flydsl_origin <url>` overrides the FlyDSL origin (a fork, or a local
checkout via `file:///abs/path`), exactly like `--triton_origin`.
`--llvm_tarball <path>` reuses a tarball you already have. Both are errors
without `--flydsl_commit`, which is the switch that turns the whole path on;
a release with no FlyDSL flags installs the pinned wheel and never builds
anything here.

### Cost, caching and what the caches are keyed on

**The LLVM build takes about an hour.** Everything about these scripts is
arranged so that a cache miss is the only thing that costs anything:

* The tarball is `llvm-<sha8>-<distro>-x64.tar.gz`, the same filename shape
  `.ci/triton-patch/docker-script-build.sh` already consumes, so one built here
  drops into `$HOME/.triton/llvm` unchanged. It is keyed on the LLVM commit
  only — **not** on the Python version, because FlyDSL rebuilds MLIR's Python
  bindings from this tarball's sources for each target interpreter.
* The pin normally names a *branch* (`aotriton/0.14b/rc0` advances as the RC
  does), so the ref is resolved to a SHA against the git mirror before the
  cache is consulted. Naming a tarball after a branch would let two different
  builds collide under one filename.
* The wheel cache **is** keyed on the CPython ABI tag, because flydsl wheels
  are ABI specific (`flydsl-…-cp313-cp313-linux_x86_64.whl`) and CMake rejects
  one whose tag does not match the build venv. A wheel cached for a different
  Python is not a cache hit. The wheel's version carries the FlyDSL commit as
  `+git<sha8>`, the same way Triton's does.
* A cached wheel is found before the LLVM tarball is even looked for, so a
  re-run of an already-built configuration does no work at all.

**The filename shape changed, and that is not a broken cache.** A wheel built
by hand — `bash scripts/build_wheels.sh` in a FlyDSL checkout — is named from
FlyDSL's own default version scheme, `<base>.dev<commit count>`, giving e.g.
`flydsl-0.3.1.dev1129-cp313-cp313-linux_x86_64.whl`. `build_flydsl_wheel.sh`
does not use that scheme, because a `--depth=1` checkout has a commit count of
1 and renders every commit as `.dev1`. It sets FlyDSL's own
`FLYDSL_PACKAGE_VERSION_OVERRIDE` instead, producing
`flydsl-0.3.1+git421935cc.aotriton0.14-cp313-cp313-linux_x86_64.whl`. Both are
the same kind of artifact and either is accepted by `--flydsl_wheel`; only the
second identifies which FlyDSL commit is inside it, which is what makes it
cacheable. A hand-built wheel already sitting in the output directory will
therefore *not* be seen as a cache hit.

Docker volumes maintained automatically, all of them harmless local caches
that are never wiped (see `.ci/CLAUDE.md`):

| volume | holds |
|---|---|
| `llvm-mirror`, `flydsl-mirror` | bare git mirrors; a per-origin `-<md5>` slug is used for a non-default origin |
| `aotriton-llvm-build` | the LLVM checkout, build and install trees |
| `aotriton-flydsl-build` | the extracted LLVM prefix, the FlyDSL checkout and its build tree |

The two build volumes are named volumes rather than the `--tmpfs /scratch` the
Triton wheel build uses: an LLVM tree with assertions is tens of gigabytes, and
RAM-backed scratch that large cannot be assumed. Keeping them also makes a
re-spin of the same RC an incremental rebuild. Both scripts key their
subdirectories by commit, so two different inputs never share a `CMakeCache`.

### Build environments

The LLVM half builds in `aotriton:base-py<X.Y>`, built on demand like every
other image here. The FlyDSL half needs more: FlyDSL's
`lib/Runtime/ROCm/CMakeLists.txt` does an unconditional
`find_package(hip REQUIRED CONFIG PATHS /opt/rocm*)` under its only backend, so
it cannot configure without ROCm. `buildenv-flydsl.Dockerfile` adds the ROCm
dev packages on top of the base image (`--rocm <ver>` selects the version,
default 7.2.4). That is a *link* requirement for a HIP launcher AOTriton never
uses — it compiles under `COMPILE_ONLY=1` — so the ROCm version has no bearing
on the kernels the wheel produces, and **no GPU is needed** for any of this.

Neither script requires a credential: the FlyDSL compiler repo
(`https://github.com/ROCm/FlyDSL`) and `https://github.com/ROCm/llvm-project`
are both public. `--pat_environ <VAR>` names — by name, never by value — an
environment variable holding a token if you point one at a private origin;
it reuses `common-git-cache.sh`'s existing plumbing rather than a second
mechanism.

## Release the Package

Case 1: Build Both Runtime and Image

``` bash
bash .ci/releasesuite-git-head.sh
```

Case 2: Build Image Only

``` bash
bash .ci/releasesuite-git-head.sh --image
```

Case 3: Build Runtime Only

``` bash
bash .ci/releasesuite-git-head.sh --runtime
```

# AOTriton Tester for Triton Mainline

> **Retired.** `triton-tester-run.sh` and `modules/flash/tests/triton_tester.py`
> have been removed: the tester pinned the V2 API, which no longer exists.
> `run-test.sh` covers the same ground against a build made with a mainline
> Triton wheel.

