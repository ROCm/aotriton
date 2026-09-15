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

FlyDSL bundles its own LLVM/MLIR, so `third_party/flydsl-compiler.txt` pins a
*wheel* rather than a submodule: a FlyDSL carrying a different LLVM must be
rebuilt, not reconfigured. `third_party/flydsl-llvm.txt` names that LLVM -- a
third pin, independent of the other two, **expected to be empty almost always**
("empty" = no non-comment, non-blank line, so the explanatory comment survives).

| `flydsl-llvm.txt` | what a build does |
|---|---|
| empty | steady state: install the wheel `flydsl-compiler.txt` pins |
| non-empty, local wheel supplied (`-DAOTRITON_USE_LOCAL_FLYDSL_WHEEL`) | proceed with that wheel |
| non-empty, no local wheel | **fail at configure time** (`CMakeLists.txt`); a runtime-only build (`AOTRITON_NOIMAGE_MODE`) is exempt, it compiles no kernels |

Row three is deliberate: the bad LLVM miscompiles register spills, so kernels come
back *wrong rather than absent* and only numerical tests would notice. For a release
whose `flydsl-llvm.txt` is non-empty the sequence below is the only supported way to
build; emptying it pairs with bumping `flydsl-compiler.txt` once upstream LLVM is
fixed.

### The two scripts

```
build_llvm_tarball.sh --tarball_output_dir <dir>
                      [--llvm_origin <url>] [--llvm_commit <ref>]
                      [--python <X.Y>] [--jobs <N>] [--pat_environ <VAR>]

build_flydsl_wheel.sh --wheel_output_dir <dir> --flydsl_commit <ref>
                      --llvm_tarball <path> [--flydsl_origin <url>]
                      [--python <X.Y>] [--version_suffix <s>]
                      [--rocm <ver>] [--jobs <N>] [--pat_environ <VAR>]
```

Siblings, not caller and callee: the wheel build never builds LLVM, it is handed one
(`--llvm_tarball` is required), so a tarball can serve other consumers. Each prints
only its product's absolute path on stdout. `build_llvm_tarball.sh` defaults origin
and ref from `flydsl-llvm.txt`, and errors when that file is empty unless both
`--llvm_origin` and `--llvm_commit` are given. `--flydsl_commit` is a **git
ref** (`flydsl-kernel.txt` holds one, `v0.3.1`), not a pip requirement
(`flydsl-compiler.txt`, `flydsl==0.3.1`).

```bash
tarball=$(bash .ci/build_llvm_tarball.sh \
  --tarball_output_dir ../llvm-tarballs --python 3.13)
wheel=$(bash .ci/build_flydsl_wheel.sh \
  --wheel_output_dir ../flydsl-wheels \
  --flydsl_commit "$(cat third_party/flydsl-kernel.txt)" \
  --llvm_tarball "${tarball}" --python 3.13)
bash .ci/build-test.sh gfx950 <triton wheel> --flydsl_wheel "${wheel}"
```

For a release, `releasesuite-git-head.sh` runs both, in that order:

```bash
bash .ci/releasesuite-git-head.sh --image --flydsl_commit v0.3.1 \
  $HOME/aotriton-gh-release/release/0.14b/hashed
```

`--flydsl_commit` is optional there: a non-empty `flydsl-llvm.txt` selects the
source build anyway, taking the ref from `flydsl-compiler.txt` (`flydsl==0.3.1`
-> `v0.3.1`; any other shape is an error, not a guess). `--flydsl_origin` (fork,
or `file:///abs/path`, like `--triton_origin`) and `--llvm_tarball` (reuse a
tarball) are errors without an explicit `--flydsl_commit`, and `--flydsl_commit`
is an error on `--runtime`.

### Cost, caching and what the caches are keyed on

**The LLVM build takes about an hour**, so only a cache miss costs anything.

| artifact | name | keyed on |
|---|---|---|
| tarball | `llvm-<sha12>-<distro>-x64.tar.gz` | the LLVM commit only -- **not** Python: FlyDSL rebuilds MLIR's bindings from the tarball's sources per interpreter |
| wheel | `flydsl-<base>+git<sha8><version_suffix>.llvm<sha12>.p<patches>-cp<XY>-...whl` | FlyDSL commit, `--version_suffix`, LLVM commit, patch count, CPython ABI tag |

* That tarball name is the shape `.ci/triton-patch/docker-script-build.sh` already
  consumes, so one built here drops into `$HOME/.triton/llvm` unchanged.
* Both pins can name a moving ref (`aotriton/0.14b/rc0`, `v0.3.1`), so each is
  resolved to a SHA against the git mirror before the cache is consulted.
* The ABI tag is in the wheel key because flydsl wheels are ABI specific and CMake
  rejects a mismatched one; the LLVM commit is, because the wrong LLVM miscompiles
  rather than fails. Hence `--llvm_tarball` must be named `llvm-<sha12>-<...>.tar.gz`
  -- a name carrying no commit is refused rather than keyed on something invented.
* A hand-built wheel (`scripts/build_wheels.sh`) uses FlyDSL's
  `<base>.dev<commit count>` scheme, which a `--depth=1` checkout renders `.dev1` for
  every commit; these scripts set `FLYDSL_PACKAGE_VERSION_OVERRIDE` instead. Both work
  with `--flydsl_wheel`, but only ours can be a cache hit.

The only Docker volumes are the bare git mirrors `llvm-mirror` and `flydsl-mirror`
(`-<md5>` slug per non-default origin), never wiped, see `.ci/CLAUDE.md`. Both builds
run in `--tmpfs /scratch:exec` and keep nothing.

### Build environments

The LLVM half builds in `aotriton:base-py<X.Y>`, the FlyDSL half in
`aotriton:buildenv-rocm<ver>-py<X.Y>` (the release suite's `theRock.Dockerfile`,
`--rocm <ver>` selecting the TheRock version); both are built on demand. FlyDSL needs
ROCm only to link -- `lib/Runtime/ROCm/CMakeLists.txt` does `find_package(hip
REQUIRED)` under its only backend -- for a HIP launcher AOTriton never uses
(`COMPILE_ONLY=1`), so the ROCm version does not affect the kernels and **no GPU is
needed**.

`.ci/flydsl-patch/*.patch` is applied to the checkout first. Two today:

* `lib/Runtime/ROCm/CMakeLists.txt` searches `/opt/rocm*` for HIP, which a TheRock
  root is not (a site-packages directory named by `rocm-sdk path --root`); the
  patch prefers `ROCM_PATH`, keeping the glob as fallback.
* `pyproject.toml` asks for `nanobind>=2.0`, resolving to 3.0.1, while MLIR's
  `MLIRDetectPythonEnv.cmake` does `find_package(nanobind 2.9)` and nanobind treats
  a major bump as incompatible. The bound must live in `pyproject.toml`: pip's
  build isolation installs build requirements into its own overlay.

A patch that stops applying is a hard error, as is a FlyDSL whose
`scripts/build_llvm.sh` patches LLVM itself (added after v0.3.1) -- the tarball
here is built without it. Neither script needs a credential; both origins are
public. `--pat_environ <VAR>` names -- by name, never by value -- an environment
variable holding a token for a private origin.

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

