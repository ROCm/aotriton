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
| triton-tester-build.sh     | (Under redesign) Build AOTriton with a Triton mainline wheel   |
| triton-tester-run.sh       | Run tests to check correctness of Triton mainline         |

## Naming Scheme

* `build-*.sh`: script that runs directly, either manually by user inputs, or
  invoked indirectly by another script.
* `run-*.sh`: run some task given a success `build-*.sh`, within the same environment.
* `*-build.sh`: build something inside a docker environment created by this
  script. The actual build task should be done by `build-*.sh` script
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

To build wheels manually:
```bash
bash .ci/build_triton_wheels.sh \
  --wheel_output_dir <output_dir> \
  --version_suffix "+aotriton0.12" \
  <hash1> [<hash2> ...]
```

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

# Example Usages of AOTrtion Tester for Triton Mainline

> **Under redesign.** `triton-wheel-build.sh` has been removed and
> `triton-tester-build.sh` is a placeholder while this flow is reworked to use
> the git mirror caching in `common-git-cache.sh`. The
> `triton-tester-run.sh` step below still applies to an already-built tester
> package.

## Run Triton Tester in Target System

Example
```
# Inside GPU-enabled docker container from image
# rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0
git clone https://github.com/ROCm/aotriton.git
cd aotriton
pip install -r requirements-dev.txt
# Suppose the tester package is put under /
tar xf /aotriton-triton_tester-103aae3ca9da15039785b24070bfeee79bb1fc54-gfx1100.tar.gz
bash .ci/triton-tester-run.sh 0
tail triton_tester_pass0.out  # Check pytest output
```

Syntax:
```bash
bash .ci/triton-tester-run.sh <Pass Number>
```
