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
| releasesuite-git-head.sh   | Build AOTriton release tarballs                           |
| triton-wheel-build.sh      | Build AOTrton-compatible Triton Wheel from mainline       |
| triton-tester-build.sh     | Build AOTrtion with Triton Wheel from triton-wheel-build.sh    |
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

## Build AOTriton-Compatible Triton from Triton mainline

Certain Upstream Triton may need some patches to be able to build AOTrtion
(referred as AOTriton-compatible here). Patches are specified in `triton-patch/patch-*.sh` files.
See [triton-patch's README.md file](triton-patch/README.md) for more details of their format.

Example to build Triton Wheel
```bash
bash .ci/triton-wheel-build.sh 3.12 103aae3ca9da15039785b24070bfeee79bb1fc54
```

Syntax:
```
bash .ci/triton-wheel-build.sh <pyver> <upstream commit>
```

pyver is the python version planned to use.

The wheel will be stored in `dockerfile/input` directory

Note: do not use python 3.10. There is no python 3.10 package in almalinux8.

## Build Triton Tester Package with Upstream Compiler

Example:
```bash
bash .ci/triton-tester-build.sh \
  rocm/pytorch:rocm6.4.3_ubuntu24.04_py3.12_pytorch_release_2.6.0 \
  ~/aotriton-triton_tester/ \
  gfx1100 \
  103aae3ca9da15039785b24070bfeee79bb1fc54
# Output file
# ~/aotriton-triton_tester/aotriton-triton_tester-103aae3ca9da15039785b24070bfeee79bb1fc54-gfx1100.tar.gz
```

Syntax:
```bash
bash .ci/triton-tester-build.sh <docker image> <output directory> <arch> <upstream commit>
```

Docker image must use the same `pyver` specified in `.ci/triton-wheel-build.sh`
`upstream commit` must match `.ci/triton-wheel-build.sh`, otherwise the wheel cannot be located.

CAVEAT: `.ci/triton-tester-build.sh` may fail due to timeout when compiling
Triton kernels with Triton upstream. This should be considered as a regression.

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
