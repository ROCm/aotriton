# Basic Usage of CI Scripts

NOTE: ALL SCRIPTS REQUIRE **BASH**.

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
