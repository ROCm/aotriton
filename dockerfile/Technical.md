# Technical documentation about Manylinux-2-28 builds from AlmaLinux8

## Build Process

The build process can be partitions into two steps

1. Build a base docker image which is suitable for compiling AOTriton
2. Launch a container with this base docker image for actual builds of AOTriton

### Base Docker Image

1. GCC 13 is use to avoid `undefined references to std::__glibcxx_assert_fail`
    bug. Details: https://bugzilla.redhat.com/show_bug.cgi?id=2060755
    + GCC 9/11 is tested but not working.
2. Python 3.11 is used and selected as default because ROCm stack will install
   Python 3.11 eventually, and then AOTriton build system will pick this latest version...
3. AOTriton 0.7 only needs `zstd`, but this will be replaced by `liblzma` in
   AOTriton 0.8 for (much) smaller binary size.
4. To keep the image smaller, Only `hip` Usecase is installed.
5. `hipcc` and `rocm-device-libs` install the remaining essential ROCm packages.

### Container to Build AOTriton

1. For now `TRITON_HASH` is hardcoded since there is no easy way to read the file content from git submodule.
2. AOTriton cannot be built with `hipcc`+GCC 13 because `hipcc` invokes Clang
   shipped by ROCm, but this Clang cannot find `libstdc++-13` provided by gcc-toolset-13.
3. The solution to the problem is to introduce a patching system (in
   `input/install_aotriton.sh`) which patches 0.7.x AOTriton source code to let
   them use GCC, and link to hip packages.

## Various decisions

1. AlmaLinux base docker image is used for developers who prefer bare-metal over containers.
   In this case, the `manylinux_2_28.Dockerfile` can be used as guides to build AOTriton
   on bare-metal systems.
2. Conda is not used because Conda as of today (Oct 2024) is still on glibc_2_17

## Development Notes

### Support newer AOTriton

Edit the `case "${AOTRITON_GIT_NAME}" in` part of `build.sh` and add Triton
hashes for newer AOTriton.
