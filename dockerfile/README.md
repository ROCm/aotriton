# Build Manylinux-2-28 Compatible AOTriton From Source

Technical details of this build system can be found at Technical.md

## Use different ROCm releases

Set env var `AMDGPU_INSTALLER` to URL of RHEL 8.10 `amdgpu-install` package.
The latest package can be found at
https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/amdgpu-install.html#red-hat-enterprise-linux

## TL;DR (Use 0.10b as an example)

Throughout this text, `aotriton/` always refers to the root directory of
AOTriton source code.

```
cd aotriton/dockerfile
bash build.sh input tmpfs output 0.10b "gfx90a;gfx942;gfx950;gfx1201"
```

### For Release Before 0.10b

``` bash
cd aotriton/dockerfile
bash build.sh input tmpfs output 0.7.1b "MI300X;MI200;Navi31"
```

To check its usage, run `bash build.sh` without any arguments.

Note: as of this version, the `build.sh` only supports AOTriton 0.7.x releases.
Support of future releases will be added later (notably the Triton hash problem).

## Argument: input

This specifies input directory, which must contain all files and subdirectories
under `aotriton/dockerfile/input/`, which are critical files for the build, including:

1. install scripts `install*.sh`
2. Patches to build on AlmaLinux 8 (under `patch-*`)

You may use the `aotriton/dockerfile/input/` directly, or copy the whole
directory to some other place and start build from there.

### Cached Triton

It is highly recommended to cache the LLVM/MLIR package used by Triton in the `input` directory.
It can be downloaded from
```
https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-657ec732-almalinux-x64.tar.gz
```

Here `657ec732` is the Hash of this MLIR package and will change depending on
the Triton compiler used by the AOTriton. The pinned version can be found at
`aotriton/third_party/triton/cmake/llvm-hash.txt`.

*What If I don't cache this MLIR package*

The build script of Triton will download this ~1GiB tarball for you, but will take
undetermined amount of time (depending on your Internet speed). In addition,
there is *NO progress bar* shown during this process, and the build process
would simply fail if the downloading failed.

## Argument: tmpfs

Can be any directory to store temporary build files.
Use `tmpfs` to mount `tmpfs` for better performance.
`tmpfs` is only recommended for system with large physical memories (> 128GB),
because AOTriton build needs ~15GiB disk space per architecture.

## Argument: output

Output directory. The compiled aotriton tarball will be put here.
