## Build Instructions

```
mkdir build
cd build
export PKG_CONFIG_PATH="${PKG_CONFIG_PATH}:${CONDA_PREFIX}/lib/pkgconfig"
cmake .. -DCMAKE_INSTALL_PREFIX=./install_dir -DCMAKE_BUILD_TYPE=Release -DAOTRITON_GPU_BUILD_TIMEOUT=0 -G Ninja
# Use ccmake to tweak options
ninja install
```

The library and the header file can be found under `build/install_dir` afterwards.
You may ignore the `export PKG_CONFIG_PATH` part if you're not building with conda

Note: do not run `ninja` separately, due to the limit of the current build
system, `ninja install` will run the whole build process unconditionally.

### Prerequisites

* `python >= 3.10`
* `gcc >= 8` or `clang >= 10`
  - For Designated initializers, but only gcc >= 9 is tested.
  - The binary delivery is compiled with gcc13
* `cmake >= 3.26`
  - Only `cmake >= 3.30` is tested
* `ninja`
  - Only `ninja >= 1.11` is tested
* `liblzma`
  - Common names are `liblzma-dev` or `xz-devel`.

## Generation

The kernel definition for generation is done in
[rules.py](https://github.com/ROCm/aotriton/blob/main/python/rules.py). Edits
to this file are needed for each new kernel, but it is extensible and generic.

Include files can be added in
[this](https://github.com/ROCm/aotriton/tree/main/include/aotriton) directory.

The final build output is an archive object file any new project may link
against.

The archive file and header files are installed in the path specified by
`CMAKE_INSTALL_PREFIX`.

## Kernel Support

Currently the first kernel supported is FlashAttention as based on the
[algorithm from Tri Dao](https://github.com/Dao-AILab/flash-attention).

## PyTorch Consumption & Compatibility

AOTriton is consumed in PyTorch through
the [SDPA kernels](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/transformers/hip/flash_attn/aot/mha_all_aot.hip).
The precompiled binaries will be downloaded and shipped with PyTorch during [builds](https://github.com/pytorch/pytorch/blob/main/cmake/External/aotriton.cmake).

CAVEAT: As a fast moving target, AOTriton's FlashAttention API changes over
time. Hence, a specific PyTorch release is only compatible with a few versions
of AOTriton. The compatibility matrix is shown below

|  PyTorch Upstream     |           AOTriton Feature Release              |
|-----------------------|-------------------------------------------------|
|  2.2 and earlier      |               N/A, no support                   |
|        2.3            |                   0.4b                          |
|        2.4            |                   0.6b                          |
|        2.5            |                   0.7b, 0.8b<sup>(1)</sup>      |
|        2.6            |                   0.8b<sup>(2)</sup>            |
|        2.7            |                   0.9b<sup>(3)</sup>            |

1. 0.8b's API is backward compatible with 0.7b, but the packaging scheme
   has changed drastically.
2. PyTorch 2.6 requires some 0.8b-only features. Hence even if PyTorch 2.6
   can compile with 0.7b due to API compatibility, the end product will
   suffer from runtime errors.
3. To be specific, it is shipped with 0.9.2b. 0.9b and 0.9.1b should not be
   used in order to avoid linking issues, and confusion about version strings.

ROCm's PyTorch release/\<version\> branch is slightly different from PyTorch
upstream and may support more recent version of AOTriton

|  PyTorch ROCm Fork    |           AOTriton Feature Release              |
|-----------------------|-------------------------------------------------|
|  2.2 and earlier      |               N/A, no support                   |
|        2.3            |                   0.4b                          |
|        2.4            |                   0.7b (backported)             |
|        2.5            |                   0.8b (backported)             |
|        2.6            |                   0.8b (backported)             |
|        2.7            |                   0.9b (once released)          |

### Point Release

AOTriton's point releases maintain ABI compatibility and can be used as drop-in
replacement of their corresponding feature releases.
