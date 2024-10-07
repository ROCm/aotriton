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

* `hipcc` in `/opt/rocm/bin`, as a part of [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)
* `cmake`
* `ninja`
* `libzstd`
  - Common names are `libzstd-dev` or `libzstd-devel`.

## Generation

The kernel definition for generation is done in
[rules.py](https://github.com/ROCm/aotriton/blob/main/python/rules.py). Edits
to this file are needed for each new kernel, but it is extensible and generic.

Include files can be added in
[this](https://github.com/ROCm/aotriton/tree/main/include/aotriton) directory.

The final build output is an archive object file any new project may link
against.

The archive file and header files are installed in the path specified by
CMAKE_INSTALL_PREFIX.

## Kernel Support

Currently the first kernel supported is FlashAttention as based on the
[algorithm from Tri Dao](https://github.com/Dao-AILab/flash-attention).

## PyTorch Consumption & Compatibility

PyTorch [recently](https://github.com/pytorch/pytorch/pull/121561) expanded
AOTriton support for FlashAttention. AOTriton is consumed in PyTorch through
the [SDPA kernels](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/transformers/hip/flash_attn/flash_api.hip).
The Triton kernels and bundled archive are built at PyTorch [build time](https://github.com/pytorch/pytorch/blob/main/cmake/External/aotriton.cmake).

CAVEAT: As a fast moving target, AOTriton's FlashAttention API changes over
time. Hence, a specific PyTorch release is only compatible with a few versions
of AOTriton. The compatibility matrix is shown below

|  PyTorch Upstream     |           AOTriton Feature Release              |
|-----------------------|-------------------------------------------------|
|  2.2 and earlier      |               N/A, no support                   |
|        2.3            |                   0.4b                          |
|        2.4            |                   0.6b                          |
|        2.5            |                   0.7b                          |

ROCm's PyTorch release/<version> branch is slightly different from PyTorch
upstream and may support more recent version of AOTriton

|  PyTorch ROCm Fork    |           AOTriton Feature Release              |
|-----------------------|-------------------------------------------------|
|  2.2 and earlier      |               N/A, no support                   |
|        2.3            |                   0.4b                          |
|        2.4            |                   0.7b (backported)             |
|        2.5            |                   0.7b (once released)          |

### Point Release

AOTriton's point releases maintain ABI compatibility and can be used as drop-in
replacement of their corresponding feature releases.

For PyTorch main branch, check
[aotriton_version.txt](https://github.com/pytorch/pytorch/blob/main/.ci/docker/aotriton_version.txt).
The first line is the tag name, and the 4th line is the SHA-1 commit of
AOTriton.

Note: we are migrating away from `aotriton_version.txt` file. If this file disappears, check
[aotriton.cmake](https://github.com/pytorch/pytorch/blob/main/cmake/External/aotriton.cmake)
instead.
