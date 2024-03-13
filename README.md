## Build Instructions

```
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=./install_dir
# Use ccmake to tweak options
make install
```

The library and the header file can be found under `build/install_dir` afterwards.

Note: do not run `make` separately, due to the limit of the current build
system, `make install` will run the whole build process unconditionally.

### Prerequisites

* `hipcc` in `/opt/rocm/bin`, as a part of [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)
## Generation
The kernel definition for generation is done in [rules.py](https://github.com/ROCm/aotriton/blob/main/python/rules.py). Edits to this file are needed for each new kernel, but it is extensible and generic.

Include files can be added in [this](https://github.com/ROCm/aotriton/tree/main/include/aotriton) directory.

The final build output is a shared object file any new project may link against.

## Kernel Support
Currently the first kernel supported is FlashAttention as based on the [algorithm from Tri Dao](https://github.com/Dao-AILab/flash-attention).

## PyTorch Consumption
PyTorch [recently](https://github.com/pytorch/pytorch/pull/121561) expanded AOTriton support for FlashAttention. AOTriton is consumed in PyTorch through the [SDPA kernels](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/transformers/hip/flash_attn/flash_api.hip). The Triton kernels and bundled SO are built at PyTorch [build time](https://github.com/pytorch/pytorch/blob/main/cmake/External/aotriton.cmake).
