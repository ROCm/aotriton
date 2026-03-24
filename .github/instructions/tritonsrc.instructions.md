---
applyTo: "tritonsrc/*.py"
---

All python function decorated with `@triton.jit` are compiled by Triton compiler
and you must treat `range` usage as builtin iterators by Triton compiler,
which has the same interface as `tl.range`.

All code, if these code are not triton kernels, under `tritonsrc/` directory
are for Triton only testing on GPU systems. You must assume the code will be
run with CUDA/ROCm environment and stop considering what will happen if GPU is
not available.
