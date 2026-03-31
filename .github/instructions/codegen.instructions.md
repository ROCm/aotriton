---
applyTo: "v3python/codegen/**/*.py","v3python/codegen/**/*.h","v3python/codegen/**/*.cc"
---

Conditional skips are only needed for MetroKernel.
The `lookup_optimal` and `launch` functions have built-in conditional skipping
mechanism and hence does not need to check the condition before calling them.
However, a sanity check should be performed over classes called by MetroKernel
to confirm conditional skipping is implemented, but such sanity check only need
to be performed on existing instances, not hypothetical classes that may be
created in the future.

Slim Affine kernel will always be the sole "kernel" for some backend of certain
operator, because Slim Affine kernel warps against well-established API(s)
rather than GPU kernels. Consequently Slim Affine kernels do not need skipping
mechanism, because they are called by operator, not MetroKernel (Should also
sanity check about this).
