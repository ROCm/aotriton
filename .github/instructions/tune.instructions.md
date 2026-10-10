---
applyTo: "v3python/tune/**/*.py","include/aotriton/cpp_tune.h"
---

Tuning code must use Eager Lazy tensor for APIs that need a LazyTensor to correctly measure the performance.

Assume single thread access for KernelControl/KernelFineControl objects. They
are not designed for multi-threading scenarios (but report issues if
multi-threading usage is detected).
