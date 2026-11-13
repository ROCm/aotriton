// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Build-time stand-in for libtorch_cpu.so. It is only linked (never
// installed, never on a run-time search path) so that pyaotriton records
// "NEEDED libtorch_cpu.so" without PyTorch being present at build time. At
// run time the dynamic loader matches that name to the libtorch_cpu that
// `import torch` has already loaded, which provides the torch_* and
// aoti_torch_* symbols declared in torch_stable_min.h.
extern "C" void aotriton_libtorch_cpu_stub() {}
