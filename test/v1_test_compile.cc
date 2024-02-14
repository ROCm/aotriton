// Copyright © 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include "attn_fwd.h"

int main()
{
  __fp16 q[4][16][1024][16];
  __fp16 k[4][16][1024][16];
  __fp16 v[4][16][1024][16];
  float M[4 * 16][1024];
  __fp16 Out[4][16][1024][16];
  constexpr int BLOCK_M = 128;
  constexpr int BLOCK_DMODEL = 16;
  constexpr int BLOCK_N = 64;
  aotriton::attn_fwd<1, BLOCK_M, BLOCK_DMODEL, BLOCK_N, true> op;
  dim3 grid { 1024 / BLOCK_M, 4 * 16, 1};
  dim3 block{ 256, 1, 1};
  op(grid, block,
     (const __fp16*)q,
     (const __fp16*)k,
     (const __fp16*)v,
     1.3,
     (const float*)M,
     (const __fp16*)Out,
     16 * 1024 * 16, 1024 * 16, 16, 1, // stride_q?
     16 * 1024 * 16, 1024 * 16, 16, 1, // stride_k?
     16 * 1024 * 16, 1024 * 16, 16, 1, // stride_v?
     16 * 1024 * 16, 1024 * 16, 16, 1, // stride_o?
     4,     // Z = q.shape(0)
     16,    // H = q.shape(1)
     1024,  // N_CTX=q.shape[2]
     nullptr);
  return 0;
}
