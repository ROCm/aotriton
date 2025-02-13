# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)
- Adam P. Goucher for simplified vector math

"""

import triton
import triton.language as tl

from fwd_kernel import attn_fwd
from bwd_preprocess import bwd_preprocess, bwd_preprocess_varlen
from bwd_split_kernel import bwd_kernel_dk_dv, bwd_kernel_dq
from bwd_kernel_fuse import bwd_kernel_fuse
from dropout_rng import (
    debug_fill_dropout_rng,
    debug_fill_dropout_rng_tensor,
    debug_simulate_encoded_softmax,
)
