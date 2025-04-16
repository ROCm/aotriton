# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from .attn_fwd import attn_fwd
from .bwd_preprocess import bwd_preprocess, bwd_preprocess_varlen
from .bwd_kernel_dk_dv import bwd_kernel_dk_dv
from .bwd_kernel_dq import bwd_kernel_dq
from .bwd_kernel_fuse import bwd_kernel_fuse
from .debug_fill_dropout_rng import debug_fill_dropout_rng, debug_fill_dropout_rng_tensor
from .debug_simulate_encoded_softmax import debug_simulate_encoded_softmax

SOURCE_FILE = 'tritonsrc/flash.py'
kernels = [
    bwd_preprocess('bwd_preprocess', SOURCE_FILE),
    bwd_preprocess_varlen('bwd_preprocess_varlen', SOURCE_FILE),
    attn_fwd('attn_fwd', SOURCE_FILE),
    bwd_kernel_dk_dv('bwd_kernel_dk_dv', SOURCE_FILE),
    bwd_kernel_dq('bwd_kernel_dq', SOURCE_FILE),
    bwd_kernel_fuse('bwd_kernel_fuse', SOURCE_FILE),
    debug_fill_dropout_rng('debug_fill_dropout_rng', SOURCE_FILE),
    debug_fill_dropout_rng_tensor('debug_fill_dropout_rng_tensor', SOURCE_FILE),
    debug_simulate_encoded_softmax('debug_simulate_encoded_softmax', SOURCE_FILE),
]
