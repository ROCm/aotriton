# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from ._common import FlashKernel, get_possible_choices, select_pattern, BinningLessOrEqual, BinningExact
from .attn_fwd import attn_fwd
from .op_attn_bwd import OpAttnBwd
match_op = lambda aname : get_possible_choices(OpAttnBwd, aname)


# TODO: rename stride_on to stride_ok and stride_don to stride_dok in tritonsrc
class bwd_preprocess(FlashKernel):
    SHARED_IFACE = OpAttnBwd
    ARGUMENTS = [
        'Out', 'DO',
        'D',
        'stride_oz', 'stride_oh', 'stride_om', 'stride_ok',
        'stride_doz', 'stride_doh', 'stride_dom', 'stride_dok',
        'max_seqlen_q',
        'head_dim',
        'BLOCK_M', # tl.constexpr starts here
        'BLOCK_DMODEL',  # TODO: Rename the triton kernel
        'PADDED_HEAD',
    ]
    PERF_CHOICES = {
        frozenset(['BLOCK_M']) : [128], # TODO: All possible values?
    }
    DEFAULT_NUM_WARPS=4
    DEFAULT_NUM_STAGES=1
    NAME = 'bwd_preprocess'

    AUTOTUNE_KEYS = { }
    PARTIALLY_TUNED_FUNCTIONALS = {
        'PADDED_HEAD': False,
    }
    DOWNGRADER = []

class bwd_preprocess_varlen(FlashKernel):
    SHARED_IFACE = OpAttnBwd
    ARGUMENTS = [
        'Out', 'DO',
        'D',
        'stride_oz', 'stride_oh', 'stride_om', 'stride_ok',
        'stride_doz', 'stride_doh', 'stride_dom', 'stride_dok',
        'cu_seqlens_q',
        'max_seqlen_q',
        'head_dim',
        'BLOCK_M', # tl.constexpr starts here
        'BLOCK_DMODEL',
        'PADDED_HEAD',
    ]
    PERF_CHOICES = {
        frozenset(['BLOCK_M']) : [128], # TODO: All possible values?
    }
    DEFAULT_NUM_WARPS=4
    DEFAULT_NUM_STAGES=1
    NAME = 'bwd_preprocess_varlen'

    AUTOTUNE_KEYS = { }
    PARTIALLY_TUNED_FUNCTIONALS = {
        'PADDED_HEAD': False,
    }
    DOWNGRADER = []
