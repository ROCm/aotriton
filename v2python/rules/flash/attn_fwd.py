# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import itertools
from ._common import FlashKernel, select_pattern, BinningLessOrEqual, BinningExact, Config

class attn_fwd(FlashKernel):
    ARGUMENTS = [
        'Q', 'K', 'V', 'B', 'sm_scale', 'M', 'Out',
        'stride_qz', 'stride_qh', 'stride_qm', 'stride_qk',
        'stride_kz', 'stride_kh', 'stride_kn', 'stride_kk',
        'stride_vz', 'stride_vh', 'stride_vk', 'stride_vn',
        'stride_bz', 'stride_bh', 'stride_bm', 'stride_bn',
        'stride_oz', 'stride_oh', 'stride_om', 'stride_on',
        'num_head_q',
        'num_head_k',
        'cu_seqlens_q',
        'cu_seqlens_k',
        'num_seqlens',
        'max_seqlen_q',
        'max_seqlen_k',
        'head_dim',
        'dropout_p',
        'philox_seed_ptr',
        'philox_offset1',
        'philox_offset2',
        'philox_seed_output',
        'philox_offset_output',
        'encoded_softmax',
        'CAUSAL', # tl.constexpr starts here
        'BLOCK_M',
        'BLOCK_DMODEL',
        'BLOCK_N',
        'pre_load_v',
        'ENABLE_DROPOUT',
        'RETURN_ENCODED_SOFTMAX',
        'PADDED_HEAD',
        'BIAS_TYPE',
    ]
    TENSOR_STRIDE_INPUTS = {
        'Q' : select_pattern(ARGUMENTS, 'stride_q'),
        'K' : select_pattern(ARGUMENTS, 'stride_k'),
        'V' : select_pattern(ARGUMENTS, 'stride_v'),
        'B' : select_pattern(ARGUMENTS, 'stride_b'),
        'Out' : select_pattern(ARGUMENTS, 'stride_o'),
    }
    TYPE_CHOICES = {
        frozenset(['Q', 'K', 'V', 'B', 'Out', 'encoded_softmax']) : ['*fp16:16', '*bf16:16', '*fp32:16'],
        frozenset(['sm_scale']) : ['fp32'],
        frozenset(['M']) : ['*fp32:16'],
        frozenset(['cu_seqlens_q', 'cu_seqlens_k']) : ['*i32:16'],
        frozenset(['num_head_q', 'num_head_k', 'num_seqlens', 'max_seqlen_q', 'max_seqlen_k']) : ['i32'],
        frozenset(['head_dim']) : ['i32'],
        frozenset(['dropout_p']) : ['fp32'],
        frozenset(['philox_seed_ptr', 'philox_seed_output', 'philox_offset_output']) : ['*u64'],
        frozenset(['philox_offset1']) : ['*u32'],
        frozenset(['philox_offset2']) : ['u32'],
    }
    FEAT_CHOICES = {
        frozenset(['CAUSAL']) : [False, True],
        frozenset(['BLOCK_DMODEL']) : [16, 32, 64, 128, 256],
        frozenset(['ENABLE_DROPOUT']) : [False, True],
        frozenset(['RETURN_ENCODED_SOFTMAX']) : [False, True],
        frozenset(['PADDED_HEAD']) : [False, True],
        frozenset(['BIAS_TYPE']) : [0, 1],
    }
    PERF_CHOICES = {
        frozenset(['BLOCK_M']) : [16],
        frozenset(['BLOCK_N']) : [16],
        frozenset(['pre_load_v']) : [True, False],
    }
    TENSOR_RANKS = {
        '_default' : 4,
        'M': 2,
        'cu_seqlens_q': 1,
        'cu_seqlens_k': 1,
        'philox_seed_ptr': 0,
        'philox_offset1': 0,
        'philox_seed_output': 0,
        'philox_offset_output': 0,
    }
    EXPECTED_IDENTICAL_TENSOR_STRIDES = [
        # Not needed stride_o* exist
    ]
    SHIM_KERNEL_NAME = 'attn_fwd'

    # AUTOTUNE_KEYS can have Functional choices, which will be discarded later
    AUTOTUNE_KEYS = {
        'max_seqlen_q' : BinningLessOrEqual,
        'max_seqlen_k' : BinningLessOrEqual,
        'CAUSAL' : BinningExact,
        'ENABLE_DROPOUT' : BinningExact,
    }
    # List of functionals that are not fully tuned in the tuning database
    # First element of the tuple is name. Second is the value to use instead
    PARTIALLY_TUNED_FUNCTIONALS = [
        ('RETURN_ENCODED_SOFTMAX', False),
        ('PADDED_HEAD', None),
        ('BIAS_TYPE', None)
    ]

    # Python Trick: do not use @staticmethod, and also do not add 'self', and
    #               then there is no need to prefix the classname in DOWNGRADER list
    def DOWNGRADE_RETURN_ENCODED_SOFTMAX(tuned_kernel, compiler_options):
        return
        '''
        # tuned_kernel['BLOCK_M'] //= 2
        # tuned_kernel['BLOCK_N'] //= 2
        square = min(tuned_kernel['BLOCK_M'], tuned_kernel['BLOCK_N'])
        tuned_kernel['BLOCK_M'] = square // 2
        tuned_kernel['BLOCK_N'] = square // 2
        tuned_kernel['pre_load_v'] = False
        tuned_kernel['waves_per_eu'] = 0
        compiler_options['num_stages'] = 2
        '''

    DOWNGRADER = [(('RETURN_ENCODED_SOFTMAX', True), DOWNGRADE_RETURN_ENCODED_SOFTMAX)]

    @staticmethod
    def gen_autotune_configs(fsel_dict : 'dict[str, Any]'):
        dtype = fsel_dict['Q']
        ret = []
        BLOCK_SIZES = [(128, 64), (64, 64), (64, 32)]
        WAVES_PER_EU = [0, 1, 2, 3, 4]
        PRE_LOAD_V = [True, False]
        for (M, N), waves, pre in itertools.product(BLOCK_SIZES,
                                                    WAVES_PER_EU,
                                                    PRE_LOAD_V):
            if dtype == '*fp32:16':
                M //= 2
            kw = {'BLOCK_M': M, 'BLOCK_N': N, 'waves_per_eu': waves, 'pre_load_v': pre}
            yield Config(kw, num_stages=1, num_warps=4)
        yield from [
           Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'waves_per_eu': 2, 'pre_load_v': False}, num_stages=1, num_warps=8),
           Config({'BLOCK_M': 128, 'BLOCK_N':128, 'waves_per_eu': 2, 'pre_load_v': False}, num_stages=1, num_warps=4),
           Config({'BLOCK_M': 256, 'BLOCK_N':128, 'waves_per_eu': 2, 'pre_load_v': False}, num_stages=1, num_warps=8),
           Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 3, 'pre_load_v':  True}, num_stages=1, num_warps=4),
           Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 3, 'pre_load_v': False}, num_stages=1, num_warps=4),
           Config({'BLOCK_M':  64, 'BLOCK_N': 64, 'waves_per_eu': 4, 'pre_load_v': False}, num_stages=1, num_warps=8),
           Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 1, 'pre_load_v': False}, num_stages=1, num_warps=4),
           Config({'BLOCK_M':  32, 'BLOCK_N': 32, 'waves_per_eu': 4, 'pre_load_v': False}, num_stages=1, num_warps=8),
           Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'waves_per_eu': 1, 'pre_load_v': False}, num_stages=1, num_warps=4),
        ]
