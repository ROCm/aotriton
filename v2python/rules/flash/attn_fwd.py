# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from ._common import FlashKernel, select_pattern, BinningLessOrEqual, BinningExact

class attn_fwd(FlashKernel):
    ARGUMENTS = [
        'Q', 'K', 'V', 'B', 'sm_scale', 'M', 'Out',
        'stride_qz', 'stride_qh', 'stride_qm', 'stride_qk',
        'stride_kz', 'stride_kh', 'stride_kn', 'stride_kk',
        'stride_vz', 'stride_vh', 'stride_vk', 'stride_vn',
        'stride_oz', 'stride_oh', 'stride_om', 'stride_on',
        'stride_bz', 'stride_bh', 'stride_bm', 'stride_bn',
        'cu_seqlens_q', 'cu_seqlens_k',
        'seqlen_q', 'seqlen_k',  # Note: they were renamed to max_seqlens_q/k respectively, we kept it untouched for backward compatibility with tuning database
        'head_dim_q', 'head_dim_k',
        'dropout_p',
        'philox_seed',
        'philox_offset_base',
        'encoded_softmax',
        'VARLEN', # tl.constexpr starts here
        'STAGE',
        'BLOCK_M',
        'BLOCK_DMODEL',
        'BLOCK_N',
        'pre_load_v',  # TODO: kernel uses PRE_LOAD_V. We use this to keep backward compatibility
        'ENABLE_DROPOUT',
        'RETURN_ENCODED_SOFTMAX',
        'BIAS_TYPE',
        'PADDED_HEAD',
    ]
    TENSOR_STRIDE_INPUTS = {
        'Q' : select_pattern(ARGUMENTS, 'stride_q'),
        'K' : select_pattern(ARGUMENTS, 'stride_k'),
        'V' : select_pattern(ARGUMENTS, 'stride_v'),
        'B' : select_pattern(ARGUMENTS, 'stride_b'),
        'Out' : select_pattern(ARGUMENTS, 'stride_o'),
    }
    TYPE_CHOICES = {
        frozenset(['Q', 'K', 'V', 'B', 'Out', 'encoded_softmax']) : ['*fp16:16', '*bf16:16'],
        frozenset(['sm_scale']) : ['fp32'],
        frozenset(['M']) : ['*fp32:16'],
        frozenset(['cu_seqlens_q', 'cu_seqlens_k']) : ['*u32:16'],
        frozenset(['seqlen_q', 'seqlen_k', 'head_dim_q', 'head_dim_k']) : ['i32'],
        frozenset(['dropout_p']) : ['fp32'],
        frozenset(['philox_seed']) : ['u64'],
        frozenset(['philox_offset_base']) : ['u32'],
    }
    FEAT_CHOICES = {
        frozenset(['VARLEN']) : [False],  # TODO: support varlen
        frozenset(['STAGE']) : [1, 3],
        frozenset(['BLOCK_DMODEL']) : [16, 32, 64, 128, 256],
        frozenset(['ENABLE_DROPOUT']) : [True, False],
        frozenset(['RETURN_ENCODED_SOFTMAX']) : [True, False],
        frozenset(['BIAS_TYPE']) : [0],  # TODO: support bias
        frozenset(['PADDED_HEAD']) : [True, False],
    }
    PERF_CHOICES = {
        frozenset(['BLOCK_M']) : [16],
        frozenset(['BLOCK_N']) : [16],
        frozenset(['pre_load_v']) : [True, False],
    }
    TENSOR_RANKS = {
        '_default' : 4,
        'M' : 2,
        'cu_seqlens_q' : 1,
        'cu_seqlens_k' : 1,
    }
    EXPECTED_IDENTICAL_TENSOR_STRIDES = [
        # Not needed stride_o* exist
    ]
    # LAUNCHER_PARAMETERS is not used
    LAUNCHER_PARAMETERS = [
        'Q', 'K', 'V', 'sm_scale', 'M', 'Out',  # Basic functions
        'dropout_p', 'philox_seed', 'philox_offset', 'encoded_softmax',  # dropout
        'is_causal',  # Causal
    ]
    SHIM_KERNEL_NAME = 'attn_fwd'

    # AUTOTUNE_KEYS can have Functional choices, which will be discarded later
    AUTOTUNE_KEYS = {
        'seqlen_q' : BinningLessOrEqual,
        'seqlen_k' : BinningLessOrEqual,
        'STAGE' : BinningExact,
    }

    # List of functionals that are not fully tuned in the tuning database
    # First element of the tuple is name. Second is the value to use instead
    PARTIALLY_TUNED_FUNCTIONALS = [('RETURN_ENCODED_SOFTMAX', False), ('PADDED_HEAD', None)]

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

