from ._common import FlashKernel, select_pattern, BinningLessOrEqual, BinningExact

class attn_fwd(FlashKernel):
    ARGUMENTS = [
        'Q', 'K', 'V', 'sm_scale', 'M', 'Out',
        'stride_qz', 'stride_qh', 'stride_qm', 'stride_qk',
        'stride_kz', 'stride_kh', 'stride_kn', 'stride_kk',
        'stride_vz', 'stride_vh', 'stride_vk', 'stride_vn',
        'stride_oz', 'stride_oh', 'stride_om', 'stride_on',
        'seqlen_q', 'seqlen_k',
        'dropout_p',
        'philox_seed',
        'philox_offset_base',
        'encoded_softmax',
        'STAGE', # tl.constexpr starts here
        'BLOCK_M',
        'BLOCK_DMODEL',
        'BLOCK_N',
        'pre_load_v',
        'ENABLE_DROPOUT',
        'RETURN_ENCODED_SOFTMAX',
    ]
    TENSOR_STRIDE_INPUTS = {
        'Q' : select_pattern(ARGUMENTS, 'stride_q'),
        'K' : select_pattern(ARGUMENTS, 'stride_k'),
        'V' : select_pattern(ARGUMENTS, 'stride_v'),
        'Out' : select_pattern(ARGUMENTS, 'stride_o'),
    }
    TYPE_CHOICES = {
        frozenset(['Q', 'K', 'V', 'Out', 'encoded_softmax']) : ['*fp16:16', '*bf16:16'],
        frozenset(['sm_scale']) : ['fp32'],
        frozenset(['M']) : ['*fp32:16'],
        # frozenset(select_pattern(ARGUMENTS, 'stride_', trim=1)) : ['u64'],
        # frozenset(select_pattern(ARGUMENTS, 'stride_', trim=1)) : ['u64'],
        frozenset(['seqlen_q', 'seqlen_k']) : ['u64'],
        frozenset(['dropout_p']) : ['fp32'],
        frozenset(['philox_seed']) : ['u64'],
        frozenset(['philox_offset_base']) : ['u32'],
    }
    FEAT_CHOICES = {
        frozenset(['STAGE']) : [1, 3],
        frozenset(['BLOCK_DMODEL']) : [16, 32, 64, 128],
        frozenset(['ENABLE_DROPOUT']) : [True, False],
        frozenset(['RETURN_ENCODED_SOFTMAX']) : [True, False],
    }
    PERF_CHOICES = {
        frozenset(['BLOCK_M']) : [16],
        frozenset(['BLOCK_N']) : [16],
        frozenset(['pre_load_v']) : [True, False],
    }
    # Optional, can be derived from 
    TENSOR_RANKS_OVERRIDE = {
        '_default' : 4,
        'M': 2,
    }
    EXPECTED_IDENTICAL_TENSOR_STRIDES = {
    }
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
    PARTIALLY_TUNED_FUNCTIONALS = [('RETURN_ENCODED_SOFTMAX', False)]

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

