from ..kernel_desc import KernelDescription, get_possible_types, select_pattern
from ..autotune_binning import BinningLessOrEqual, BinningExact

class FlashKernel(KernelDescription):
    KERNEL_FAMILY = 'flash'

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
    TYPE_CHOICES = {
        frozenset(['Q', 'K', 'V', 'Out', 'encoded_softmax']) : ['*fp16:16', '*bf16:16'],
        frozenset(['sm_scale']) : ['fp32'],
        frozenset(['M']) : ['*fp32:16'],
        frozenset(select_pattern(ARGUMENTS, 'stride_')) : ['u64'],
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

    TENSOR_STRIDE_INPUTS = {
        'Q' : select_pattern(ARGUMENTS, 'stride_q'),
        'K' : select_pattern(ARGUMENTS, 'stride_k'),
        'V' : select_pattern(ARGUMENTS, 'stride_v'),
        'Out' : select_pattern(ARGUMENTS, 'stride_o'),
    }
    # Optional, can be derived from 
    TENSOR_RANKS_OVERRIDE = {
        '_default' : 4,
        'M': 1,
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
    # List of functionals that only has fixed values in the tuning database
    UNTUNED_FUNCTIONALS = ['RETURN_ENCODED_SOFTMAX']

class bwd_preprocess(FlashKernel):
    ARGUMENTS = [
        'Out', 'DO',
        'NewDO', 'Delta',
        'BLOCK_M', # tl.constexpr starts here
        'D_HEAD',
    ]
    TYPE_CHOICES = {
        frozenset(['Out', 'DO', 'NewDO']) : ['*fp16:16', '*bf16:16'],
        frozenset(['Delta']) : ['*fp32:16'],
    }
    FEAT_CHOICES = {
        frozenset(['D_HEAD']) : get_possible_types(attn_fwd, 'BLOCK_DMODEL'),
    }
    PERF_CHOICES = {
        frozenset(['BLOCK_M']) : [16, 64, 128], # TODO: All possible values?
    }
    SHIM_KERNEL_NAME = 'bwd_preprocess'

class bwd_kernel_dk_dv(FlashKernel):
    ARGUMENTS = [
        'Q', 'K', 'V', 'sm_scale', 'Out', 'DO',
        'DK', 'DV',
        'L', 'D',
        'stride_qz', 'stride_qh', 'stride_qm', 'stride_qk',
        'stride_kz', 'stride_kh', 'stride_kn', 'stride_kk',
        'stride_vz', 'stride_vh', 'stride_vk', 'stride_vn',
        'seqlen_q', 'seqlen_k',
        'dropout_p',
        'philox_seed',
        'philox_offset_base',
        'BLOCK_M', # tl.constexpr starts here
        'BLOCK_DMODEL',
        'BLOCK_N',
        'CAUSAL',
        'ENABLE_DROPOUT',
    ]
    match_fwd = lambda aname : get_possible_types(attn_fwd, aname)
    TYPE_CHOICES = {
        frozenset(['Q', 'K', 'V', 'Out', 'DO', 'DK', 'DV']) : match_fwd('Q'),
        frozenset(['sm_scale']) : match_fwd( 'sm_scale'),
        frozenset(['L', 'D']) : ['*fp32:16'],
        frozenset(select_pattern(ARGUMENTS, 'stride_')) : ['u64'],
        frozenset(['seqlen_q', 'seqlen_k']) : ['u64'],
        frozenset(['dropout_p']) : match_fwd('dropout_p'),
        frozenset(['philox_seed']) : match_fwd('philox_seed'),
        frozenset(['philox_offset_base']) : match_fwd('philox_offset_base'),
    }
    FEAT_CHOICES = {
        frozenset(['BLOCK_DMODEL']) : [16, 32, 64, 128],
        frozenset(['CAUSAL']) : [True, False],
        frozenset(['ENABLE_DROPOUT']) : match_fwd('ENABLE_DROPOUT'),
    }
    PERF_CHOICES = {
        frozenset(['BLOCK_M', 'BLOCK_N']) : match_fwd('BLOCK_M'),
    }
    TENSOR_STRIDE_INPUTS = {
        'Q' : select_pattern(ARGUMENTS, 'stride_q'),
        'K' : select_pattern(ARGUMENTS, 'stride_k'),
        'V' : select_pattern(ARGUMENTS, 'stride_v'),
    }
    EXPECTED_IDENTICAL_TENSOR_STRIDES = {
        'DO': 'Q',
        'DK': 'K',
        'DV': 'V',
    }
    DEFAULT_NUM_WARPS=4
    DEFAULT_NUM_STAGES=1
    SHIM_KERNEL_NAME = 'bwd_kernel_dk_dv'

class bwd_kernel_dq(FlashKernel):
    ARGUMENTS = [
        'Q', 'K', 'V', 'sm_scale', 'Out', 'dO',
        'dQ',
        'L', 'D',
        'stride_qz', 'stride_qh', 'stride_qm', 'stride_qk',
        'stride_kz', 'stride_kh', 'stride_kn', 'stride_kk',
        'stride_vz', 'stride_vh', 'stride_vk', 'stride_vn',
        'seqlen_q', 'seqlen_k',
        'dropout_p',
        'philox_seed',
        'philox_offset_base',
        'BLOCK_M', # tl.constexpr starts here
        'BLOCK_DMODEL',
        'BLOCK_N',
        'CAUSAL',
        'ENABLE_DROPOUT',
    ]
    match_fwd = lambda aname : get_possible_types(attn_fwd, aname)
    match_kv = lambda aname : get_possible_types(bwd_kernel_dk_dv, aname)
    TYPE_CHOICES = {
        frozenset(['Q', 'K', 'V', 'Out', 'dO', 'dQ']) : match_fwd('Q'),
        frozenset(['sm_scale']) : match_fwd( 'sm_scale'),
        frozenset(['L', 'D']) : ['*fp32:16'],
        frozenset(select_pattern(ARGUMENTS, 'stride_')) : ['u64'],
        # FIXME: type of num_block_q and num_block_kv
        frozenset(['seqlen_q', 'seqlen_k']) : ['u64'],
        frozenset(['dropout_p']) : match_fwd('dropout_p'),
        frozenset(['philox_seed']) : match_fwd('philox_seed'),
        frozenset(['philox_offset_base']) : match_fwd('philox_offset_base'),
    }
    FEAT_CHOICES = {
        frozenset(['BLOCK_DMODEL']) : [16, 32, 64, 128],
        frozenset(['CAUSAL']) : match_kv('CAUSAL'),
        frozenset(['ENABLE_DROPOUT']) : match_fwd('ENABLE_DROPOUT'),
    }
    PERF_CHOICES = {
        frozenset(['BLOCK_M', 'BLOCK_N']) : match_fwd('BLOCK_M'),
    }
    TENSOR_STRIDE_INPUTS = {
        'Q' : select_pattern(ARGUMENTS, 'stride_q'),
        'K' : select_pattern(ARGUMENTS, 'stride_k'),
        'V' : select_pattern(ARGUMENTS, 'stride_v'),
    }
    EXPECTED_IDENTICAL_TENSOR_STRIDES = {
        'DO': 'Q',
        'DK': 'K',
        'DV': 'V',
    }
    DEFAULT_NUM_WARPS=4
    DEFAULT_NUM_STAGES=1
    # TODO: waves_per_eu=1
    SHIM_KERNEL_NAME = 'bwd_kernel_dq'

SOURCE_FILE = 'tritonsrc/flash.py'
kernels = [
    attn_fwd('attn_fwd', SOURCE_FILE),
    # bwd_preprocess('bwd_preprocess', SOURCE_FILE),
    # bwd_kernel_dk_dv('bwd_kernel_dk_dv', SOURCE_FILE),
    # bwd_kernel_dq('bwd_kernel_dq', SOURCE_FILE),
]
