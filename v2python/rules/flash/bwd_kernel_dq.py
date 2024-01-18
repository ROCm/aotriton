from ._common import FlashKernel, get_possible_types, select_pattern, BinningLessOrEqual, BinningExact
from .attn_fwd import attn_fwd
from .bwd_kernel_dk_dv import bwd_kernel_dk_dv

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
