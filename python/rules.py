from kernel_desc import KernelDescription, get_possible_types

def _pattern(arguments, prefix):
    ret = []
    for s in arguments:
        if s.startswith(prefix):
            ret.append(s)
    return ret

class FlashAttention_attn_fwd(KernelDescription):
    ARGUMENTS = [
        'Q', 'K', 'V', 'sm_scale', 'M', 'Out',
        'stride_qz', 'stride_qh', 'stride_qm', 'stride_qk',
        'stride_kz', 'stride_kh', 'stride_kn', 'stride_kk',
        'stride_vz', 'stride_vh', 'stride_vk', 'stride_vn',
        'stride_oz', 'stride_oh', 'stride_om', 'stride_on',
        'Z', 'H',
        'seqlen_q', 'seqlen_k',
        'STAGE', # tl.constexpr starts here
        'BLOCK_M',
        'BLOCK_DMODEL',
        'BLOCK_N',
        'pre_load_v',
    ]
    ARGUMENT_CHOICES = {
            # frozenset(['Q', 'K', 'V', 'Out']) : ['*fp16:16', '*bf16:16'],
            frozenset(['Q', 'K', 'V', 'Out']) : ['*fp16:16', '*bf16:16'],
            frozenset(['sm_scale']) : ['fp32'],
            frozenset(['M']) : ['*fp32:16'],
            frozenset(_pattern(ARGUMENTS, 'stride_')) : ['u64'],
            frozenset(['Z', 'H', 'seqlen_q', 'seqlen_k']) : ['u64'],
            frozenset(['STAGE']) : [1, 3],
            frozenset(['BLOCK_M']) : [128],
            frozenset(['BLOCK_DMODEL']) : [16, 32, 64, 128],
            frozenset(['BLOCK_N']) : [64],
            frozenset(['pre_load_v']) : [True],
    }
    SHIM_KERNEL_NAME = 'attn_fwd'

class FlashAttention_bwd_preprocess(KernelDescription):
    ARGUMENTS = [
        'Out', 'DO',
        'NewDO', 'Delta',
        'BLOCK_M', # tl.constexpr starts here
        'D_HEAD',
    ]
    ARGUMENT_CHOICES = {
        frozenset(['Out', 'DO', 'NewDO']) : ['*fp16:16', '*bf16:16'],
        frozenset(['Delta']) : ['*fp32:16'],
        frozenset(['BLOCK_M']) : [64, 128], # TODO: All possible values?
        frozenset(['D_HEAD']) : get_possible_types(FlashAttention_attn_fwd, 'BLOCK_DMODEL'),
    }
    SHIM_KERNEL_NAME = 'bwd_preprocess'

# CAVEAT: Not well tested in trition
# Backward must use split kernel
class FlashAttention_bwd_kernel(KernelDescription):
    ARGUMENTS = [
        'Q', 'K', 'V', 'sm_scale',
        'Out', 'dO',
        'dQ', 'dK', 'dV',
        'L', 'D',
        'stride_qz', 'stride_qh', 'stride_qm', 'stride_qk',
        'stride_kz', 'stride_kh', 'stride_kn', 'stride_kk',
        'stride_vz', 'stride_vh', 'stride_vk', 'stride_vn',
        'Z', 'H',
        'seqlen_q', 'seqlen_k',
        'BLOCK_M',  # tl.constexpr starts here
        'BLOCK_DMODEL',
        'BLOCK_N',
        'CAUSAL',
    ]
    ARGUMENT_CHOICES = {
            # frozenset(['Q', 'K', 'V', 'Out']) : ['*fp16:16', '*bf16:16'],
            frozenset(['Q', 'K', 'V', 'Out', 'dO', 'dK', 'dV']) : get_possible_types(FlashAttention_attn_fwd, 'Q'),
            frozenset(['sm_scale']) : get_possible_types(FlashAttention_attn_fwd, 'sm_scale'),
            frozenset(['dQ', 'L', 'D']) : ['*fp32:16'],
            frozenset(_pattern(ARGUMENTS, 'stride_')) : ['u64'],
            # FIXME: type of num_block_q and num_block_kv
            frozenset(['Z', 'H', 'seqlen_q', 'seqlen_k']) : ['u64'],
            frozenset(['BLOCK_M']) : [128],
            frozenset(['BLOCK_DMODEL']) : [16, 32, 64, 128],
            frozenset(['BLOCK_N']) : [64],
            frozenset(['CAUSAL']) : [True],
    }
    SHIM_KERNEL_NAME = 'bwd_kernel'

class FlashAttention_bwd_kernel_dk_dv(KernelDescription):
    ARGUMENTS = [
        'Q', 'K', 'V', 'sm_scale', 'Out', 'DO',
        'DK', 'DV',
        'L', 'D',
        'stride_qz', 'stride_qh', 'stride_qm', 'stride_qk',
        'stride_kz', 'stride_kh', 'stride_kn', 'stride_kk',
        'stride_vz', 'stride_vh', 'stride_vk', 'stride_vn',
        'Z', 'H',
        'seqlen_q', 'seqlen_k',
        'BLOCK_M', # tl.constexpr starts here
        'BLOCK_DMODEL',
        'BLOCK_N',
    ]
    match_fwd = lambda aname : get_possible_types(FlashAttention_attn_fwd, aname)
    ARGUMENT_CHOICES = {
        frozenset(['Q', 'K', 'V', 'Out', 'DO', 'DK', 'DV']) : match_fwd('Q'),
        frozenset(['sm_scale']) : match_fwd( 'sm_scale'),
        frozenset(['L', 'D']) : ['*fp32:16'],
        frozenset(_pattern(ARGUMENTS, 'stride_')) : ['u64'],
        # FIXME: type of num_block_q and num_block_kv
        frozenset(['Z', 'H', 'seqlen_q', 'seqlen_k']) : ['u64'],
        frozenset(['BLOCK_M']) : [16],
        frozenset(['BLOCK_DMODEL']) : [16, 32, 64, 128],
        frozenset(['BLOCK_N']) : [16],
    }
    # TODO: num_warps=4, num_stages=1
    SHIM_KERNEL_NAME = 'bwd_kernel_dk_dv'

class FlashAttention_bwd_kernel_dq(KernelDescription):
    ARGUMENTS = [
        'Q', 'K', 'V', 'sm_scale', 'Out', 'dO',
        'dQ',
        'L', 'D',
        'stride_qz', 'stride_qh', 'stride_qm', 'stride_qk',
        'stride_kz', 'stride_kh', 'stride_kn', 'stride_kk',
        'stride_vz', 'stride_vh', 'stride_vk', 'stride_vn',
        'Z', 'H',
        'seqlen_q', 'seqlen_k',
        'BLOCK_M', # tl.constexpr starts here
        'BLOCK_DMODEL',
        'BLOCK_N',
    ]
    match_fwd = lambda aname : get_possible_types(FlashAttention_attn_fwd, aname)
    ARGUMENT_CHOICES = {
        frozenset(['Q', 'K', 'V', 'Out', 'dO', 'dQ']) : match_fwd('Q'),
        frozenset(['sm_scale']) : match_fwd( 'sm_scale'),
        frozenset(['L', 'D']) : ['*fp32:16'],
        frozenset(_pattern(ARGUMENTS, 'stride_')) : ['u64'],
        # FIXME: type of num_block_q and num_block_kv
        frozenset(['Z', 'H', 'seqlen_q', 'seqlen_k']) : ['u64'],
        frozenset(['BLOCK_M']) : [32],
        frozenset(['BLOCK_DMODEL']) : [16],
        frozenset(['BLOCK_N']) : [16],
    }
    # TODO: num_warps=4, num_stages=1, waves_per_eu=1
    SHIM_KERNEL_NAME = 'bwd_kernel_dq'

kernels = [
    FlashAttention_attn_fwd('attn_fwd', 'tritonsrc/fused_attention_trimmed.py'),
    FlashAttention_bwd_preprocess('bwd_preprocess', 'tritonsrc/fused_attention_trimmed.py'),
    FlashAttention_bwd_kernel('bwd_kernel', 'tritonsrc/fused_attention_trimmed.py'),
    FlashAttention_bwd_kernel_dk_dv('bwd_kernel_dk_dv', 'tritonsrc/fused_attention_trimmed.py'),
    FlashAttention_bwd_kernel_dq('bwd_kernel_dq', 'tritonsrc/fused_attention_trimmed.py'),
]
