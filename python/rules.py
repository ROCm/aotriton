from kernel_desc import KernelDescription

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
        'N_CTX',
        'STAGE', # tl.constexpr starts here
        'BLOCK_M',
        'BLOCK_DMODEL',
        'BLOCK_N',
        'pre_load_v',
    ]
    ARGUMENT_CHOICES = {
            # frozenset(['Q', 'K', 'V', 'Out']) : ['*fp16:16', '*bf16:16'],
            frozenset(['Q', 'K', 'V', 'Out']) : ['*fp16:16'], # TODO: The kernel provided in Triton doesn't support bf16
            frozenset(['sm_scale']) : ['fp32:16'],
            frozenset(['M']) : ['*fp32:16'],
            frozenset(_pattern(ARGUMENTS, 'stride_')) : ['u64'],
            frozenset(['Z', 'H', 'N_CTX']) : ['u64'],
            frozenset(['STAGE']) : [3, 1],
            frozenset(['BLOCK_M']) : [128],
            frozenset(['BLOCK_DMODEL']) : [16, 32, 64, 128],
            frozenset(['BLOCK_N']) : [64],
            frozenset(['pre_load_v']) : [True],
    }
    SHIM_KERNEL_NAME = 'attn_fwd'

kernels = [
    FlashAttention_attn_fwd('attn_fwd', 'tritonsrc/fused_attention_trimmed.py'),
    # KernelDescription('_bwd_preprocess', '06-fused-attention.py'),
    # KernelDescription('_bwd_kernel', '06-fused-attention.py'),
]
