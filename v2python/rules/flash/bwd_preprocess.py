from ._common import FlashKernel, get_possible_types, select_pattern, BinningLessOrEqual, BinningExact
from .attn_fwd import attn_fwd

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
