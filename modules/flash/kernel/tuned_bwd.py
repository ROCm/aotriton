import triton
import triton.language as tl
import torch

from flash import (
    bwd_preprocess as bare_bwd_preprocess,
    bwd_kernel_dk_dv as bare_bwd_kernel_dk_dv,
    bwd_kernel_dq as bare_bwd_kernel_dq,
)

def evaluate_gfx_arch_within(arch_list):
    assert torch.cuda.is_available()
    gcn_arch_name = torch.cuda.get_device_properties('cuda').gcnArchName
    return any(arch in gcn_arch_name for arch in arch_list)

def is_rdna():
    return evaluate_gfx_arch_within(['gfx1100', 'gfx1101', 'gfx1102', 'gfx1103', 'gfx1150', 'gfx1151', 'gfx1152', 'gfx1153', 'gfx1200', 'gfx1201'])

IS_RDNA = is_rdna()

def get_num_of_xcds():
    if evaluate_gfx_arch_within(['gfx942', 'gfx950']):
        return 8
    # Unknown
    return 1

NUM_XCDS = get_num_of_xcds()

TRITON_CONFIG_LIST_BWD = []
# BLOCK_SIZES = [(64, 64), (64, 32), (64, 16), (32, 32), (32, 16)]
# NUM_WARPS = [4]
# WAVES_PER_EU = [0,1,2,3,4]
BLOCK_SIZES = [(64, 16), (32, 16)]
NUM_WARPS = [4, 8]
WAVES_PER_EU = [2, 3]
# NUM_WARPS = [2, 4, 8]
# BLOCK_SIZES = [(64, 16), (16, 64)]
# BLOCK_SIZES = [(64, 64), (64, 32), (64, 16), (16, 64), (32, 32), (32, 16)]
# WAVES_PER_EU = [1,2,3,4]
# for BLOCK_M in [16, 32, 64]:
#     for BLOCK_N in [16, 32, 64]:
# for BLOCK_M, BLOCK_N in [(32, 64), (64, 16)]:
for BLOCK_M, BLOCK_N in BLOCK_SIZES:
    dic = {}
    dic['BLOCK_M'] = BLOCK_M
    dic['BLOCK_N'] = BLOCK_N
    # for waves_per_eu in range(0, 4+1):
    for waves_per_eu in WAVES_PER_EU:
        dic['waves_per_eu'] = waves_per_eu
        # for num_stages in [0, 1]:
        for num_stages in [1]:
            # for num_warps in [1,2,4,8]:
            for num_warps in NUM_WARPS:
                cfg = triton.Config(dict(dic), num_stages=num_stages, num_warps=num_warps)
                TRITON_CONFIG_LIST_BWD.append(cfg)

FAST_DKDV = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 16, 'waves_per_eu': 3}, num_stages=1, num_warps=4)
]

FAST_DQDB = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 16, 'waves_per_eu': 3}, num_stages=1, num_warps=4)
]

bwd_dkdv_tuner = triton.autotune(
   configs=TRITON_CONFIG_LIST_BWD,
   # configs=FAST_DKDV,
   key=['BLOCK_DMODEL', 'max_seqlen_q', 'max_seqlen_k'],
)
tuned_bwd_kernel_dk_dv = bwd_dkdv_tuner(bare_bwd_kernel_dk_dv)

bwd_dqdb_tuner = triton.autotune(
   configs=TRITON_CONFIG_LIST_BWD,
   # configs=FAST_DQDB,
   key=['BLOCK_DMODEL', 'max_seqlen_q', 'max_seqlen_k'],
)
tuned_bwd_kernel_dq = bwd_dqdb_tuner(bare_bwd_kernel_dq)
