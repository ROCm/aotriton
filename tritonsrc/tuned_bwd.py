import triton
import triton.language as tl

from flash import (
    bwd_preprocess as bare_bwd_preprocess,
    bwd_kernel_dk_dv as bare_bwd_kernel_dk_dv,
    bwd_kernel_dq as bare_bwd_kernel_dq,
)

TRITON_CONFIG_LIST_BWD = []
# for BLOCK_M in [16, 32, 64]:
#     for BLOCK_N in [16, 32, 64]:
for BLOCK_M, BLOCK_N in [(32, 64), (64, 16)]:
    dic = {}
    dic['BLOCK_M'] = BLOCK_M
    dic['BLOCK_N'] = BLOCK_N
    # for waves_per_eu in range(0, 4+1):
    for waves_per_eu in [0, 3]:
        dic['waves_per_eu'] = waves_per_eu
        # for num_stages in [0, 1]:
        for num_stages in [1]:
            # for num_warps in [1,2,4,8]:
            for num_warps in [1,2]:
                cfg = triton.Config(dict(dic), num_stages=num_stages, num_warps=num_warps)
                TRITON_CONFIG_LIST_BWD.append(cfg)

print(TRITON_CONFIG_LIST_BWD)

bwd_dkdv_tuner = triton.autotune(
   configs=TRITON_CONFIG_LIST_BWD,
   key=['BLOCK_DMODEL', 'max_seqlen_q', 'max_seqlen_k'],
)
tuned_bwd_kernel_dk_dv = bwd_dkdv_tuner(bare_bwd_kernel_dk_dv)

bwd_dqdb_tuner = triton.autotune(
   configs=TRITON_CONFIG_LIST_BWD,
   key=['BLOCK_DMODEL', 'max_seqlen_q', 'max_seqlen_k'],
)
tuned_bwd_kernel_dq = bwd_dqdb_tuner(bare_bwd_kernel_dq)
