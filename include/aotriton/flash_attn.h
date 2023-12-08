#ifndef AOTRITON_V2_API_FLASH_ATTN_H
#define AOTRITON_V2_API_FLASH_ATTN_H

#include "util.h"
#include "runtime.h"

namespace aotriton::v2::flash {

using T4 = TensorView<4>;
using T1 = TensorView<1>;

hipError_t attn_fwd(T4 q,                   // batch_size x num_heads x seqlen_q x head_size
                    T4 k,                   // batch_size x num_heads x seqlen_k x head_size
                    T4 v,                   // batch_size x num_heads x seqlen_k x head_size
                    float sm_scale,
                    T4 M,
                    T4 Out,                 // batch_size x num_heads x seqlen_q x head_size
                    float dropout_p,
                    uint64_t philox_seed,
                    uint64_t philox_offset,
                    T4 encoded_softmax,
                    hipStream_t stream);

hipError_t attn_bwd(T4 q,
                    T4 k,
                    T4 v,
                    float sm_scale,
                    T4 Out,
                    T4 dOut,
                    T4 dq,
                    T4 dk,
                    T4 dv,
                    T1 L,
                    T1 delta,
                    float dropout_p,
                    uint64_t philox_seed,
                    uint64_t philox_offset,
                    hipStream_t stream);

} // aotriton::v2::flash 

#endif
