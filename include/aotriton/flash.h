#ifndef AOTRITON_V2_API_FLASH_ATTN_H
#define AOTRITON_V2_API_FLASH_ATTN_H

#include "util.h"
#include "runtime.h"

namespace aotriton::v2::flash {

using T4 = aotriton::TensorView<4>;
using T2 = aotriton::TensorView<2>;
using T1 = aotriton::TensorView<1>;

hipError_t attn_fwd(T4 q,                   // batch_size x num_heads x seqlen_q x head_size
                    T4 k,                   // batch_size x num_heads x seqlen_k x head_size
                    T4 v,                   // batch_size x num_heads x seqlen_k x head_size
                    float sm_scale,
                    T2 softmax_lse,
                    T4 Out,                 // batch_size x num_heads x seqlen_q x head_size
                    float dropout_p,
                    uint64_t philox_seed,
                    uint64_t philox_offset,
                    T4 encoded_softmax,
                    bool is_causal,
                    aotriton::Stream stream);

hipError_t attn_bwd(T4 q,                   // batch_size x num_heads x seqlen_q x head_size
                    T4 k,                   // batch_size x num_heads x seqlen_k x head_size
                    T4 v,                   // batch_size x num_heads x seqlen_k x head_size
                    float sm_scale,
                    T4 out,                 // batch_size x num_heads x seqlen_q x head_size
                    T4 dout,                // batch_size x num_heads x seqlen_q x head_size
                    T4 dq,                  // batch_size x num_heads x seqlen_q x head_size
                    T4 dk,                  // batch_size x num_heads x seqlen_q x head_size
                    T4 dv,                  // batch_size x num_heads x seqlen_q x head_size
                    T2 softmax_lse,
                    T2 delta,              // buffer, empty_like(softmax_lse)
                    float dropout_p,
                    uint64_t philox_seed,
                    uint64_t philox_offset,
                    bool is_causal,
                    aotriton::Stream stream);

} // aotriton::v2::flash 

#endif
