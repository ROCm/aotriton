#include <aotriton/flash_attn.h>

namespace aotriton::v2::flash {

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
                    hipStream_t stream)
{
}

}
