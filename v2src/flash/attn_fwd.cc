#include <aotriton/flash_attn.h>
#include <aotriton/v2/_internal_gen/flash/attn_fwd_params.h>
#include <aotriton/v2/_internal_gen/flash/attn_fwd_launcher.h>
#include "../static_switch.h"

namespace aotriton::v2::flash {

hipError_t attn_fwd(T4 q,
                    T4 k,
                    T4 v,
                    float sm_scale,
                    T4 softmax_lse,
                    T4 Out,
                    float dropout_p,
                    uint64_t philox_seed,
                    uint64_t philox_offset,
                    T4 encoded_softmax,
                    bool is_causal,
                    hipStream_t stream)
{
    hipError_t err;
    auto arch = getArchFromStream(stream);
    // TODO: An alternative way of doing this is to use a lookup table
    //       We will reconsider this approach later
#if 0
    ARCH_SWITCH(arch, Arch, [&] {
        TENSOR_DTYPE_SWITCH(q, Type, [&] {
            BOOL_SWITCH(is_causal, kUseCausal, [&] {
                BOOL_SWITCH(dropout_p != 0.0, kUseDropout, [&] {
                    BOOL_SWITCH(encoded_softmax, kReturnSoftmax, [&] {
                        DHEAD_SWITCH(head_size, kHeadSize, [&] {
                            auto kernel = attn_fwd_trait<Arch, Type, kHeadSize, kUseCausal, kUseDropout, kReturnSoftmax>::select_optimal_kernel();
                            err = attn_fwd_launcher(kernel,
                                                    q,
                                                    k,
                                                    v,
                                                    sm_scale,
                                                    M,
                                                    Out,
                                                    dropout_p,
                                                    philox_seed,
                                                    philox_offset,
                                                    encoded_softmax,
                                                    is_causal,
                                                    stream);
                        });
                    });
                });
            });
        });
    });
#endif
    constexpr int kUseCausalBits = 3;
    constexpr int kNoCausalBits = 1;
    auto grid_calculator = [](const AttnFwdParams& params) -> dim3 {
        return dim3 { cdiv(seqlen_q, params.BLOCK_N),
                      params.Q.size(0) * params.Q.size(1),
                      1
        };
    };
    // Requires C++ 20
    AttnFwdParams params {
        .Q = q,
        .K = k
        .V = v,
        .sm_scale = sm_scale,
        .M = softmax_lse,
        .Out = Out,
        .seqlen_q = q.size(2),
        .seqlen_k = k.size(2),
        .dropout_p = dropout_p,
        .philox_seed = philox_seed,
        .philox_offset = philox_offset,
        .encoded_softmax = encoded_softmax,
        .STAGE = is_causal ? kUseCausalBits : kNoCausalBits,
        .BLOCK_DMODEL = head_size,
        .ENABLE_DROPOUT = dropout_p > 0.0,
        .RETURN_ENCODED_SOFTMAX = bool(encoded_softmax),
        .grid_calculator = grid_calculator
    };
    err = params.lookup_optimal(arch);
    if (err != hipSuccess) {
        return err;
    }
    err = params.launch(stream);
    return err;
}

}
