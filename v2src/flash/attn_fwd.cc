#include <aotriton/flash.h>
#include <aotriton/util.h>
#include <flash/shim.attn_fwd.h>
#include <iostream>

namespace aotriton::v2::flash {

hipError_t attn_fwd(T4 q,
                    T4 k,
                    T4 v,
                    float sm_scale,
                    T2 softmax_lse,
                    T4 out,
                    float dropout_p,
                    uint64_t philox_seed,
                    uint64_t philox_offset,
                    T4 encoded_softmax,
                    bool is_causal,
                    aotriton::Stream stream_wrap)
{
    hipError_t err;
    auto stream = stream_wrap.native();
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
    if (arch == "gfx90a") {
        constexpr Arch = Gfx90a;
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
    } else if (arch = "gfx1101") {
        constexpr Arch = Gfx1101;
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
    }
#endif
    constexpr int kUseCausalBits = 3;
    constexpr int kNoCausalBits = 1;
    auto grid_calculator = [](const AttnFwdParams& params) -> dim3 {
        std::cerr << "Selected Kernel "
                  << " BLOCK_M = " << params.BLOCK_M
                  << " BLOCK_N = " << params.BLOCK_N
                  << " pre_load_v = " << params.pre_load_v
                  << std::endl;
        dim3 grid { aotriton::cdiv<uint32_t>(params.seqlen_q, params.BLOCK_M),
                      uint32_t(params.Q->size(0) * params.Q->size(1)),
                      1
        };
        std::cerr << "Grid conf " << grid.x << " " << grid.y << " " << grid.z << std::endl;
        return grid;
    };
    int head_size = q.size(3);
    // Requires C++ 20
    AttnFwdParams params = {
        .Q = &q,
        .K = &k,
        .V = &v,
        .Out = &out,
        .encoded_softmax = &encoded_softmax,
        .sm_scale = sm_scale,
        .M = &softmax_lse,
        .seqlen_q = q.size(2),
        .seqlen_k = k.size(2),
        .dropout_p = dropout_p,
        .philox_seed = philox_seed,
        .philox_offset_base = static_cast<uint32_t>(philox_offset),
        .STAGE = is_causal ? kUseCausalBits : kNoCausalBits,
        .BLOCK_DMODEL = head_size,
        .ENABLE_DROPOUT = dropout_p > 0.0,
        .RETURN_ENCODED_SOFTMAX = bool(encoded_softmax),
    };
    AttnFwdContext context;
    context.grid_calculator = grid_calculator;
    // .grid_calculator = grid_calculator
    err = context.lookup_optimal(params, arch);
    if (err != hipSuccess) {
        return err;
    }
    err = context.launch(params, stream);
    return err;
}

}
