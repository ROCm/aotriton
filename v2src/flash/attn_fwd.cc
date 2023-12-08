#include <aotriton/flash_attn.h>
#include <aotriton/v2/_internal_gen/traits_gpu_kernel.h>
#include <aotriton/v2/_internal_gen/gpu_kernel_launcher.h>
#include "../static_switch.h"

namespace aotriton::v2::flash {

hipError_t attn_fwd(T4 q,
                    T4 k,
                    T4 v,
                    float sm_scale,
                    T4 M,
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
    ARCH_SWITCH(arch, Arch, [&] {
        TENSOR_DTYPE_SWITCH(q, Type, [&] {
            BOOL_SWITCH(is_causal, kUseCausal, [&] {
                BOOL_SWITCH(dropout_p != 0.0, kUseDropout, [&] {
                    BOOL_SWITCH(encoded_softmax, kReturnSoftmax, [&] {
                        auto kernel = attn_fwd_trait<Arch, Type, kUseCausal, kUseDropout, kReturnSoftmax>::select_optimal_kernel();
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
    return err;
}

}
