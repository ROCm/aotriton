// Copyright © 2023-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// clang-format off
#pragma once

#include <aotriton/config.h>
#include <aotriton/_internal/triton_kernel.h>
#include <aotriton/dtypes.h>
#include <aotriton/runtime.h>
#include <aotriton/util.h>
#include <aotriton/_internal/lazy_tensor_internal.h>
#include <functional>
#include <string>
#include <vector>

#if 1
namespace AOTRITON_NS::v3::flash {
    struct OpAttnFwdParams;
    struct attn_options;
}
#endif

namespace AOTRITON_NS::v3::flash {

#if 1
using AOTRITON_NS::v3::flash::OpAttnFwdParams;
#else
// The parameter class must be defined here when
// There is no common operator for debug_simulate_encoded_softmax.
struct OpAttnFwdParams {
    const TensorView<4>* encoded_softmax;
    float                dropout_p;
    int32_t              Num_head_q;
    int32_t              Max_seqlen_q;
    int32_t              Max_seqlen_k;
    const TensorView<0>* philox_seed_ptr;
    const TensorView<0>* philox_offset1;
    uint64_t             philox_offset2;
};
#endif

struct DebugSimulateEncodedSoftmaxContext {
    const OpAttnFwdParams *params = nullptr;
#if 1
    const attn_options *call_options = nullptr;
#endif
    template <typename ParentContext>
    DebugSimulateEncodedSoftmaxContext(const ParentContext& pcontext, bool condition)
      : launch_condition(condition)
    {
        params = pcontext.params;
#if 1
        call_options = pcontext.call_options;
#endif
    }
    // Performance related arguments for current selection
    int16_t BLOCK_M;
    int16_t BLOCK_N;

    TritonKernel* kernel_on_device = nullptr;
    int pp_args_index = -1;
    pstring_view flatzip_path;
    std::string_view aks2_entry;
    std::string_view func_name;
    std::string_view arch_name;
    // Note to save ELF space, this object is constructed on the fly.
    const char* _debug_kernel_name = nullptr;
#if AOTRITON_BUILD_FOR_TUNING
    int _has_preferred_kernel = -1; // For C++ based autotune database generation
    int _total_number_of_kernels = -1;
    const char* _preferred_kernel_psels = nullptr;
    const char* _preferred_kernel_copts = nullptr;
    bool peek_kernel_image = false;
#endif
    bool launch_condition = true;

    hipError_t lookup_optimal(Gpu gpu);
    hipError_t launch(hipStream_t stream) const;

    dim3 grid_calculator() const;
    std::function<dim3(const DebugSimulateEncodedSoftmaxContext&)> custom_grid_calculator;

    int64_t godel_number() const;
    static std::tuple<int, int> get_archmod_number(Gpu gpu);
    static constexpr int kMaxGodelNumber = 3;

    typedef void (*AutoTuneTableEntry)(DebugSimulateEncodedSoftmaxContext& context, int mod_number);
    static AutoTuneTableEntry autotune_table[][ kMaxGodelNumber ];
};

struct DebugSimulateEncodedSoftmaxMetadata {
    // Note: FEAT_CHOICES here
    static const std::vector<std::string>& get_encoded_softmax_choices();
    static const std::vector<std::string>& get_dropout_p_choices();
    static const std::vector<std::string>& get_Num_head_q_choices();
    static const std::vector<std::string>& get_philox_seed_ptr_choices();
    static const std::vector<std::string>& get_philox_offset2_choices();
};

namespace autotune {

extern const char debug_simulate_encoded_softmax_packed_string[];

extern int debug_simulate_encoded_softmax__lut_lambda__0(const OpAttnFwdParams& params, int mod_number, int8_t lut[1][1]);

void Autotune_debug_simulate_encoded_softmax__A0__F0(DebugSimulateEncodedSoftmaxContext& params, int mod_number);
void Autotune_debug_simulate_encoded_softmax__A0__F1(DebugSimulateEncodedSoftmaxContext& params, int mod_number);
void Autotune_debug_simulate_encoded_softmax__A0__F2(DebugSimulateEncodedSoftmaxContext& params, int mod_number);

}


}

// vim: set fileencoding=utf-8

