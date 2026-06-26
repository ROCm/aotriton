// Copyright © 2023-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// clang-format off
#include "shim.debug_simulate_encoded_softmax.h"
#include <aotriton/util.h>
#include <tuple>
#include <iostream>
#if AOTRITON_BUILD_FOR_TUNING
#include <aotriton/cpp_tune.h>
#endif
#include "iface.op_attn_fwd.h"

namespace AOTRITON_NS::v3::flash {

#if 1
using AOTRITON_NS::v3::flash::OpAttnFwdParams;
#define KERNEL_SLOT_INDEX (attn_options::KernelSlot::debug_simulate_encoded_softmax)
#endif

#define CAST(x) const_cast<void*>(static_cast<const void*>(x))
typedef std::vector<void*>(*PP_FUNC)(const OpAttnFwdParams& context, const TritonAuxiliaryArguments&);

namespace {
extern PP_FUNC prepare_arguments[ 1 ];
}

int64_t DebugSimulateEncodedSoftmaxContext::godel_number() const
{
    int64_t sum = 0;
    const auto& args = *params;
    {
        int64_t number = -1;
        if (args.encoded_softmax->dtype() == DType::kFloat16) number = 0 ;
        if (args.encoded_softmax->dtype() == DType::kBFloat16) number = 1 ;
        if (args.encoded_softmax->dtype() == DType::kFloat32) number = 2 ;
        if (number < 0) {
#ifndef NDEBUG
            std::cerr << __FILE__ << ":" << __LINE__ << ": Unsupported encoded_softmax, value: " << args.encoded_softmax->dtype() << std::endl;
#endif
            return -1;
        }
        sum += number * 1;
    }

    return sum;
}

hipError_t
DebugSimulateEncodedSoftmaxContext::lookup_optimal(Gpu gpu) {
    // Note:
    // context object must be called in this order
    //  ctor -> lookup_optimal -> launch
    // Here launch_condition is re-used as initial value from ctor, which is
    // the condition set by metro kernel to completely disable a specific
    // kernel (e.g. debug_simulate_encoded_softmax).
    // Hence, launch_condition must be checked at the beginning of
    // lookup_optimal() as well.
    if (!launch_condition)
      return hipSuccess;
#if AOTRITON_BUILD_FOR_TUNING && 1
    if (call_options) {
        auto& kctl = *call_options->kernel_fine_control[KERNEL_SLOT_INDEX];
        uint16_t ctrl = kctl.control_bits;

        // Check Ignore flag - skip lookup/execution if set
        if (ctrl & KernelControl::Ignore) {
            launch_condition = false;
            return hipSuccess;
        }

        // Check Skip flag - lookup but skip execution
        if (ctrl & KernelControl::Skip) {
            launch_condition = false;
        }

        // Check Manual flag - use hsaco_index if set
        if (ctrl & KernelControl::Manual) {
            _has_preferred_kernel = kctl.hsaco_index;
            peek_kernel_image = (ctrl & KernelControl::ExtractImage);
        }
    }
#endif

    auto [arch_number, mod_number] = get_archmod_number(gpu);
    if (arch_number < 0) {
        return hipErrorNoBinaryForGpu;
    }
    kernel_on_device = nullptr;
    auto number = godel_number();
    if (number < 0)
        return hipErrorNotSupported;
    auto tune_func = autotune_table[arch_number][number];
    if (!tune_func)
        return hipErrorProfilerNotInitialized;
    tune_func(*this, mod_number);
    if (!kernel_on_device)
        return hipErrorSharedObjectSymbolNotFound;

#if AOTRITON_BUILD_FOR_TUNING && 1
    if (call_options) {
        auto& kctl = *call_options->kernel_fine_control[KERNEL_SLOT_INDEX];
        uint16_t ctrl = kctl.control_bits;

        // Write total_hsacos if Query is set
        if (ctrl & KernelControl::Query) {
            kctl.total_hsacos = _total_number_of_kernels;
            kctl.kernel_psels = _preferred_kernel_psels;
            kctl.kernel_copts = _preferred_kernel_copts;
        }
    }
#endif
    return hipSuccess;
}

hipError_t
DebugSimulateEncodedSoftmaxContext::launch(hipStream_t stream) const {
    if (!launch_condition)
      return hipSuccess;
    constexpr std::string_view triton_kernel_name { "debug_simulate_encoded_softmax" };
    TritonAuxiliaryArguments aux;
    auto args = prepare_arguments[pp_args_index](*this->params, aux);
    dim3 grid;
    if (custom_grid_calculator) {
        grid = custom_grid_calculator(*this);
    } else {
        grid = grid_calculator();
    }
#if AOTRITON_BUILD_FOR_TUNING && 1
    auto ret = kernel_on_device->invoke(triton_kernel_name,
                                        flatzip_path,
                                        aks2_entry,
                                        func_name,
                                        arch_name,
                                        grid,
                                        args,
                                        peek_kernel_image,
                                        stream);
    if (ret != hipSuccess)
         return ret;
    if (call_options) {
        auto& kctl = *call_options->kernel_fine_control[KERNEL_SLOT_INDEX];
        uint16_t ctrl = kctl.control_bits;
        if (ctrl & KernelControl::Manual && ctrl & KernelControl::ExtractImage) {
            auto essentials = kernel_on_device->get_image_info_iff_decompressed();
            kctl.kernel_image = essentials.image;
            kctl.image_size = essentials.size;
        }
    }
    return ret;
#else
    return kernel_on_device->invoke(triton_kernel_name,
                                    flatzip_path,
                                    aks2_entry,
                                    func_name,
                                    arch_name,
                                    grid,
                                    args,
                                    stream);
#endif
}

std::tuple<int, int>
DebugSimulateEncodedSoftmaxContext::get_archmod_number(Gpu gpu) {
    if (gpu == GPU_AMD_ARCH_GFX942_MOD0) return { 0, 0 };
    // TODO: print warning about tuning for this GPU mod is not built.
    // Note: if some mod does not have tuning info in the database at all, the
    //       getGpuFromStream should not return that mod from beginning.
    return std::make_tuple(-1, 0);
}


static std::vector<void*>
debug_simulate_encoded_softmax_pp_args_0(const OpAttnFwdParams& params,
                                         const TritonAuxiliaryArguments& aux) {
  return { params.encoded_softmax->kparam_data_ptr(), // encoded_softmax
           params.encoded_softmax->kparam_stride(0), // stride_rz
           params.encoded_softmax->kparam_stride(1), // stride_rh
           params.encoded_softmax->kparam_stride(2), // stride_rm
           CAST(&params.dropout_p), // dropout_p
           CAST(&params.Num_head_q), // Num_head_q
           CAST(&params.Max_seqlen_q), // Max_seqlen_q
           CAST(&params.Max_seqlen_k), // Max_seqlen_k
           params.philox_seed_ptr->kparam_data_ptr(), // philox_seed_ptr
           params.philox_offset1->kparam_data_ptr(), // philox_offset1
           CAST(&params.philox_offset2), // philox_offset2
           CAST(&aux.global_scratch),
           CAST(&aux.profile_scratch)
         };
}

namespace {
PP_FUNC prepare_arguments[ 1 ] = {
  debug_simulate_encoded_softmax_pp_args_0
};
}


const std::vector<std::string>& DebugSimulateEncodedSoftmaxMetadata::get_encoded_softmax_choices()
{
    static const std::vector<std::string> choices = { "*fp16:16", "*bf16:16", "*fp32:16" };
    return choices;
}

const std::vector<std::string>& DebugSimulateEncodedSoftmaxMetadata::get_dropout_p_choices()
{
    static const std::vector<std::string> choices = { "fp32" };
    return choices;
}

const std::vector<std::string>& DebugSimulateEncodedSoftmaxMetadata::get_Num_head_q_choices()
{
    static const std::vector<std::string> choices = { "i32" };
    return choices;
}

const std::vector<std::string>& DebugSimulateEncodedSoftmaxMetadata::get_philox_seed_ptr_choices()
{
    static const std::vector<std::string> choices = { "*u64" };
    return choices;
}

const std::vector<std::string>& DebugSimulateEncodedSoftmaxMetadata::get_philox_offset2_choices()
{
    static const std::vector<std::string> choices = { "u64" };
    return choices;
}

namespace autotune {

const char debug_simulate_encoded_softmax_packed_string[] =
"BLOCK_M=64;BLOCK_N=32\0"
"waves_per_eu=2;num_warps=4;num_stages=1\0";

int debug_simulate_encoded_softmax__lut_lambda__0 (const OpAttnFwdParams& params, int mod_number, int8_t lut[1][1]) {
    
    return lut[mod_number][0];
};

} // namespace autotune

DebugSimulateEncodedSoftmaxContext::AutoTuneTableEntry
DebugSimulateEncodedSoftmaxContext::autotune_table[][ 3 ] = {
    {
        &autotune::Autotune_debug_simulate_encoded_softmax__A0__F0,
        &autotune::Autotune_debug_simulate_encoded_softmax__A0__F1,
        &autotune::Autotune_debug_simulate_encoded_softmax__A0__F2,
    },
};

}

// vim: set fileencoding=utf-8

