// Copyright © 2023-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// clang-format off
#include "shim.attn_fwd.h"
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
#define KERNEL_SLOT_INDEX (attn_options::KernelSlot::attn_fwd)
#endif

#define CAST(x) const_cast<void*>(static_cast<const void*>(x))
typedef std::vector<void*>(*PP_FUNC)(const OpAttnFwdParams& context, const TritonAuxiliaryArguments&);

namespace {
extern PP_FUNC prepare_arguments[ 12 ];
}

int64_t AttnFwdContext::godel_number() const
{
    int64_t sum = 0;
    const auto& args = *params;
    {
        int64_t number = -1;
        if (args.Q->dtype() == DType::kFloat16) number = 0 ;
        if (args.Q->dtype() == DType::kBFloat16) number = 1 ;
        if (args.Q->dtype() == DType::kFloat32) number = 2 ;
        if (number < 0) {
#ifndef NDEBUG
            std::cerr << __FILE__ << ":" << __LINE__ << ": Unsupported Q, value: " << args.Q->dtype() << std::endl;
#endif
            return -1;
        }
        sum += number * 192;
    }
    {
        int64_t number = -1;
        if (args.BLOCK_DMODEL == 16) number = 0 ;
        if (args.BLOCK_DMODEL == 32) number = 1 ;
        if (args.BLOCK_DMODEL == 48) number = 2 ;
        if (args.BLOCK_DMODEL == 64) number = 3 ;
        if (args.BLOCK_DMODEL == 80) number = 4 ;
        if (args.BLOCK_DMODEL == 96) number = 5 ;
        if (args.BLOCK_DMODEL == 128) number = 6 ;
        if (args.BLOCK_DMODEL == 160) number = 7 ;
        if (args.BLOCK_DMODEL == 192) number = 8 ;
        if (args.BLOCK_DMODEL == 224) number = 9 ;
        if (args.BLOCK_DMODEL == 256) number = 10 ;
        if (args.BLOCK_DMODEL == 512) number = 11 ;
        if (number < 0) {
#ifndef NDEBUG
            std::cerr << __FILE__ << ":" << __LINE__ << ": Unsupported BLOCK_DMODEL, value: " << +args.BLOCK_DMODEL << std::endl;
#endif
            return -1;
        }
        sum += number * 16;
    }
    {
        int64_t number = -1;
        if (args.PADDED_HEAD == false) number = 0 ;
        if (args.PADDED_HEAD == true) number = 1 ;
        if (number < 0) {
#ifndef NDEBUG
            std::cerr << __FILE__ << ":" << __LINE__ << ": Unsupported PADDED_HEAD, value: " << args.PADDED_HEAD << std::endl;
#endif
            return -1;
        }
        sum += number * 8;
    }
    {
        int64_t number = -1;
        if (args.ENABLE_DROPOUT == false) number = 0 ;
        if (args.ENABLE_DROPOUT == true) number = 1 ;
        if (number < 0) {
#ifndef NDEBUG
            std::cerr << __FILE__ << ":" << __LINE__ << ": Unsupported ENABLE_DROPOUT, value: " << args.ENABLE_DROPOUT << std::endl;
#endif
            return -1;
        }
        sum += number * 4;
    }
    {
        int64_t number = -1;
        if (args.CAUSAL_TYPE == 0) number = 0 ;
        if (args.CAUSAL_TYPE == 3) number = 1 ;
        if (number < 0) {
#ifndef NDEBUG
            std::cerr << __FILE__ << ":" << __LINE__ << ": Unsupported CAUSAL_TYPE, value: " << +args.CAUSAL_TYPE << std::endl;
#endif
            return -1;
        }
        sum += number * 2;
    }
    {
        int64_t number = -1;
        if (args.BIAS_TYPE == 0) number = 0 ;
        if (args.BIAS_TYPE == 1) number = 1 ;
        if (number < 0) {
#ifndef NDEBUG
            std::cerr << __FILE__ << ":" << __LINE__ << ": Unsupported BIAS_TYPE, value: " << +args.BIAS_TYPE << std::endl;
#endif
            return -1;
        }
        sum += number * 1;
    }

    return sum;
}

hipError_t
AttnFwdContext::lookup_optimal(Gpu gpu) {
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
AttnFwdContext::launch(hipStream_t stream) const {
    if (!launch_condition)
      return hipSuccess;
    constexpr std::string_view triton_kernel_name { "attn_fwd" };
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
AttnFwdContext::get_archmod_number(Gpu gpu) {
    if (gpu == GPU_AMD_ARCH_GFX942_MOD0) return { 0, 0 };
    // TODO: print warning about tuning for this GPU mod is not built.
    // Note: if some mod does not have tuning info in the database at all, the
    //       getGpuFromStream should not return that mod from beginning.
    return std::make_tuple(-1, 0);
}


static std::vector<void*>
attn_fwd_pp_args_0(const OpAttnFwdParams& params,
                   const TritonAuxiliaryArguments& aux) {
  return { params.Q->kparam_data_ptr(), // Q
           params.K->kparam_data_ptr(), // K
           params.V->kparam_data_ptr(), // V
           // params.B->kparam_data_ptr(), // B as constexpr 0
           // params.A->kparam_data_ptr(), // A as constexpr 0
           CAST(&params.Sm_scale), // Sm_scale
           params.L->kparam_data_ptr(), // L
           params.Out->kparam_data_ptr(), // Out
           params.Q->kparam_stride(0), // stride_qz
           params.Q->kparam_stride(1), // stride_qh
           params.Q->kparam_stride(2), // stride_qm
           params.K->kparam_stride(0), // stride_kz
           params.K->kparam_stride(1), // stride_kh
           params.K->kparam_stride(2), // stride_kn
           params.V->kparam_stride(0), // stride_vz
           params.V->kparam_stride(1), // stride_vh
           params.V->kparam_stride(2), // stride_vk
           params.Out->kparam_stride(0), // stride_oz
           params.Out->kparam_stride(1), // stride_oh
           params.Out->kparam_stride(2), // stride_om
           // params.B->kparam_stride(0), // stride_bz as constexpr 0
           // params.B->kparam_stride(1), // stride_bh as constexpr 0
           // params.B->kparam_stride(2), // stride_bm as constexpr 0
           // params.A->kparam_stride(0), // stride_az as constexpr 0
           CAST(&params.Num_head_q), // Num_head_q
           CAST(&params.Num_head_k), // Num_head_k
           CAST(&params.Num_seqlens), // Num_seqlens
           params.cu_seqlens_q->kparam_data_ptr(), // cu_seqlens_q
           params.cu_seqlens_k->kparam_data_ptr(), // cu_seqlens_k
           CAST(&params.Max_seqlen_q), // Max_seqlen_q
           CAST(&params.Max_seqlen_k), // Max_seqlen_k
           params.seq_strides_q->kparam_data_ptr(), // seq_strides_q
           params.seq_strides_k->kparam_data_ptr(), // seq_strides_k
           // CAST(&params.Hdim_qk), // Hdim_qk as constexpr 16/32/48/64/80/96/128/160/192/224/256/512
           // CAST(&params.Hdim_vo), // Hdim_vo as constexpr 16/32/48/64/80/96/128/160/192/224/256/512
           // CAST(&params.dropout_p), // dropout_p as constexpr 0
           // params.philox_seed_ptr->kparam_data_ptr(), // philox_seed_ptr as constexpr 0
           // params.philox_offset1->kparam_data_ptr(), // philox_offset1 as constexpr 0
           // CAST(&params.philox_offset2), // philox_offset2 as constexpr 0
           // params.philox_seed_output->kparam_data_ptr(), // philox_seed_output as constexpr 0
           // params.philox_offset_output->kparam_data_ptr(), // philox_offset_output as constexpr 0
           // params.encoded_softmax->kparam_data_ptr(), // encoded_softmax as constexpr 0
           // CAST(&params.Window_left), // Window_left as constexpr 0
           // CAST(&params.Window_right), // Window_right as constexpr 0
           // params.persistent_atomic_counter->kparam_data_ptr(), // persistent_atomic_counter as constexpr 0
           CAST(&params.Num_CU), // Num_CU
           CAST(&params.Batch), // Batch
           CAST(&aux.global_scratch),
           CAST(&aux.profile_scratch)
         };
}
static std::vector<void*>
attn_fwd_pp_args_1(const OpAttnFwdParams& params,
                   const TritonAuxiliaryArguments& aux) {
  return { params.Q->kparam_data_ptr(), // Q
           params.K->kparam_data_ptr(), // K
           params.V->kparam_data_ptr(), // V
           params.B->kparam_data_ptr(), // B
           // params.A->kparam_data_ptr(), // A as constexpr 0
           CAST(&params.Sm_scale), // Sm_scale
           params.L->kparam_data_ptr(), // L
           params.Out->kparam_data_ptr(), // Out
           params.Q->kparam_stride(0), // stride_qz
           params.Q->kparam_stride(1), // stride_qh
           params.Q->kparam_stride(2), // stride_qm
           params.K->kparam_stride(0), // stride_kz
           params.K->kparam_stride(1), // stride_kh
           params.K->kparam_stride(2), // stride_kn
           params.V->kparam_stride(0), // stride_vz
           params.V->kparam_stride(1), // stride_vh
           params.V->kparam_stride(2), // stride_vk
           params.Out->kparam_stride(0), // stride_oz
           params.Out->kparam_stride(1), // stride_oh
           params.Out->kparam_stride(2), // stride_om
           params.B->kparam_stride(0), // stride_bz
           params.B->kparam_stride(1), // stride_bh
           params.B->kparam_stride(2), // stride_bm
           // params.A->kparam_stride(0), // stride_az as constexpr 0
           CAST(&params.Num_head_q), // Num_head_q
           CAST(&params.Num_head_k), // Num_head_k
           CAST(&params.Num_seqlens), // Num_seqlens
           params.cu_seqlens_q->kparam_data_ptr(), // cu_seqlens_q
           params.cu_seqlens_k->kparam_data_ptr(), // cu_seqlens_k
           CAST(&params.Max_seqlen_q), // Max_seqlen_q
           CAST(&params.Max_seqlen_k), // Max_seqlen_k
           params.seq_strides_q->kparam_data_ptr(), // seq_strides_q
           params.seq_strides_k->kparam_data_ptr(), // seq_strides_k
           // CAST(&params.Hdim_qk), // Hdim_qk as constexpr 16/32/48/64/80/96/128/160/192/224/256/512
           // CAST(&params.Hdim_vo), // Hdim_vo as constexpr 16/32/48/64/80/96/128/160/192/224/256/512
           // CAST(&params.dropout_p), // dropout_p as constexpr 0
           // params.philox_seed_ptr->kparam_data_ptr(), // philox_seed_ptr as constexpr 0
           // params.philox_offset1->kparam_data_ptr(), // philox_offset1 as constexpr 0
           // CAST(&params.philox_offset2), // philox_offset2 as constexpr 0
           // params.philox_seed_output->kparam_data_ptr(), // philox_seed_output as constexpr 0
           // params.philox_offset_output->kparam_data_ptr(), // philox_offset_output as constexpr 0
           // params.encoded_softmax->kparam_data_ptr(), // encoded_softmax as constexpr 0
           // CAST(&params.Window_left), // Window_left as constexpr 0
           // CAST(&params.Window_right), // Window_right as constexpr 0
           // params.persistent_atomic_counter->kparam_data_ptr(), // persistent_atomic_counter as constexpr 0
           CAST(&params.Num_CU), // Num_CU
           CAST(&params.Batch), // Batch
           CAST(&aux.global_scratch),
           CAST(&aux.profile_scratch)
         };
}
static std::vector<void*>
attn_fwd_pp_args_2(const OpAttnFwdParams& params,
                   const TritonAuxiliaryArguments& aux) {
  return { params.Q->kparam_data_ptr(), // Q
           params.K->kparam_data_ptr(), // K
           params.V->kparam_data_ptr(), // V
           // params.B->kparam_data_ptr(), // B as constexpr 0
           // params.A->kparam_data_ptr(), // A as constexpr 0
           CAST(&params.Sm_scale), // Sm_scale
           params.L->kparam_data_ptr(), // L
           params.Out->kparam_data_ptr(), // Out
           params.Q->kparam_stride(0), // stride_qz
           params.Q->kparam_stride(1), // stride_qh
           params.Q->kparam_stride(2), // stride_qm
           params.K->kparam_stride(0), // stride_kz
           params.K->kparam_stride(1), // stride_kh
           params.K->kparam_stride(2), // stride_kn
           params.V->kparam_stride(0), // stride_vz
           params.V->kparam_stride(1), // stride_vh
           params.V->kparam_stride(2), // stride_vk
           params.Out->kparam_stride(0), // stride_oz
           params.Out->kparam_stride(1), // stride_oh
           params.Out->kparam_stride(2), // stride_om
           // params.B->kparam_stride(0), // stride_bz as constexpr 0
           // params.B->kparam_stride(1), // stride_bh as constexpr 0
           // params.B->kparam_stride(2), // stride_bm as constexpr 0
           // params.A->kparam_stride(0), // stride_az as constexpr 0
           CAST(&params.Num_head_q), // Num_head_q
           CAST(&params.Num_head_k), // Num_head_k
           CAST(&params.Num_seqlens), // Num_seqlens
           params.cu_seqlens_q->kparam_data_ptr(), // cu_seqlens_q
           params.cu_seqlens_k->kparam_data_ptr(), // cu_seqlens_k
           CAST(&params.Max_seqlen_q), // Max_seqlen_q
           CAST(&params.Max_seqlen_k), // Max_seqlen_k
           params.seq_strides_q->kparam_data_ptr(), // seq_strides_q
           params.seq_strides_k->kparam_data_ptr(), // seq_strides_k
           // CAST(&params.Hdim_qk), // Hdim_qk as constexpr 16/32/48/64/80/96/128/160/192/224/256/512
           // CAST(&params.Hdim_vo), // Hdim_vo as constexpr 16/32/48/64/80/96/128/160/192/224/256/512
           // CAST(&params.dropout_p), // dropout_p as constexpr 0
           // params.philox_seed_ptr->kparam_data_ptr(), // philox_seed_ptr as constexpr 0
           // params.philox_offset1->kparam_data_ptr(), // philox_offset1 as constexpr 0
           // CAST(&params.philox_offset2), // philox_offset2 as constexpr 0
           // params.philox_seed_output->kparam_data_ptr(), // philox_seed_output as constexpr 0
           // params.philox_offset_output->kparam_data_ptr(), // philox_offset_output as constexpr 0
           // params.encoded_softmax->kparam_data_ptr(), // encoded_softmax as constexpr 0
           CAST(&params.Window_left), // Window_left
           CAST(&params.Window_right), // Window_right
           params.persistent_atomic_counter->kparam_data_ptr(), // persistent_atomic_counter
           CAST(&params.Num_CU), // Num_CU
           CAST(&params.Batch), // Batch
           CAST(&aux.global_scratch),
           CAST(&aux.profile_scratch)
         };
}
static std::vector<void*>
attn_fwd_pp_args_3(const OpAttnFwdParams& params,
                   const TritonAuxiliaryArguments& aux) {
  return { params.Q->kparam_data_ptr(), // Q
           params.K->kparam_data_ptr(), // K
           params.V->kparam_data_ptr(), // V
           // params.B->kparam_data_ptr(), // B as constexpr 0
           // params.A->kparam_data_ptr(), // A as constexpr 0
           CAST(&params.Sm_scale), // Sm_scale
           params.L->kparam_data_ptr(), // L
           params.Out->kparam_data_ptr(), // Out
           params.Q->kparam_stride(0), // stride_qz
           params.Q->kparam_stride(1), // stride_qh
           params.Q->kparam_stride(2), // stride_qm
           params.K->kparam_stride(0), // stride_kz
           params.K->kparam_stride(1), // stride_kh
           params.K->kparam_stride(2), // stride_kn
           params.V->kparam_stride(0), // stride_vz
           params.V->kparam_stride(1), // stride_vh
           params.V->kparam_stride(2), // stride_vk
           params.Out->kparam_stride(0), // stride_oz
           params.Out->kparam_stride(1), // stride_oh
           params.Out->kparam_stride(2), // stride_om
           // params.B->kparam_stride(0), // stride_bz as constexpr 0
           // params.B->kparam_stride(1), // stride_bh as constexpr 0
           // params.B->kparam_stride(2), // stride_bm as constexpr 0
           // params.A->kparam_stride(0), // stride_az as constexpr 0
           CAST(&params.Num_head_q), // Num_head_q
           CAST(&params.Num_head_k), // Num_head_k
           CAST(&params.Num_seqlens), // Num_seqlens
           params.cu_seqlens_q->kparam_data_ptr(), // cu_seqlens_q
           params.cu_seqlens_k->kparam_data_ptr(), // cu_seqlens_k
           CAST(&params.Max_seqlen_q), // Max_seqlen_q
           CAST(&params.Max_seqlen_k), // Max_seqlen_k
           params.seq_strides_q->kparam_data_ptr(), // seq_strides_q
           params.seq_strides_k->kparam_data_ptr(), // seq_strides_k
           // CAST(&params.Hdim_qk), // Hdim_qk as constexpr 16/32/48/64/80/96/128/160/192/224/256/512
           // CAST(&params.Hdim_vo), // Hdim_vo as constexpr 16/32/48/64/80/96/128/160/192/224/256/512
           CAST(&params.dropout_p), // dropout_p
           params.philox_seed_ptr->kparam_data_ptr(), // philox_seed_ptr
           params.philox_offset1->kparam_data_ptr(), // philox_offset1
           CAST(&params.philox_offset2), // philox_offset2
           params.philox_seed_output->kparam_data_ptr(), // philox_seed_output
           params.philox_offset_output->kparam_data_ptr(), // philox_offset_output
           // params.encoded_softmax->kparam_data_ptr(), // encoded_softmax as constexpr 0
           // CAST(&params.Window_left), // Window_left as constexpr 0
           // CAST(&params.Window_right), // Window_right as constexpr 0
           // params.persistent_atomic_counter->kparam_data_ptr(), // persistent_atomic_counter as constexpr 0
           CAST(&params.Num_CU), // Num_CU
           CAST(&params.Batch), // Batch
           CAST(&aux.global_scratch),
           CAST(&aux.profile_scratch)
         };
}
static std::vector<void*>
attn_fwd_pp_args_4(const OpAttnFwdParams& params,
                   const TritonAuxiliaryArguments& aux) {
  return { params.Q->kparam_data_ptr(), // Q
           params.K->kparam_data_ptr(), // K
           params.V->kparam_data_ptr(), // V
           params.B->kparam_data_ptr(), // B
           // params.A->kparam_data_ptr(), // A as constexpr 0
           CAST(&params.Sm_scale), // Sm_scale
           params.L->kparam_data_ptr(), // L
           params.Out->kparam_data_ptr(), // Out
           params.Q->kparam_stride(0), // stride_qz
           params.Q->kparam_stride(1), // stride_qh
           params.Q->kparam_stride(2), // stride_qm
           params.K->kparam_stride(0), // stride_kz
           params.K->kparam_stride(1), // stride_kh
           params.K->kparam_stride(2), // stride_kn
           params.V->kparam_stride(0), // stride_vz
           params.V->kparam_stride(1), // stride_vh
           params.V->kparam_stride(2), // stride_vk
           params.Out->kparam_stride(0), // stride_oz
           params.Out->kparam_stride(1), // stride_oh
           params.Out->kparam_stride(2), // stride_om
           params.B->kparam_stride(0), // stride_bz
           params.B->kparam_stride(1), // stride_bh
           params.B->kparam_stride(2), // stride_bm
           // params.A->kparam_stride(0), // stride_az as constexpr 0
           CAST(&params.Num_head_q), // Num_head_q
           CAST(&params.Num_head_k), // Num_head_k
           CAST(&params.Num_seqlens), // Num_seqlens
           params.cu_seqlens_q->kparam_data_ptr(), // cu_seqlens_q
           params.cu_seqlens_k->kparam_data_ptr(), // cu_seqlens_k
           CAST(&params.Max_seqlen_q), // Max_seqlen_q
           CAST(&params.Max_seqlen_k), // Max_seqlen_k
           params.seq_strides_q->kparam_data_ptr(), // seq_strides_q
           params.seq_strides_k->kparam_data_ptr(), // seq_strides_k
           // CAST(&params.Hdim_qk), // Hdim_qk as constexpr 16/32/48/64/80/96/128/160/192/224/256/512
           // CAST(&params.Hdim_vo), // Hdim_vo as constexpr 16/32/48/64/80/96/128/160/192/224/256/512
           CAST(&params.dropout_p), // dropout_p
           params.philox_seed_ptr->kparam_data_ptr(), // philox_seed_ptr
           params.philox_offset1->kparam_data_ptr(), // philox_offset1
           CAST(&params.philox_offset2), // philox_offset2
           params.philox_seed_output->kparam_data_ptr(), // philox_seed_output
           params.philox_offset_output->kparam_data_ptr(), // philox_offset_output
           // params.encoded_softmax->kparam_data_ptr(), // encoded_softmax as constexpr 0
           // CAST(&params.Window_left), // Window_left as constexpr 0
           // CAST(&params.Window_right), // Window_right as constexpr 0
           // params.persistent_atomic_counter->kparam_data_ptr(), // persistent_atomic_counter as constexpr 0
           CAST(&params.Num_CU), // Num_CU
           CAST(&params.Batch), // Batch
           CAST(&aux.global_scratch),
           CAST(&aux.profile_scratch)
         };
}
static std::vector<void*>
attn_fwd_pp_args_5(const OpAttnFwdParams& params,
                   const TritonAuxiliaryArguments& aux) {
  return { params.Q->kparam_data_ptr(), // Q
           params.K->kparam_data_ptr(), // K
           params.V->kparam_data_ptr(), // V
           // params.B->kparam_data_ptr(), // B as constexpr 0
           // params.A->kparam_data_ptr(), // A as constexpr 0
           CAST(&params.Sm_scale), // Sm_scale
           params.L->kparam_data_ptr(), // L
           params.Out->kparam_data_ptr(), // Out
           params.Q->kparam_stride(0), // stride_qz
           params.Q->kparam_stride(1), // stride_qh
           params.Q->kparam_stride(2), // stride_qm
           params.K->kparam_stride(0), // stride_kz
           params.K->kparam_stride(1), // stride_kh
           params.K->kparam_stride(2), // stride_kn
           params.V->kparam_stride(0), // stride_vz
           params.V->kparam_stride(1), // stride_vh
           params.V->kparam_stride(2), // stride_vk
           params.Out->kparam_stride(0), // stride_oz
           params.Out->kparam_stride(1), // stride_oh
           params.Out->kparam_stride(2), // stride_om
           // params.B->kparam_stride(0), // stride_bz as constexpr 0
           // params.B->kparam_stride(1), // stride_bh as constexpr 0
           // params.B->kparam_stride(2), // stride_bm as constexpr 0
           // params.A->kparam_stride(0), // stride_az as constexpr 0
           CAST(&params.Num_head_q), // Num_head_q
           CAST(&params.Num_head_k), // Num_head_k
           CAST(&params.Num_seqlens), // Num_seqlens
           params.cu_seqlens_q->kparam_data_ptr(), // cu_seqlens_q
           params.cu_seqlens_k->kparam_data_ptr(), // cu_seqlens_k
           CAST(&params.Max_seqlen_q), // Max_seqlen_q
           CAST(&params.Max_seqlen_k), // Max_seqlen_k
           params.seq_strides_q->kparam_data_ptr(), // seq_strides_q
           params.seq_strides_k->kparam_data_ptr(), // seq_strides_k
           // CAST(&params.Hdim_qk), // Hdim_qk as constexpr 16/32/48/64/80/96/128/160/192/224/256/512
           // CAST(&params.Hdim_vo), // Hdim_vo as constexpr 16/32/48/64/80/96/128/160/192/224/256/512
           CAST(&params.dropout_p), // dropout_p
           params.philox_seed_ptr->kparam_data_ptr(), // philox_seed_ptr
           params.philox_offset1->kparam_data_ptr(), // philox_offset1
           CAST(&params.philox_offset2), // philox_offset2
           params.philox_seed_output->kparam_data_ptr(), // philox_seed_output
           params.philox_offset_output->kparam_data_ptr(), // philox_offset_output
           // params.encoded_softmax->kparam_data_ptr(), // encoded_softmax as constexpr 0
           CAST(&params.Window_left), // Window_left
           CAST(&params.Window_right), // Window_right
           params.persistent_atomic_counter->kparam_data_ptr(), // persistent_atomic_counter
           CAST(&params.Num_CU), // Num_CU
           CAST(&params.Batch), // Batch
           CAST(&aux.global_scratch),
           CAST(&aux.profile_scratch)
         };
}
static std::vector<void*>
attn_fwd_pp_args_6(const OpAttnFwdParams& params,
                   const TritonAuxiliaryArguments& aux) {
  return { params.Q->kparam_data_ptr(), // Q
           params.K->kparam_data_ptr(), // K
           params.V->kparam_data_ptr(), // V
           // params.B->kparam_data_ptr(), // B as constexpr 0
           // params.A->kparam_data_ptr(), // A as constexpr 0
           CAST(&params.Sm_scale), // Sm_scale
           params.L->kparam_data_ptr(), // L
           params.Out->kparam_data_ptr(), // Out
           params.Q->kparam_stride(0), // stride_qz
           params.Q->kparam_stride(1), // stride_qh
           params.Q->kparam_stride(2), // stride_qm
           params.K->kparam_stride(0), // stride_kz
           params.K->kparam_stride(1), // stride_kh
           params.K->kparam_stride(2), // stride_kn
           params.V->kparam_stride(0), // stride_vz
           params.V->kparam_stride(1), // stride_vh
           params.V->kparam_stride(2), // stride_vk
           params.Out->kparam_stride(0), // stride_oz
           params.Out->kparam_stride(1), // stride_oh
           params.Out->kparam_stride(2), // stride_om
           // params.B->kparam_stride(0), // stride_bz as constexpr 0
           // params.B->kparam_stride(1), // stride_bh as constexpr 0
           // params.B->kparam_stride(2), // stride_bm as constexpr 0
           // params.A->kparam_stride(0), // stride_az as constexpr 0
           CAST(&params.Num_head_q), // Num_head_q
           CAST(&params.Num_head_k), // Num_head_k
           CAST(&params.Num_seqlens), // Num_seqlens
           params.cu_seqlens_q->kparam_data_ptr(), // cu_seqlens_q
           params.cu_seqlens_k->kparam_data_ptr(), // cu_seqlens_k
           CAST(&params.Max_seqlen_q), // Max_seqlen_q
           CAST(&params.Max_seqlen_k), // Max_seqlen_k
           params.seq_strides_q->kparam_data_ptr(), // seq_strides_q
           params.seq_strides_k->kparam_data_ptr(), // seq_strides_k
           CAST(&params.Hdim_qk), // Hdim_qk
           CAST(&params.Hdim_vo), // Hdim_vo
           // CAST(&params.dropout_p), // dropout_p as constexpr 0
           // params.philox_seed_ptr->kparam_data_ptr(), // philox_seed_ptr as constexpr 0
           // params.philox_offset1->kparam_data_ptr(), // philox_offset1 as constexpr 0
           // CAST(&params.philox_offset2), // philox_offset2 as constexpr 0
           // params.philox_seed_output->kparam_data_ptr(), // philox_seed_output as constexpr 0
           // params.philox_offset_output->kparam_data_ptr(), // philox_offset_output as constexpr 0
           // params.encoded_softmax->kparam_data_ptr(), // encoded_softmax as constexpr 0
           // CAST(&params.Window_left), // Window_left as constexpr 0
           // CAST(&params.Window_right), // Window_right as constexpr 0
           // params.persistent_atomic_counter->kparam_data_ptr(), // persistent_atomic_counter as constexpr 0
           CAST(&params.Num_CU), // Num_CU
           CAST(&params.Batch), // Batch
           CAST(&aux.global_scratch),
           CAST(&aux.profile_scratch)
         };
}
static std::vector<void*>
attn_fwd_pp_args_7(const OpAttnFwdParams& params,
                   const TritonAuxiliaryArguments& aux) {
  return { params.Q->kparam_data_ptr(), // Q
           params.K->kparam_data_ptr(), // K
           params.V->kparam_data_ptr(), // V
           params.B->kparam_data_ptr(), // B
           // params.A->kparam_data_ptr(), // A as constexpr 0
           CAST(&params.Sm_scale), // Sm_scale
           params.L->kparam_data_ptr(), // L
           params.Out->kparam_data_ptr(), // Out
           params.Q->kparam_stride(0), // stride_qz
           params.Q->kparam_stride(1), // stride_qh
           params.Q->kparam_stride(2), // stride_qm
           params.K->kparam_stride(0), // stride_kz
           params.K->kparam_stride(1), // stride_kh
           params.K->kparam_stride(2), // stride_kn
           params.V->kparam_stride(0), // stride_vz
           params.V->kparam_stride(1), // stride_vh
           params.V->kparam_stride(2), // stride_vk
           params.Out->kparam_stride(0), // stride_oz
           params.Out->kparam_stride(1), // stride_oh
           params.Out->kparam_stride(2), // stride_om
           params.B->kparam_stride(0), // stride_bz
           params.B->kparam_stride(1), // stride_bh
           params.B->kparam_stride(2), // stride_bm
           // params.A->kparam_stride(0), // stride_az as constexpr 0
           CAST(&params.Num_head_q), // Num_head_q
           CAST(&params.Num_head_k), // Num_head_k
           CAST(&params.Num_seqlens), // Num_seqlens
           params.cu_seqlens_q->kparam_data_ptr(), // cu_seqlens_q
           params.cu_seqlens_k->kparam_data_ptr(), // cu_seqlens_k
           CAST(&params.Max_seqlen_q), // Max_seqlen_q
           CAST(&params.Max_seqlen_k), // Max_seqlen_k
           params.seq_strides_q->kparam_data_ptr(), // seq_strides_q
           params.seq_strides_k->kparam_data_ptr(), // seq_strides_k
           CAST(&params.Hdim_qk), // Hdim_qk
           CAST(&params.Hdim_vo), // Hdim_vo
           // CAST(&params.dropout_p), // dropout_p as constexpr 0
           // params.philox_seed_ptr->kparam_data_ptr(), // philox_seed_ptr as constexpr 0
           // params.philox_offset1->kparam_data_ptr(), // philox_offset1 as constexpr 0
           // CAST(&params.philox_offset2), // philox_offset2 as constexpr 0
           // params.philox_seed_output->kparam_data_ptr(), // philox_seed_output as constexpr 0
           // params.philox_offset_output->kparam_data_ptr(), // philox_offset_output as constexpr 0
           // params.encoded_softmax->kparam_data_ptr(), // encoded_softmax as constexpr 0
           // CAST(&params.Window_left), // Window_left as constexpr 0
           // CAST(&params.Window_right), // Window_right as constexpr 0
           // params.persistent_atomic_counter->kparam_data_ptr(), // persistent_atomic_counter as constexpr 0
           CAST(&params.Num_CU), // Num_CU
           CAST(&params.Batch), // Batch
           CAST(&aux.global_scratch),
           CAST(&aux.profile_scratch)
         };
}
static std::vector<void*>
attn_fwd_pp_args_8(const OpAttnFwdParams& params,
                   const TritonAuxiliaryArguments& aux) {
  return { params.Q->kparam_data_ptr(), // Q
           params.K->kparam_data_ptr(), // K
           params.V->kparam_data_ptr(), // V
           // params.B->kparam_data_ptr(), // B as constexpr 0
           // params.A->kparam_data_ptr(), // A as constexpr 0
           CAST(&params.Sm_scale), // Sm_scale
           params.L->kparam_data_ptr(), // L
           params.Out->kparam_data_ptr(), // Out
           params.Q->kparam_stride(0), // stride_qz
           params.Q->kparam_stride(1), // stride_qh
           params.Q->kparam_stride(2), // stride_qm
           params.K->kparam_stride(0), // stride_kz
           params.K->kparam_stride(1), // stride_kh
           params.K->kparam_stride(2), // stride_kn
           params.V->kparam_stride(0), // stride_vz
           params.V->kparam_stride(1), // stride_vh
           params.V->kparam_stride(2), // stride_vk
           params.Out->kparam_stride(0), // stride_oz
           params.Out->kparam_stride(1), // stride_oh
           params.Out->kparam_stride(2), // stride_om
           // params.B->kparam_stride(0), // stride_bz as constexpr 0
           // params.B->kparam_stride(1), // stride_bh as constexpr 0
           // params.B->kparam_stride(2), // stride_bm as constexpr 0
           // params.A->kparam_stride(0), // stride_az as constexpr 0
           CAST(&params.Num_head_q), // Num_head_q
           CAST(&params.Num_head_k), // Num_head_k
           CAST(&params.Num_seqlens), // Num_seqlens
           params.cu_seqlens_q->kparam_data_ptr(), // cu_seqlens_q
           params.cu_seqlens_k->kparam_data_ptr(), // cu_seqlens_k
           CAST(&params.Max_seqlen_q), // Max_seqlen_q
           CAST(&params.Max_seqlen_k), // Max_seqlen_k
           params.seq_strides_q->kparam_data_ptr(), // seq_strides_q
           params.seq_strides_k->kparam_data_ptr(), // seq_strides_k
           CAST(&params.Hdim_qk), // Hdim_qk
           CAST(&params.Hdim_vo), // Hdim_vo
           // CAST(&params.dropout_p), // dropout_p as constexpr 0
           // params.philox_seed_ptr->kparam_data_ptr(), // philox_seed_ptr as constexpr 0
           // params.philox_offset1->kparam_data_ptr(), // philox_offset1 as constexpr 0
           // CAST(&params.philox_offset2), // philox_offset2 as constexpr 0
           // params.philox_seed_output->kparam_data_ptr(), // philox_seed_output as constexpr 0
           // params.philox_offset_output->kparam_data_ptr(), // philox_offset_output as constexpr 0
           // params.encoded_softmax->kparam_data_ptr(), // encoded_softmax as constexpr 0
           CAST(&params.Window_left), // Window_left
           CAST(&params.Window_right), // Window_right
           params.persistent_atomic_counter->kparam_data_ptr(), // persistent_atomic_counter
           CAST(&params.Num_CU), // Num_CU
           CAST(&params.Batch), // Batch
           CAST(&aux.global_scratch),
           CAST(&aux.profile_scratch)
         };
}
static std::vector<void*>
attn_fwd_pp_args_9(const OpAttnFwdParams& params,
                   const TritonAuxiliaryArguments& aux) {
  return { params.Q->kparam_data_ptr(), // Q
           params.K->kparam_data_ptr(), // K
           params.V->kparam_data_ptr(), // V
           // params.B->kparam_data_ptr(), // B as constexpr 0
           // params.A->kparam_data_ptr(), // A as constexpr 0
           CAST(&params.Sm_scale), // Sm_scale
           params.L->kparam_data_ptr(), // L
           params.Out->kparam_data_ptr(), // Out
           params.Q->kparam_stride(0), // stride_qz
           params.Q->kparam_stride(1), // stride_qh
           params.Q->kparam_stride(2), // stride_qm
           params.K->kparam_stride(0), // stride_kz
           params.K->kparam_stride(1), // stride_kh
           params.K->kparam_stride(2), // stride_kn
           params.V->kparam_stride(0), // stride_vz
           params.V->kparam_stride(1), // stride_vh
           params.V->kparam_stride(2), // stride_vk
           params.Out->kparam_stride(0), // stride_oz
           params.Out->kparam_stride(1), // stride_oh
           params.Out->kparam_stride(2), // stride_om
           // params.B->kparam_stride(0), // stride_bz as constexpr 0
           // params.B->kparam_stride(1), // stride_bh as constexpr 0
           // params.B->kparam_stride(2), // stride_bm as constexpr 0
           // params.A->kparam_stride(0), // stride_az as constexpr 0
           CAST(&params.Num_head_q), // Num_head_q
           CAST(&params.Num_head_k), // Num_head_k
           CAST(&params.Num_seqlens), // Num_seqlens
           params.cu_seqlens_q->kparam_data_ptr(), // cu_seqlens_q
           params.cu_seqlens_k->kparam_data_ptr(), // cu_seqlens_k
           CAST(&params.Max_seqlen_q), // Max_seqlen_q
           CAST(&params.Max_seqlen_k), // Max_seqlen_k
           params.seq_strides_q->kparam_data_ptr(), // seq_strides_q
           params.seq_strides_k->kparam_data_ptr(), // seq_strides_k
           CAST(&params.Hdim_qk), // Hdim_qk
           CAST(&params.Hdim_vo), // Hdim_vo
           CAST(&params.dropout_p), // dropout_p
           params.philox_seed_ptr->kparam_data_ptr(), // philox_seed_ptr
           params.philox_offset1->kparam_data_ptr(), // philox_offset1
           CAST(&params.philox_offset2), // philox_offset2
           params.philox_seed_output->kparam_data_ptr(), // philox_seed_output
           params.philox_offset_output->kparam_data_ptr(), // philox_offset_output
           // params.encoded_softmax->kparam_data_ptr(), // encoded_softmax as constexpr 0
           // CAST(&params.Window_left), // Window_left as constexpr 0
           // CAST(&params.Window_right), // Window_right as constexpr 0
           // params.persistent_atomic_counter->kparam_data_ptr(), // persistent_atomic_counter as constexpr 0
           CAST(&params.Num_CU), // Num_CU
           CAST(&params.Batch), // Batch
           CAST(&aux.global_scratch),
           CAST(&aux.profile_scratch)
         };
}
static std::vector<void*>
attn_fwd_pp_args_10(const OpAttnFwdParams& params,
                    const TritonAuxiliaryArguments& aux) {
  return { params.Q->kparam_data_ptr(), // Q
           params.K->kparam_data_ptr(), // K
           params.V->kparam_data_ptr(), // V
           params.B->kparam_data_ptr(), // B
           // params.A->kparam_data_ptr(), // A as constexpr 0
           CAST(&params.Sm_scale), // Sm_scale
           params.L->kparam_data_ptr(), // L
           params.Out->kparam_data_ptr(), // Out
           params.Q->kparam_stride(0), // stride_qz
           params.Q->kparam_stride(1), // stride_qh
           params.Q->kparam_stride(2), // stride_qm
           params.K->kparam_stride(0), // stride_kz
           params.K->kparam_stride(1), // stride_kh
           params.K->kparam_stride(2), // stride_kn
           params.V->kparam_stride(0), // stride_vz
           params.V->kparam_stride(1), // stride_vh
           params.V->kparam_stride(2), // stride_vk
           params.Out->kparam_stride(0), // stride_oz
           params.Out->kparam_stride(1), // stride_oh
           params.Out->kparam_stride(2), // stride_om
           params.B->kparam_stride(0), // stride_bz
           params.B->kparam_stride(1), // stride_bh
           params.B->kparam_stride(2), // stride_bm
           // params.A->kparam_stride(0), // stride_az as constexpr 0
           CAST(&params.Num_head_q), // Num_head_q
           CAST(&params.Num_head_k), // Num_head_k
           CAST(&params.Num_seqlens), // Num_seqlens
           params.cu_seqlens_q->kparam_data_ptr(), // cu_seqlens_q
           params.cu_seqlens_k->kparam_data_ptr(), // cu_seqlens_k
           CAST(&params.Max_seqlen_q), // Max_seqlen_q
           CAST(&params.Max_seqlen_k), // Max_seqlen_k
           params.seq_strides_q->kparam_data_ptr(), // seq_strides_q
           params.seq_strides_k->kparam_data_ptr(), // seq_strides_k
           CAST(&params.Hdim_qk), // Hdim_qk
           CAST(&params.Hdim_vo), // Hdim_vo
           CAST(&params.dropout_p), // dropout_p
           params.philox_seed_ptr->kparam_data_ptr(), // philox_seed_ptr
           params.philox_offset1->kparam_data_ptr(), // philox_offset1
           CAST(&params.philox_offset2), // philox_offset2
           params.philox_seed_output->kparam_data_ptr(), // philox_seed_output
           params.philox_offset_output->kparam_data_ptr(), // philox_offset_output
           // params.encoded_softmax->kparam_data_ptr(), // encoded_softmax as constexpr 0
           // CAST(&params.Window_left), // Window_left as constexpr 0
           // CAST(&params.Window_right), // Window_right as constexpr 0
           // params.persistent_atomic_counter->kparam_data_ptr(), // persistent_atomic_counter as constexpr 0
           CAST(&params.Num_CU), // Num_CU
           CAST(&params.Batch), // Batch
           CAST(&aux.global_scratch),
           CAST(&aux.profile_scratch)
         };
}
static std::vector<void*>
attn_fwd_pp_args_11(const OpAttnFwdParams& params,
                    const TritonAuxiliaryArguments& aux) {
  return { params.Q->kparam_data_ptr(), // Q
           params.K->kparam_data_ptr(), // K
           params.V->kparam_data_ptr(), // V
           // params.B->kparam_data_ptr(), // B as constexpr 0
           // params.A->kparam_data_ptr(), // A as constexpr 0
           CAST(&params.Sm_scale), // Sm_scale
           params.L->kparam_data_ptr(), // L
           params.Out->kparam_data_ptr(), // Out
           params.Q->kparam_stride(0), // stride_qz
           params.Q->kparam_stride(1), // stride_qh
           params.Q->kparam_stride(2), // stride_qm
           params.K->kparam_stride(0), // stride_kz
           params.K->kparam_stride(1), // stride_kh
           params.K->kparam_stride(2), // stride_kn
           params.V->kparam_stride(0), // stride_vz
           params.V->kparam_stride(1), // stride_vh
           params.V->kparam_stride(2), // stride_vk
           params.Out->kparam_stride(0), // stride_oz
           params.Out->kparam_stride(1), // stride_oh
           params.Out->kparam_stride(2), // stride_om
           // params.B->kparam_stride(0), // stride_bz as constexpr 0
           // params.B->kparam_stride(1), // stride_bh as constexpr 0
           // params.B->kparam_stride(2), // stride_bm as constexpr 0
           // params.A->kparam_stride(0), // stride_az as constexpr 0
           CAST(&params.Num_head_q), // Num_head_q
           CAST(&params.Num_head_k), // Num_head_k
           CAST(&params.Num_seqlens), // Num_seqlens
           params.cu_seqlens_q->kparam_data_ptr(), // cu_seqlens_q
           params.cu_seqlens_k->kparam_data_ptr(), // cu_seqlens_k
           CAST(&params.Max_seqlen_q), // Max_seqlen_q
           CAST(&params.Max_seqlen_k), // Max_seqlen_k
           params.seq_strides_q->kparam_data_ptr(), // seq_strides_q
           params.seq_strides_k->kparam_data_ptr(), // seq_strides_k
           CAST(&params.Hdim_qk), // Hdim_qk
           CAST(&params.Hdim_vo), // Hdim_vo
           CAST(&params.dropout_p), // dropout_p
           params.philox_seed_ptr->kparam_data_ptr(), // philox_seed_ptr
           params.philox_offset1->kparam_data_ptr(), // philox_offset1
           CAST(&params.philox_offset2), // philox_offset2
           params.philox_seed_output->kparam_data_ptr(), // philox_seed_output
           params.philox_offset_output->kparam_data_ptr(), // philox_offset_output
           // params.encoded_softmax->kparam_data_ptr(), // encoded_softmax as constexpr 0
           CAST(&params.Window_left), // Window_left
           CAST(&params.Window_right), // Window_right
           params.persistent_atomic_counter->kparam_data_ptr(), // persistent_atomic_counter
           CAST(&params.Num_CU), // Num_CU
           CAST(&params.Batch), // Batch
           CAST(&aux.global_scratch),
           CAST(&aux.profile_scratch)
         };
}

namespace {
PP_FUNC prepare_arguments[ 12 ] = {
  attn_fwd_pp_args_0,
  attn_fwd_pp_args_1,
  attn_fwd_pp_args_2,
  attn_fwd_pp_args_3,
  attn_fwd_pp_args_4,
  attn_fwd_pp_args_5,
  attn_fwd_pp_args_6,
  attn_fwd_pp_args_7,
  attn_fwd_pp_args_8,
  attn_fwd_pp_args_9,
  attn_fwd_pp_args_10,
  attn_fwd_pp_args_11
};
}


const std::vector<std::string>& AttnFwdMetadata::get_Q_choices()
{
    static const std::vector<std::string> choices = { "*fp16:16", "*bf16:16", "*fp32:16" };
    return choices;
}

const std::vector<std::string>& AttnFwdMetadata::get_Sm_scale_choices()
{
    static const std::vector<std::string> choices = { "fp32" };
    return choices;
}

const std::vector<std::string>& AttnFwdMetadata::get_L_choices()
{
    static const std::vector<std::string> choices = { "*fp32:16" };
    return choices;
}

const std::vector<int>& AttnFwdMetadata::get_Q_descale_choices()
{
    static const std::vector<int> choices = { 0 };
    return choices;
}

const std::vector<std::string>& AttnFwdMetadata::get_Num_head_q_choices()
{
    static const std::vector<std::string> choices = { "i32" };
    return choices;
}

const std::vector<std::string>& AttnFwdMetadata::get_cu_seqlens_q_choices()
{
    static const std::vector<std::string> choices = { "*i32:16" };
    return choices;
}

const std::vector<int>& AttnFwdMetadata::get_BLOCK_DMODEL_choices()
{
    static const std::vector<int> choices = { 16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 512 };
    return choices;
}

const std::vector<bool>& AttnFwdMetadata::get_PADDED_HEAD_choices()
{
    static const std::vector<bool> choices = { false, true };
    return choices;
}

const std::vector<bool>& AttnFwdMetadata::get_ENABLE_DROPOUT_choices()
{
    static const std::vector<bool> choices = { false, true };
    return choices;
}

const std::vector<bool>& AttnFwdMetadata::get_RETURN_ENCODED_SOFTMAX_choices()
{
    static const std::vector<bool> choices = { false };
    return choices;
}

const std::vector<int>& AttnFwdMetadata::get_CAUSAL_TYPE_choices()
{
    static const std::vector<int> choices = { 0, 3 };
    return choices;
}

const std::vector<int>& AttnFwdMetadata::get_BIAS_TYPE_choices()
{
    static const std::vector<int> choices = { 0, 1 };
    return choices;
}

const std::vector<bool>& AttnFwdMetadata::get_USE_ALIBI_choices()
{
    static const std::vector<bool> choices = { false };
    return choices;
}

const std::vector<bool>& AttnFwdMetadata::get_INT8_choices()
{
    static const std::vector<bool> choices = { false };
    return choices;
}

const std::vector<std::string>& AttnFwdMetadata::get_Num_CU_choices()
{
    static const std::vector<std::string> choices = { "i32" };
    return choices;
}

namespace autotune {

const char attn_fwd_packed_string[] =
"PERSISTENT_TYPE=0;GRID_CU_MULTIP=2;BLOCK_M=16;BLOCK_N=16;PRE_LOAD_V=False;NUM_XCDS=8\0"
"waves_per_eu=2;num_warps=4;num_stages=1\0"
"PERSISTENT_TYPE=2;GRID_CU_MULTIP=2;BLOCK_M=16;BLOCK_N=16;PRE_LOAD_V=False;NUM_XCDS=8\0";

int attn_fwd__lut_lambda__0 (const OpAttnFwdParams& params, int mod_number, int8_t lut[1][1]) {
    
    return lut[mod_number][0];
};

} // namespace autotune

AttnFwdContext::AutoTuneTableEntry
AttnFwdContext::autotune_table[][ 576 ] = {
    {
        &autotune::Autotune_attn_fwd__A0__F0,
        &autotune::Autotune_attn_fwd__A0__F1,
        &autotune::Autotune_attn_fwd__A0__F2,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F4,
        &autotune::Autotune_attn_fwd__A0__F5,
        &autotune::Autotune_attn_fwd__A0__F6,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F8,
        &autotune::Autotune_attn_fwd__A0__F9,
        &autotune::Autotune_attn_fwd__A0__F10,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F12,
        &autotune::Autotune_attn_fwd__A0__F13,
        &autotune::Autotune_attn_fwd__A0__F14,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F16,
        &autotune::Autotune_attn_fwd__A0__F17,
        &autotune::Autotune_attn_fwd__A0__F18,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F20,
        &autotune::Autotune_attn_fwd__A0__F21,
        &autotune::Autotune_attn_fwd__A0__F22,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F24,
        &autotune::Autotune_attn_fwd__A0__F25,
        &autotune::Autotune_attn_fwd__A0__F26,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F28,
        &autotune::Autotune_attn_fwd__A0__F29,
        &autotune::Autotune_attn_fwd__A0__F30,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F32,
        &autotune::Autotune_attn_fwd__A0__F33,
        &autotune::Autotune_attn_fwd__A0__F34,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F36,
        &autotune::Autotune_attn_fwd__A0__F37,
        &autotune::Autotune_attn_fwd__A0__F38,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F40,
        &autotune::Autotune_attn_fwd__A0__F41,
        &autotune::Autotune_attn_fwd__A0__F42,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F44,
        &autotune::Autotune_attn_fwd__A0__F45,
        &autotune::Autotune_attn_fwd__A0__F46,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F48,
        &autotune::Autotune_attn_fwd__A0__F49,
        &autotune::Autotune_attn_fwd__A0__F50,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F52,
        &autotune::Autotune_attn_fwd__A0__F53,
        &autotune::Autotune_attn_fwd__A0__F54,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F56,
        &autotune::Autotune_attn_fwd__A0__F57,
        &autotune::Autotune_attn_fwd__A0__F58,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F60,
        &autotune::Autotune_attn_fwd__A0__F61,
        &autotune::Autotune_attn_fwd__A0__F62,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F64,
        &autotune::Autotune_attn_fwd__A0__F65,
        &autotune::Autotune_attn_fwd__A0__F66,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F68,
        &autotune::Autotune_attn_fwd__A0__F69,
        &autotune::Autotune_attn_fwd__A0__F70,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F72,
        &autotune::Autotune_attn_fwd__A0__F73,
        &autotune::Autotune_attn_fwd__A0__F74,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F76,
        &autotune::Autotune_attn_fwd__A0__F77,
        &autotune::Autotune_attn_fwd__A0__F78,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F80,
        &autotune::Autotune_attn_fwd__A0__F81,
        &autotune::Autotune_attn_fwd__A0__F82,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F84,
        &autotune::Autotune_attn_fwd__A0__F85,
        &autotune::Autotune_attn_fwd__A0__F86,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F88,
        &autotune::Autotune_attn_fwd__A0__F89,
        &autotune::Autotune_attn_fwd__A0__F90,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F92,
        &autotune::Autotune_attn_fwd__A0__F93,
        &autotune::Autotune_attn_fwd__A0__F94,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F96,
        &autotune::Autotune_attn_fwd__A0__F97,
        &autotune::Autotune_attn_fwd__A0__F98,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F100,
        &autotune::Autotune_attn_fwd__A0__F101,
        &autotune::Autotune_attn_fwd__A0__F102,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F104,
        &autotune::Autotune_attn_fwd__A0__F105,
        &autotune::Autotune_attn_fwd__A0__F106,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F108,
        &autotune::Autotune_attn_fwd__A0__F109,
        &autotune::Autotune_attn_fwd__A0__F110,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F112,
        &autotune::Autotune_attn_fwd__A0__F113,
        &autotune::Autotune_attn_fwd__A0__F114,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F116,
        &autotune::Autotune_attn_fwd__A0__F117,
        &autotune::Autotune_attn_fwd__A0__F118,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F120,
        &autotune::Autotune_attn_fwd__A0__F121,
        &autotune::Autotune_attn_fwd__A0__F122,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F124,
        &autotune::Autotune_attn_fwd__A0__F125,
        &autotune::Autotune_attn_fwd__A0__F126,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F128,
        &autotune::Autotune_attn_fwd__A0__F129,
        &autotune::Autotune_attn_fwd__A0__F130,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F132,
        &autotune::Autotune_attn_fwd__A0__F133,
        &autotune::Autotune_attn_fwd__A0__F134,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F136,
        &autotune::Autotune_attn_fwd__A0__F137,
        &autotune::Autotune_attn_fwd__A0__F138,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F140,
        &autotune::Autotune_attn_fwd__A0__F141,
        &autotune::Autotune_attn_fwd__A0__F142,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F144,
        &autotune::Autotune_attn_fwd__A0__F145,
        &autotune::Autotune_attn_fwd__A0__F146,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F148,
        &autotune::Autotune_attn_fwd__A0__F149,
        &autotune::Autotune_attn_fwd__A0__F150,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F152,
        &autotune::Autotune_attn_fwd__A0__F153,
        &autotune::Autotune_attn_fwd__A0__F154,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F156,
        &autotune::Autotune_attn_fwd__A0__F157,
        &autotune::Autotune_attn_fwd__A0__F158,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F160,
        &autotune::Autotune_attn_fwd__A0__F161,
        &autotune::Autotune_attn_fwd__A0__F162,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F164,
        &autotune::Autotune_attn_fwd__A0__F165,
        &autotune::Autotune_attn_fwd__A0__F166,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F168,
        &autotune::Autotune_attn_fwd__A0__F169,
        &autotune::Autotune_attn_fwd__A0__F170,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F172,
        &autotune::Autotune_attn_fwd__A0__F173,
        &autotune::Autotune_attn_fwd__A0__F174,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F176,
        &autotune::Autotune_attn_fwd__A0__F177,
        &autotune::Autotune_attn_fwd__A0__F178,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F180,
        &autotune::Autotune_attn_fwd__A0__F181,
        &autotune::Autotune_attn_fwd__A0__F182,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F184,
        &autotune::Autotune_attn_fwd__A0__F185,
        &autotune::Autotune_attn_fwd__A0__F186,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F188,
        &autotune::Autotune_attn_fwd__A0__F189,
        &autotune::Autotune_attn_fwd__A0__F190,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F192,
        &autotune::Autotune_attn_fwd__A0__F193,
        &autotune::Autotune_attn_fwd__A0__F194,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F196,
        &autotune::Autotune_attn_fwd__A0__F197,
        &autotune::Autotune_attn_fwd__A0__F198,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F200,
        &autotune::Autotune_attn_fwd__A0__F201,
        &autotune::Autotune_attn_fwd__A0__F202,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F204,
        &autotune::Autotune_attn_fwd__A0__F205,
        &autotune::Autotune_attn_fwd__A0__F206,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F208,
        &autotune::Autotune_attn_fwd__A0__F209,
        &autotune::Autotune_attn_fwd__A0__F210,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F212,
        &autotune::Autotune_attn_fwd__A0__F213,
        &autotune::Autotune_attn_fwd__A0__F214,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F216,
        &autotune::Autotune_attn_fwd__A0__F217,
        &autotune::Autotune_attn_fwd__A0__F218,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F220,
        &autotune::Autotune_attn_fwd__A0__F221,
        &autotune::Autotune_attn_fwd__A0__F222,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F224,
        &autotune::Autotune_attn_fwd__A0__F225,
        &autotune::Autotune_attn_fwd__A0__F226,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F228,
        &autotune::Autotune_attn_fwd__A0__F229,
        &autotune::Autotune_attn_fwd__A0__F230,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F232,
        &autotune::Autotune_attn_fwd__A0__F233,
        &autotune::Autotune_attn_fwd__A0__F234,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F236,
        &autotune::Autotune_attn_fwd__A0__F237,
        &autotune::Autotune_attn_fwd__A0__F238,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F240,
        &autotune::Autotune_attn_fwd__A0__F241,
        &autotune::Autotune_attn_fwd__A0__F242,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F244,
        &autotune::Autotune_attn_fwd__A0__F245,
        &autotune::Autotune_attn_fwd__A0__F246,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F248,
        &autotune::Autotune_attn_fwd__A0__F249,
        &autotune::Autotune_attn_fwd__A0__F250,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F252,
        &autotune::Autotune_attn_fwd__A0__F253,
        &autotune::Autotune_attn_fwd__A0__F254,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F256,
        &autotune::Autotune_attn_fwd__A0__F257,
        &autotune::Autotune_attn_fwd__A0__F258,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F260,
        &autotune::Autotune_attn_fwd__A0__F261,
        &autotune::Autotune_attn_fwd__A0__F262,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F264,
        &autotune::Autotune_attn_fwd__A0__F265,
        &autotune::Autotune_attn_fwd__A0__F266,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F268,
        &autotune::Autotune_attn_fwd__A0__F269,
        &autotune::Autotune_attn_fwd__A0__F270,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F272,
        &autotune::Autotune_attn_fwd__A0__F273,
        &autotune::Autotune_attn_fwd__A0__F274,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F276,
        &autotune::Autotune_attn_fwd__A0__F277,
        &autotune::Autotune_attn_fwd__A0__F278,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F280,
        &autotune::Autotune_attn_fwd__A0__F281,
        &autotune::Autotune_attn_fwd__A0__F282,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F284,
        &autotune::Autotune_attn_fwd__A0__F285,
        &autotune::Autotune_attn_fwd__A0__F286,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F288,
        &autotune::Autotune_attn_fwd__A0__F289,
        &autotune::Autotune_attn_fwd__A0__F290,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F292,
        &autotune::Autotune_attn_fwd__A0__F293,
        &autotune::Autotune_attn_fwd__A0__F294,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F296,
        &autotune::Autotune_attn_fwd__A0__F297,
        &autotune::Autotune_attn_fwd__A0__F298,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F300,
        &autotune::Autotune_attn_fwd__A0__F301,
        &autotune::Autotune_attn_fwd__A0__F302,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F304,
        &autotune::Autotune_attn_fwd__A0__F305,
        &autotune::Autotune_attn_fwd__A0__F306,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F308,
        &autotune::Autotune_attn_fwd__A0__F309,
        &autotune::Autotune_attn_fwd__A0__F310,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F312,
        &autotune::Autotune_attn_fwd__A0__F313,
        &autotune::Autotune_attn_fwd__A0__F314,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F316,
        &autotune::Autotune_attn_fwd__A0__F317,
        &autotune::Autotune_attn_fwd__A0__F318,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F320,
        &autotune::Autotune_attn_fwd__A0__F321,
        &autotune::Autotune_attn_fwd__A0__F322,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F324,
        &autotune::Autotune_attn_fwd__A0__F325,
        &autotune::Autotune_attn_fwd__A0__F326,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F328,
        &autotune::Autotune_attn_fwd__A0__F329,
        &autotune::Autotune_attn_fwd__A0__F330,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F332,
        &autotune::Autotune_attn_fwd__A0__F333,
        &autotune::Autotune_attn_fwd__A0__F334,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F336,
        &autotune::Autotune_attn_fwd__A0__F337,
        &autotune::Autotune_attn_fwd__A0__F338,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F340,
        &autotune::Autotune_attn_fwd__A0__F341,
        &autotune::Autotune_attn_fwd__A0__F342,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F344,
        &autotune::Autotune_attn_fwd__A0__F345,
        &autotune::Autotune_attn_fwd__A0__F346,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F348,
        &autotune::Autotune_attn_fwd__A0__F349,
        &autotune::Autotune_attn_fwd__A0__F350,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F352,
        &autotune::Autotune_attn_fwd__A0__F353,
        &autotune::Autotune_attn_fwd__A0__F354,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F356,
        &autotune::Autotune_attn_fwd__A0__F357,
        &autotune::Autotune_attn_fwd__A0__F358,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F360,
        &autotune::Autotune_attn_fwd__A0__F361,
        &autotune::Autotune_attn_fwd__A0__F362,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F364,
        &autotune::Autotune_attn_fwd__A0__F365,
        &autotune::Autotune_attn_fwd__A0__F366,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F368,
        &autotune::Autotune_attn_fwd__A0__F369,
        &autotune::Autotune_attn_fwd__A0__F370,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F372,
        &autotune::Autotune_attn_fwd__A0__F373,
        &autotune::Autotune_attn_fwd__A0__F374,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F376,
        &autotune::Autotune_attn_fwd__A0__F377,
        &autotune::Autotune_attn_fwd__A0__F378,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F380,
        &autotune::Autotune_attn_fwd__A0__F381,
        &autotune::Autotune_attn_fwd__A0__F382,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F384,
        &autotune::Autotune_attn_fwd__A0__F385,
        &autotune::Autotune_attn_fwd__A0__F386,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F388,
        &autotune::Autotune_attn_fwd__A0__F389,
        &autotune::Autotune_attn_fwd__A0__F390,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F392,
        &autotune::Autotune_attn_fwd__A0__F393,
        &autotune::Autotune_attn_fwd__A0__F394,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F396,
        &autotune::Autotune_attn_fwd__A0__F397,
        &autotune::Autotune_attn_fwd__A0__F398,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F400,
        &autotune::Autotune_attn_fwd__A0__F401,
        &autotune::Autotune_attn_fwd__A0__F402,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F404,
        &autotune::Autotune_attn_fwd__A0__F405,
        &autotune::Autotune_attn_fwd__A0__F406,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F408,
        &autotune::Autotune_attn_fwd__A0__F409,
        &autotune::Autotune_attn_fwd__A0__F410,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F412,
        &autotune::Autotune_attn_fwd__A0__F413,
        &autotune::Autotune_attn_fwd__A0__F414,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F416,
        &autotune::Autotune_attn_fwd__A0__F417,
        &autotune::Autotune_attn_fwd__A0__F418,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F420,
        &autotune::Autotune_attn_fwd__A0__F421,
        &autotune::Autotune_attn_fwd__A0__F422,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F424,
        &autotune::Autotune_attn_fwd__A0__F425,
        &autotune::Autotune_attn_fwd__A0__F426,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F428,
        &autotune::Autotune_attn_fwd__A0__F429,
        &autotune::Autotune_attn_fwd__A0__F430,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F432,
        &autotune::Autotune_attn_fwd__A0__F433,
        &autotune::Autotune_attn_fwd__A0__F434,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F436,
        &autotune::Autotune_attn_fwd__A0__F437,
        &autotune::Autotune_attn_fwd__A0__F438,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F440,
        &autotune::Autotune_attn_fwd__A0__F441,
        &autotune::Autotune_attn_fwd__A0__F442,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F444,
        &autotune::Autotune_attn_fwd__A0__F445,
        &autotune::Autotune_attn_fwd__A0__F446,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F448,
        &autotune::Autotune_attn_fwd__A0__F449,
        &autotune::Autotune_attn_fwd__A0__F450,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F452,
        &autotune::Autotune_attn_fwd__A0__F453,
        &autotune::Autotune_attn_fwd__A0__F454,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F456,
        &autotune::Autotune_attn_fwd__A0__F457,
        &autotune::Autotune_attn_fwd__A0__F458,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F460,
        &autotune::Autotune_attn_fwd__A0__F461,
        &autotune::Autotune_attn_fwd__A0__F462,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F464,
        &autotune::Autotune_attn_fwd__A0__F465,
        &autotune::Autotune_attn_fwd__A0__F466,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F468,
        &autotune::Autotune_attn_fwd__A0__F469,
        &autotune::Autotune_attn_fwd__A0__F470,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F472,
        &autotune::Autotune_attn_fwd__A0__F473,
        &autotune::Autotune_attn_fwd__A0__F474,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F476,
        &autotune::Autotune_attn_fwd__A0__F477,
        &autotune::Autotune_attn_fwd__A0__F478,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F480,
        &autotune::Autotune_attn_fwd__A0__F481,
        &autotune::Autotune_attn_fwd__A0__F482,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F484,
        &autotune::Autotune_attn_fwd__A0__F485,
        &autotune::Autotune_attn_fwd__A0__F486,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F488,
        &autotune::Autotune_attn_fwd__A0__F489,
        &autotune::Autotune_attn_fwd__A0__F490,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F492,
        &autotune::Autotune_attn_fwd__A0__F493,
        &autotune::Autotune_attn_fwd__A0__F494,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F496,
        &autotune::Autotune_attn_fwd__A0__F497,
        &autotune::Autotune_attn_fwd__A0__F498,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F500,
        &autotune::Autotune_attn_fwd__A0__F501,
        &autotune::Autotune_attn_fwd__A0__F502,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F504,
        &autotune::Autotune_attn_fwd__A0__F505,
        &autotune::Autotune_attn_fwd__A0__F506,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F508,
        &autotune::Autotune_attn_fwd__A0__F509,
        &autotune::Autotune_attn_fwd__A0__F510,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F512,
        &autotune::Autotune_attn_fwd__A0__F513,
        &autotune::Autotune_attn_fwd__A0__F514,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F516,
        &autotune::Autotune_attn_fwd__A0__F517,
        &autotune::Autotune_attn_fwd__A0__F518,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F520,
        &autotune::Autotune_attn_fwd__A0__F521,
        &autotune::Autotune_attn_fwd__A0__F522,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F524,
        &autotune::Autotune_attn_fwd__A0__F525,
        &autotune::Autotune_attn_fwd__A0__F526,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F528,
        &autotune::Autotune_attn_fwd__A0__F529,
        &autotune::Autotune_attn_fwd__A0__F530,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F532,
        &autotune::Autotune_attn_fwd__A0__F533,
        &autotune::Autotune_attn_fwd__A0__F534,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F536,
        &autotune::Autotune_attn_fwd__A0__F537,
        &autotune::Autotune_attn_fwd__A0__F538,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F540,
        &autotune::Autotune_attn_fwd__A0__F541,
        &autotune::Autotune_attn_fwd__A0__F542,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F544,
        &autotune::Autotune_attn_fwd__A0__F545,
        &autotune::Autotune_attn_fwd__A0__F546,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F548,
        &autotune::Autotune_attn_fwd__A0__F549,
        &autotune::Autotune_attn_fwd__A0__F550,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F552,
        &autotune::Autotune_attn_fwd__A0__F553,
        &autotune::Autotune_attn_fwd__A0__F554,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F556,
        &autotune::Autotune_attn_fwd__A0__F557,
        &autotune::Autotune_attn_fwd__A0__F558,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F560,
        &autotune::Autotune_attn_fwd__A0__F561,
        &autotune::Autotune_attn_fwd__A0__F562,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F564,
        &autotune::Autotune_attn_fwd__A0__F565,
        &autotune::Autotune_attn_fwd__A0__F566,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F568,
        &autotune::Autotune_attn_fwd__A0__F569,
        &autotune::Autotune_attn_fwd__A0__F570,
        nullptr,
        &autotune::Autotune_attn_fwd__A0__F572,
        &autotune::Autotune_attn_fwd__A0__F573,
        &autotune::Autotune_attn_fwd__A0__F574,
        nullptr,
    },
};

}

// vim: set fileencoding=utf-8

