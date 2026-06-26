// Copyright © 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// clang-format off
#include "iface.op_attn_fwd.h"
#include <aotriton/util.h>
#include <tuple>
#include <iostream>
#include "affine.aiter_fmha_v3_fwd.h"
#include "shim.attn_fwd.h"
#include "shim.debug_simulate_encoded_softmax.h"

namespace AOTRITON_NS::v3::flash {

int64_t OpAttnFwdContext::godel_number() const
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
OpAttnFwdContext::lookup_optimal(Gpu gpu) {
    auto [arch_number, mod_number] = get_archmod_number(gpu);
    if (arch_number < 0) {
        return hipErrorNoBinaryForGpu;
    }
    backend_index = BackendEnum::None;
    auto number = godel_number();
    if (number < 0)
        return hipErrorNotSupported;
    auto tune_func = optune_table[arch_number][number];
    if (!tune_func)
        return hipErrorProfilerNotInitialized;
    tune_func(*this, mod_number);
    // In case tuning database is broken
    if (backend_index < 0)
        backend_index = fallback_backend;
    return hipSuccess;
}

std::tuple<int, int>
OpAttnFwdContext::get_archmod_number(Gpu gpu) {
    if (gpu == GPU_AMD_ARCH_GFX942_MOD0) return { 0, 0 };
    // TODO: print warning about tuning for this GPU mod is not built.
    // Note: if some mod does not have tuning info in the database at all, the
    //       getGpuFromStream should not return that mod from beginning.
    return std::make_tuple(-1, 0);
}

hipError_t
OpAttnFwdContext::launch(Gpu gpu, hipStream_t stream) const {
    if (backend_index < 0) {
        return hipErrorPriorLaunchFailure;
    }
    auto ret = launcher_table[backend_index](*this, gpu, stream);
    // It is possible that the optimal backend does not support certain inputs
    // In this case hipErrorPeerAccessUnsupported will be returned
    if (ret == hipErrorPeerAccessUnsupported) {
        if (!disable_fallback) {
#ifndef NDEBUG
          std::cerr << "OpAttnFwdContext::launch failed with optimal backend, "
                     << "calling fallback." << std::endl;
#endif
          return launcher_table[fallback_backend](*this, gpu, stream);
        }
#ifndef NDEBUG
        std::cerr << "OpAttnFwdContext::launch failed with optimal backend, "
                   << "fallback disabled, returning error." << std::endl;
#endif
    }
    return ret;
}

// Launchers are defined in op source file and no need to put them in to
// optune namespace
namespace {
hipError_t launcher_for_kMetro_Triton(const OpAttnFwdContext& context,
                                  Gpu gpu,
                                  hipStream_t stream) {
    hipError_t err;
  AttnFwdContext bcontext0(context, true);
  err = bcontext0.lookup_optimal(gpu);
  if (err != hipSuccess)
    return err;

  DebugSimulateEncodedSoftmaxContext bcontext1(context, context.params->encoded_softmax ->data_ptr() != nullptr);
  err = bcontext1.lookup_optimal(gpu);
  if (err != hipSuccess)
    return err;

  err = bcontext0.launch(stream);
  if (err != hipSuccess)
    return err;

  err = bcontext1.launch(stream);
  if (err != hipSuccess)
    return err;

return hipSuccess;
}


hipError_t launcher_for_kSlimAffine_AiterFmhaV3Fwd(const OpAttnFwdContext& context,
                                  Gpu gpu,
                                  hipStream_t stream) {
    AiterFmhaV3FwdContext bcontext(context, true);
    hipError_t err;
    err = bcontext.lookup_optimal(gpu);
    if (err != hipSuccess)
        return err;
    err = bcontext.launch(stream);
    return err;
}

}

OpAttnFwdContext::BackendLauncher
OpAttnFwdContext::launcher_table[ BackendEnum::Max ] = {
    &launcher_for_kMetro_Triton,
    &launcher_for_kSlimAffine_AiterFmhaV3Fwd
};

namespace optune {

int op_attn_fwd__lut_lambda__0 (const OpAttnFwdParams& params, int mod_number, int8_t lut[1][10][10]) {
    auto Max_seqlen_q_binned_index = [] (int x) {
        if (x <= 16) return 0;
        if (x <= 32) return 1;
        if (x <= 64) return 2;
        if (x <= 128) return 3;
        if (x <= 256) return 4;
        if (x <= 512) return 5;
        if (x <= 1024) return 6;
        if (x <= 2048) return 7;
        if (x <= 4096) return 8;
        if (x <= 8192) return 9;
        return 9;
    }(params.Max_seqlen_q);
    auto Max_seqlen_k_binned_index = [] (int x) {
        if (x <= 16) return 0;
        if (x <= 32) return 1;
        if (x <= 64) return 2;
        if (x <= 128) return 3;
        if (x <= 256) return 4;
        if (x <= 512) return 5;
        if (x <= 1024) return 6;
        if (x <= 2048) return 7;
        if (x <= 4096) return 8;
        if (x <= 8192) return 9;
        return 9;
    }(params.Max_seqlen_k);
    return lut[mod_number][Max_seqlen_q_binned_index][Max_seqlen_k_binned_index];
};

} // namespace autotune

// When Functional's LUT is uniform or empty
namespace {
void optune_op_attn_fwd__Trivial_kMetro_Triton(OpAttnFwdContext& context, int) {
    context.backend_index = OpAttnFwdContext::BackendEnum::kMetro_Triton;
}

}

OpAttnFwdContext::OpTuneTableEntry
OpAttnFwdContext::optune_table[][ 576 ] = {
    {
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune::Optune_op_attn_fwd__A0__F288,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune::Optune_op_attn_fwd__A0__F290,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune::Optune_op_attn_fwd__A0__F296,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune::Optune_op_attn_fwd__A0__F298,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
        &optune_op_attn_fwd__Trivial_kMetro_Triton,
    },
};

}

// vim: set fileencoding=utf-8

