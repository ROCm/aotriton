// Copyright © 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// clang-format off
#pragma once

#include <aotriton/config.h>
#include <aotriton/dtypes.h>
#include <aotriton/util.h>
#include <aotriton/runtime.h>
#include <aotriton/_internal/lazy_tensor_internal.h>
#include <aotriton/flash.h>
#include <functional>
#include <string>
#include <vector>

namespace AOTRITON_NS::v3::flash {

// Unlike KernelDescription, Operator must have its own parameter class
struct OpAttnFwdParams {
    const TensorView<4>* Q;
    const TensorView<4>* K;
    const TensorView<4>* V;
    const TensorView<4>* B;
    const TensorView<2>* A;
    float                Sm_scale;
    const TensorView<2>* L;
    const TensorView<4>* Out;
    int8_t               Q_descale;
    int8_t               K_descale;
    int8_t               P_scale;
    int8_t               P_descale;
    int8_t               V_descale;
    int32_t              Num_head_q;
    int32_t              Num_head_k;
    int32_t              Num_seqlens;
    const TensorView<1>* cu_seqlens_q;
    const TensorView<1>* cu_seqlens_k;
    int32_t              Max_seqlen_q;
    int32_t              Max_seqlen_k;
    const TensorView<1>* seq_strides_q;
    const TensorView<1>* seq_strides_k;
    int16_t              BLOCK_DMODEL;
    int32_t              Hdim_qk;
    int32_t              Hdim_vo;
    bool                 PADDED_HEAD;
    bool                 ENABLE_DROPOUT;
    float                dropout_p;
    const TensorView<0>* philox_seed_ptr;
    const TensorView<0>* philox_offset1;
    uint64_t             philox_offset2;
    const TensorView<0>* philox_seed_output;
    const TensorView<0>* philox_offset_output;
    bool                 RETURN_ENCODED_SOFTMAX;
    const TensorView<4>* encoded_softmax;
    int8_t               CAUSAL_TYPE;
    int32_t              Window_left;
    int32_t              Window_right;
    int8_t               BIAS_TYPE;
    bool                 USE_ALIBI;
    bool                 INT8;
    bool                 INT8_KV;
    bool                 USE_P_SCALE;
    const TensorView<0>* persistent_atomic_counter;
    int32_t              Num_CU;
    int32_t              Batch;
};

struct OpAttnFwdContext {
    OpAttnFwdParams *params = nullptr;
    const attn_options *call_options = nullptr;
    enum BackendEnum : int32_t {
        None = -1,
        kMetro_Triton = 0,
        kSlimAffine_AiterFmhaV3Fwd = 1,
        Max = 2
    };
    static constexpr BackendEnum fallback_backend = kMetro_Triton;
    BackendEnum backend_index = BackendEnum::None;
    bool disable_fallback = false;

#if AOTRITON_BUILD_FOR_TUNING
    int _has_preferred_backend = -1;
    static constexpr int _total_number_of_backends = BackendEnum::Max;
    const char* _backend_name = nullptr;
#endif

    // One more layer of dispatcher of functionals is added due to
    // 1. Individual kernel may use fewer arguments
    // 2. Metro kernel needs overall performance numbers over individual kernels.
    // 3. Even metro kernel only has one kernel, another set LUT is need to
    //    determine which metro kernel (or backend) need to be used
    int64_t godel_number() const;
    // get_archmod_number must be implemented in per-kernel/op basis
    // because different kernel/op may have different sets of GPU supported,
    // e.g., vector_add can be supported by all GPUs but SDPA can only be
    // supported on GPUs with WGMMA/MFMA/WMMA
    static std::tuple<int, int> get_archmod_number(Gpu gpu);
    static constexpr int kMaxGodelNumber = 576;

    hipError_t lookup_optimal(Gpu gpu);
    // Unlike Triton kernel, Operator's launch need gpu argument to eventually
    // call backend's lookup_optimal
    hipError_t launch(Gpu gpu, hipStream_t stream) const;
private:
    typedef void (*OpTuneTableEntry)(OpAttnFwdContext& context, int mod_number);
    static OpTuneTableEntry optune_table[][ kMaxGodelNumber ];

    typedef hipError_t (*BackendLauncher)(const OpAttnFwdContext& context,
                                          Gpu gpu,
                                          hipStream_t stream);
    static BackendLauncher launcher_table[ BackendEnum::Max ];
};

namespace optune {

extern int op_attn_fwd__lut_lambda__0(const OpAttnFwdParams& params, int mod_number, int8_t lut[1][10][10]);

void Optune_op_attn_fwd__A0__F288(OpAttnFwdContext& params, int mod_number);
void Optune_op_attn_fwd__A0__F290(OpAttnFwdContext& params, int mod_number);
void Optune_op_attn_fwd__A0__F296(OpAttnFwdContext& params, int mod_number);
void Optune_op_attn_fwd__A0__F298(OpAttnFwdContext& params, int mod_number);

}

}

// vim: set fileencoding=utf-8

