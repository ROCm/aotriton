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
// There is no common operator for attn_fwd.
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
#endif

struct AttnFwdContext {
    const OpAttnFwdParams *params = nullptr;
#if 1
    const attn_options *call_options = nullptr;
#endif
    template <typename ParentContext>
    AttnFwdContext(const ParentContext& pcontext, bool condition)
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
    int8_t  PERSISTENT_TYPE;
    int8_t  GRID_CU_MULTIP;
    int8_t  NUM_XCDS;
    bool    PRE_LOAD_V;

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
    std::function<dim3(const AttnFwdContext&)> custom_grid_calculator;

    int64_t godel_number() const;
    static std::tuple<int, int> get_archmod_number(Gpu gpu);
    static constexpr int kMaxGodelNumber = 576;

    typedef void (*AutoTuneTableEntry)(AttnFwdContext& context, int mod_number);
    static AutoTuneTableEntry autotune_table[][ kMaxGodelNumber ];
};

struct AttnFwdMetadata {
    // Note: FEAT_CHOICES here
    static const std::vector<std::string>& get_Q_choices();
    static const std::vector<std::string>& get_Sm_scale_choices();
    static const std::vector<std::string>& get_L_choices();
    static const std::vector<int>& get_Q_descale_choices();
    static const std::vector<std::string>& get_Num_head_q_choices();
    static const std::vector<std::string>& get_cu_seqlens_q_choices();
    static const std::vector<int>& get_BLOCK_DMODEL_choices();
    static const std::vector<bool>& get_PADDED_HEAD_choices();
    static const std::vector<bool>& get_ENABLE_DROPOUT_choices();
    static const std::vector<bool>& get_RETURN_ENCODED_SOFTMAX_choices();
    static const std::vector<int>& get_CAUSAL_TYPE_choices();
    static const std::vector<int>& get_BIAS_TYPE_choices();
    static const std::vector<bool>& get_USE_ALIBI_choices();
    static const std::vector<bool>& get_INT8_choices();
    static const std::vector<std::string>& get_Num_CU_choices();
};

namespace autotune {

extern const char attn_fwd_packed_string[];

extern int attn_fwd__lut_lambda__0(const OpAttnFwdParams& params, int mod_number, int8_t lut[1][1]);

void Autotune_attn_fwd__A0__F0(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F1(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F2(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F4(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F5(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F6(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F8(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F9(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F10(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F12(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F13(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F14(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F16(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F17(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F18(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F20(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F21(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F22(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F24(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F25(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F26(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F28(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F29(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F30(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F32(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F33(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F34(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F36(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F37(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F38(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F40(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F41(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F42(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F44(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F45(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F46(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F48(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F49(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F50(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F52(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F53(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F54(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F56(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F57(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F58(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F60(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F61(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F62(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F64(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F65(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F66(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F68(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F69(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F70(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F72(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F73(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F74(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F76(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F77(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F78(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F80(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F81(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F82(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F84(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F85(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F86(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F88(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F89(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F90(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F92(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F93(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F94(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F96(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F97(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F98(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F100(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F101(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F102(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F104(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F105(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F106(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F108(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F109(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F110(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F112(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F113(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F114(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F116(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F117(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F118(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F120(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F121(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F122(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F124(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F125(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F126(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F128(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F129(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F130(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F132(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F133(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F134(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F136(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F137(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F138(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F140(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F141(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F142(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F144(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F145(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F146(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F148(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F149(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F150(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F152(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F153(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F154(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F156(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F157(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F158(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F160(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F161(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F162(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F164(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F165(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F166(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F168(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F169(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F170(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F172(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F173(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F174(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F176(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F177(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F178(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F180(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F181(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F182(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F184(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F185(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F186(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F188(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F189(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F190(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F192(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F193(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F194(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F196(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F197(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F198(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F200(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F201(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F202(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F204(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F205(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F206(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F208(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F209(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F210(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F212(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F213(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F214(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F216(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F217(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F218(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F220(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F221(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F222(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F224(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F225(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F226(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F228(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F229(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F230(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F232(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F233(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F234(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F236(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F237(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F238(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F240(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F241(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F242(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F244(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F245(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F246(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F248(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F249(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F250(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F252(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F253(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F254(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F256(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F257(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F258(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F260(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F261(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F262(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F264(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F265(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F266(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F268(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F269(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F270(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F272(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F273(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F274(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F276(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F277(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F278(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F280(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F281(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F282(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F284(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F285(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F286(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F288(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F289(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F290(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F292(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F293(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F294(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F296(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F297(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F298(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F300(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F301(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F302(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F304(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F305(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F306(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F308(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F309(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F310(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F312(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F313(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F314(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F316(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F317(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F318(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F320(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F321(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F322(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F324(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F325(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F326(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F328(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F329(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F330(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F332(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F333(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F334(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F336(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F337(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F338(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F340(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F341(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F342(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F344(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F345(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F346(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F348(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F349(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F350(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F352(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F353(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F354(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F356(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F357(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F358(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F360(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F361(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F362(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F364(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F365(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F366(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F368(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F369(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F370(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F372(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F373(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F374(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F376(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F377(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F378(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F380(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F381(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F382(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F384(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F385(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F386(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F388(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F389(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F390(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F392(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F393(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F394(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F396(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F397(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F398(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F400(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F401(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F402(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F404(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F405(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F406(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F408(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F409(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F410(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F412(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F413(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F414(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F416(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F417(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F418(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F420(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F421(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F422(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F424(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F425(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F426(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F428(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F429(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F430(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F432(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F433(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F434(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F436(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F437(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F438(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F440(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F441(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F442(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F444(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F445(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F446(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F448(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F449(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F450(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F452(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F453(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F454(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F456(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F457(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F458(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F460(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F461(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F462(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F464(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F465(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F466(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F468(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F469(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F470(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F472(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F473(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F474(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F476(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F477(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F478(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F480(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F481(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F482(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F484(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F485(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F486(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F488(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F489(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F490(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F492(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F493(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F494(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F496(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F497(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F498(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F500(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F501(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F502(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F504(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F505(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F506(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F508(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F509(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F510(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F512(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F513(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F514(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F516(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F517(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F518(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F520(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F521(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F522(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F524(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F525(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F526(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F528(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F529(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F530(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F532(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F533(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F534(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F536(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F537(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F538(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F540(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F541(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F542(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F544(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F545(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F546(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F548(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F549(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F550(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F552(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F553(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F554(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F556(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F557(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F558(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F560(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F561(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F562(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F564(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F565(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F566(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F568(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F569(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F570(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F572(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F573(AttnFwdContext& params, int mod_number);
void Autotune_attn_fwd__A0__F574(AttnFwdContext& params, int mod_number);

}


}

// vim: set fileencoding=utf-8

