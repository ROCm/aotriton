// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// #include "aiter_hip_common.h"
// #include <hip/hip_runtime.h>
// #include <hip/hip_fp16.h>
// #include "mha_bwd.h"

#include <iostream>
#include <aotriton/_internal/flash/aiter.h>

namespace AOTRITON_NS::v3::flash::aiter {

template <ck_tile::index_t HDim_,
          typename DataType_,
          int mask_type_,
          bool kIsAtomic32_,
          ck_tile::index_t BF16Cvt_,
          bool kIsSEQPad_,
          bool kIsHDPad_,
          bool kIsGroupMode_ = false>
struct fmha_bwd_dq_dk_dv_v3_traits_
{
    static constexpr ck_tile::index_t HDim    = HDim_;
    using DataType                            = ck_tile::remove_cvref_t<DataType_>;
    static constexpr int mask_type            = mask_type_;
    static constexpr bool kIsAtomic32         = kIsAtomic32_;
    static constexpr ck_tile::index_t BF16Cvt = BF16Cvt_;
    static constexpr bool kIsSEQPad           = kIsSEQPad_;
    static constexpr bool kIsHDPad            = kIsHDPad_;
    static constexpr bool kIsGroupMode        = kIsGroupMode_;
};

template <typename fmha_bwd_dq_dk_dv_v3_traits_> struct FmhaBwdV3Name;
// ########################################################|HDim|    DataType| MaskType|kIsAtomic32|BF16Cvt|kIsSEQPad|kIsHDPad|
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      0,    false,   false>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_a16_rtne"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      1,    false,   false>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_a16_rtna"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      2,    false,   false>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_a16_rtz"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      0,    false,   false>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_a32_rtne"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      1,    false,   false>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_a32_rtna"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      2,    false,   false>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_a32_rtz"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,      false,      0,    false,   false>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_causal_a16_rtne"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,      false,      1,    false,   false>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_causal_a16_rtna"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,      false,      2,    false,   false>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_causal_a16_rtz"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      0,    false,   false>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_causal_a32_rtne"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      1,    false,   false>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_causal_a32_rtna"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      2,    false,   false>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_causal_a32_rtz"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        0,      false,      0,    false,   false>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_fp16_a16"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        0,       true,      0,    false,   false>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_fp16_a32"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        1,      false,      0,    false,   false>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_fp16_causal_a16"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        1,       true,      0,    false,   false>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_fp16_causal_a32"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      0,    false,    true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_a16_rtne_pddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      1,    false,    true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_a16_rtna_pddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      2,    false,    true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_a16_rtz_pddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      0,     true,    true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_a32_rtne_psskddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      1,     true,    true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_a32_rtna_psskddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      2,     true,    true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_a32_rtz_psskddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,      false,      0,    false,    true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_causal_a16_rtne_pddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,      false,      1,    false,    true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_causal_a16_rtna_pddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,      false,      2,    false,    true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_causal_a16_rtz_pddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      0,     true,    true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_causal_a32_rtne_psskddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      1,     true,    true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_causal_a32_rtna_psskddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      2,     true,    true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_causal_a32_rtz_psskddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        0,      false,      0,    false,    true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_fp16_a16_pddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        0,       true,      0,     true,    true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_fp16_a32_psskddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        1,      false,      0,    false,    true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_fp16_causal_a16_pddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        1,       true,      0,     true,    true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_fp16_causal_a32_psskddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,      false,      0,    false,   false>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd64_bf16_a16_rtne"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,      false,      1,    false,   false>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd64_bf16_a16_rtna"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,      false,      2,    false,   false>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd64_bf16_a16_rtz"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      0,     true,   false>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd64_bf16_a32_rtne_pssk"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      1,     true,   false>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd64_bf16_a32_rtna_pssk"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      2,     true,   false>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd64_bf16_a32_rtz_pssk"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,      false,      0,    false,   false>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd64_bf16_causal_a16_rtne"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,      false,      1,    false,   false>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd64_bf16_causal_a16_rtna"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,      false,      2,    false,   false>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd64_bf16_causal_a16_rtz"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      0,     true,   false>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd64_bf16_causal_a32_rtne_pssk"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      1,     true,   false>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd64_bf16_causal_a32_rtna_pssk"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      2,     true,   false>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd64_bf16_causal_a32_rtz_pssk"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        0,      false,      0,    false,   false>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd64_fp16_a16"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        0,       true,      0,     true,   false>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd64_fp16_a32_pssk"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        1,      false,      0,    false,   false>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd64_fp16_causal_a16"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        1,       true,      0,     true,   false>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd64_fp16_causal_a32_pssk"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        0,       true,      0,     true,    true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd192_bf16_a32_rtne_psskddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        0,       true,      1,     true,    true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd192_bf16_a32_rtna_psskddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        0,       true,      2,     true,    true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd192_bf16_a32_rtz_psskddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        1,       true,      0,     true,    true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd192_bf16_causal_a32_rtne_psskddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        1,       true,      1,     true,    true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd192_bf16_causal_a32_rtna_psskddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        1,       true,      2,     true,    true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd192_bf16_causal_a32_rtz_psskddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdFp16,        0,       true,      0,     true,    true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd192_fp16_a32_psskddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdFp16,        1,       true,      0,     true,    true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd192_fp16_causal_a32_psskddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        2,       true,      0,     true,    true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_fp16_swa_a32_psskddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        2,       true,      0,     true,    true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_swa_a32_rtne_psskddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        2,       true,      1,     true,    true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_swa_a32_rtna_psskddv"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        2,       true,      2,     true,    true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_swa_a32_rtz_psskddv"; };
// ########################################################|HDim|    DataType| MaskType|kIsAtomic32|BF16Cvt|kIsSEQPad|kIsHDPad|kIsGroupMode|
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      0,     true,   false,        true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd64_bf16_a32_rtne_pssk_group"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      1,     true,   false,        true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd64_bf16_a32_rtna_pssk_group"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      2,     true,   false,        true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd64_bf16_a32_rtz_pssk_group"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      0,     true,   false,        true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd64_bf16_causal_a32_rtne_pssk_group"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      1,     true,   false,        true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd64_bf16_causal_a32_rtna_pssk_group"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      2,     true,   false,        true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd64_bf16_causal_a32_rtz_pssk_group"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        0,       true,      0,     true,   false,        true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd64_fp16_a32_pssk_group"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        1,       true,      0,     true,   false,        true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd64_fp16_causal_a32_pssk_group"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        1,       true,      0,     true,    true,        true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_fp16_causal_a32_psskddv_group"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        1,       true,      0,     true,   false,        true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_fp16_causal_a32_pssk_group"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        0,       true,      0,     true,    true,        true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_fp16_a32_psskddv_group"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        0,       true,      0,     true,   false,        true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_fp16_a32_pssk_group"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      0,     true,    true,        true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_a32_rtne_psskddv_group"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      1,     true,    true,        true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_a32_rtna_psskddv_group"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      2,     true,    true,        true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_a32_rtz_psskddv_group"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      0,     true,    true,        true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_causal_a32_rtne_psskddv_group"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      1,     true,    true,        true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_causal_a32_rtna_psskddv_group"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      2,     true,    true,        true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_causal_a32_rtz_psskddv_group"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      0,     true,   false,        true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_a32_rtne_pssk_group"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      1,     true,   false,        true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_a32_rtna_pssk_group"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      2,     true,   false,        true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_a32_rtz_pssk_group"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      0,     true,   false,        true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_causal_a32_rtne_pssk_group"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      1,     true,   false,        true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_causal_a32_rtna_pssk_group"; };
template<> struct FmhaBwdV3Name<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      2,     true,   false,        true>> { static constexpr const char * bwd_v3_name = "fmha_bwd_hd128_bf16_causal_a32_rtz_pssk_group"; };

template <typename fmha_bwd_dq_dk_dv_v3_traits_> struct FmhaBwdV3Buf;
// #######################################################|HDim|    DataType| MaskType|kIsAtomic32|BF16Cvt|kIsSEQPad|kIsHDPad|
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      0,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_a16_rtne.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      1,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_a16_rtna.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      2,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_a16_rtz.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      0,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_a32_rtne.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      1,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_a32_rtna.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      2,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_a32_rtz.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,      false,      0,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_causal_a16_rtne.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,      false,      1,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_causal_a16_rtna.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,      false,      2,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_causal_a16_rtz.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      0,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_causal_a32_rtne.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      1,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_causal_a32_rtna.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      2,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_causal_a32_rtz.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        0,      false,      0,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_fp16_a16.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        0,       true,      0,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_fp16_a32.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        1,      false,      0,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_fp16_causal_a16.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        1,       true,      0,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_fp16_causal_a32.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      0,    false,    true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_a16_rtne_pddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      1,    false,    true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_a16_rtna_pddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      2,    false,    true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_a16_rtz_pddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      0,     true,    true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_a32_rtne_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      1,     true,    true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_a32_rtna_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      2,     true,    true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_a32_rtz_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,      false,      0,    false,    true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_causal_a16_rtne_pddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,      false,      1,    false,    true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_causal_a16_rtna_pddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,      false,      2,    false,    true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_causal_a16_rtz_pddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      0,     true,    true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_causal_a32_rtne_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      1,     true,    true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_causal_a32_rtna_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      2,     true,    true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_causal_a32_rtz_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        0,      false,      0,    false,    true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_fp16_a16_pddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        0,       true,      0,     true,    true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_fp16_a32_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        1,      false,      0,    false,    true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_fp16_causal_a16_pddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        1,       true,      0,     true,    true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_fp16_causal_a32_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,      false,      0,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_bf16_a16_rtne.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,      false,      1,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_bf16_a16_rtna.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,      false,      2,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_bf16_a16_rtz.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      0,     true,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_bf16_a32_rtne_pssk.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      1,     true,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_bf16_a32_rtna_pssk.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      2,     true,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_bf16_a32_rtz_pssk.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,      false,      0,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_bf16_causal_a16_rtne.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,      false,      1,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_bf16_causal_a16_rtna.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,      false,      2,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_bf16_causal_a16_rtz.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      0,     true,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_bf16_causal_a32_rtne_pssk.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      1,     true,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_bf16_causal_a32_rtna_pssk.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      2,     true,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_bf16_causal_a32_rtz_pssk.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        0,      false,      0,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_fp16_a16.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        0,       true,      0,     true,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_fp16_a32_pssk.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        1,      false,      0,    false,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_fp16_causal_a16.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        1,       true,      0,     true,   false>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_fp16_causal_a32_pssk.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        0,       true,      0,     true,    true>> { static constexpr const char * bwd_v3_buf = "bwd_hd192_bf16_a32_rtne_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        0,       true,      1,     true,    true>> { static constexpr const char * bwd_v3_buf = "bwd_hd192_bf16_a32_rtna_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        0,       true,      2,     true,    true>> { static constexpr const char * bwd_v3_buf = "bwd_hd192_bf16_a32_rtz_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        1,       true,      0,     true,    true>> { static constexpr const char * bwd_v3_buf = "bwd_hd192_bf16_causal_a32_rtne_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        1,       true,      1,     true,    true>> { static constexpr const char * bwd_v3_buf = "bwd_hd192_bf16_causal_a32_rtna_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        1,       true,      2,     true,    true>> { static constexpr const char * bwd_v3_buf = "bwd_hd192_bf16_causal_a32_rtz_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdFp16,        0,       true,      0,     true,    true>> { static constexpr const char * bwd_v3_buf = "bwd_hd192_fp16_a32_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdFp16,        1,       true,      0,     true,    true>> { static constexpr const char * bwd_v3_buf = "bwd_hd192_fp16_causal_a32_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        2,       true,      0,     true,    true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_fp16_swa_a32_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        2,       true,      0,     true,    true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_swa_a32_rtne_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        2,       true,      1,     true,    true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_swa_a32_rtna_psskddv.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        2,       true,      2,     true,    true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_swa_a32_rtz_psskddv.co"; };
// #######################################################|HDim|    DataType| MaskType|kIsAtomic32|BF16Cvt|kIsSEQPad|kIsHDPad|kIsGroupMode|
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      0,     true,   false,        true>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_bf16_a32_rtne_pssk_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      1,     true,   false,        true>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_bf16_a32_rtna_pssk_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      2,     true,   false,        true>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_bf16_a32_rtz_pssk_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      0,     true,   false,        true>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_bf16_causal_a32_rtne_pssk_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      1,     true,   false,        true>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_bf16_causal_a32_rtna_pssk_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      2,     true,   false,        true>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_bf16_causal_a32_rtz_pssk_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        0,       true,      0,     true,   false,        true>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_fp16_a32_pssk_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        1,       true,      0,     true,   false,        true>> { static constexpr const char * bwd_v3_buf = "bwd_hd64_fp16_causal_a32_pssk_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        1,       true,      0,     true,    true,        true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_fp16_causal_a32_psskddv_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        1,       true,      0,     true,   false,        true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_fp16_causal_a32_pssk_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        0,       true,      0,     true,    true,        true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_fp16_a32_psskddv_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        0,       true,      0,     true,   false,        true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_fp16_a32_pssk_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      0,     true,    true,        true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_a32_rtne_psskddv_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      1,     true,    true,        true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_a32_rtna_psskddv_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      2,     true,    true,        true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_a32_rtz_psskddv_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      0,     true,    true,        true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_causal_a32_rtne_psskddv_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      1,     true,    true,        true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_causal_a32_rtna_psskddv_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      2,     true,    true,        true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_causal_a32_rtz_psskddv_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      0,     true,   false,        true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_a32_rtne_pssk_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      1,     true,   false,        true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_a32_rtna_pssk_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      2,     true,   false,        true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_a32_rtz_pssk_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      0,     true,   false,        true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_causal_a32_rtne_pssk_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      1,     true,   false,        true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_causal_a32_rtna_pssk_group.co"; };
template<> struct FmhaBwdV3Buf<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      2,     true,   false,        true>> { static constexpr const char * bwd_v3_buf = "bwd_hd128_bf16_causal_a32_rtz_pssk_group.co"; };

template <typename fmha_bwd_dq_dk_dv_v3_traits_> struct FmhaBwdV3Ts;
// ######################################################|HDim|    DataType| MaskType|kIsAtomic32|BF16Cvt|kIsSEQPad|kIsHDPad|
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      0,    false,   false>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = {F_tile_size_kv}; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      1,    false,   false>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = {F_tile_size_kv}; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      2,    false,   false>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = {F_tile_size_kv}; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      0,    false,   false>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = {F_tile_size_kv}; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      1,    false,   false>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = {F_tile_size_kv}; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      2,    false,   false>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = {F_tile_size_kv}; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,      false,      0,    false,   false>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = {F_tile_size_kv}; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,      false,      1,    false,   false>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = {F_tile_size_kv}; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,      false,      2,    false,   false>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = {F_tile_size_kv}; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      0,    false,   false>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = {F_tile_size_kv}; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      1,    false,   false>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = {F_tile_size_kv}; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      2,    false,   false>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = {F_tile_size_kv}; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        0,      false,      0,    false,   false>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        0,       true,      0,    false,   false>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        1,      false,      0,    false,   false>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        1,       true,      0,    false,   false>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      0,    false,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      1,    false,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,      false,      2,    false,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      0,     true,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      1,     true,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      2,     true,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,      false,      0,    false,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,      false,      1,    false,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,      false,      2,    false,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      0,     true,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      1,     true,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      2,     true,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        0,      false,      0,    false,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        0,       true,      0,     true,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        1,      false,      0,    false,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        1,       true,      0,     true,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,      false,      0,    false,   false>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,      false,      1,    false,   false>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,      false,      2,    false,   false>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      0,     true,   false>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      1,     true,   false>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      2,     true,   false>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,      false,      0,    false,   false>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,      false,      1,    false,   false>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,      false,      2,    false,   false>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      0,     true,   false>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      1,     true,   false>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      2,     true,   false>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        0,      false,      0,    false,   false>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        0,       true,      0,     true,   false>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        1,      false,      0,    false,   false>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        1,       true,      0,     true,   false>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        0,       true,      0,     true,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 64; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        0,       true,      1,     true,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 64; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        0,       true,      2,     true,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 64; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        1,       true,      0,     true,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 64; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        1,       true,      1,     true,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 64; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16,        1,       true,      2,     true,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 64; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdFp16,        0,       true,      0,     true,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 64; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdFp16,        1,       true,      0,     true,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 64; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        2,       true,      0,     true,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        2,       true,      0,     true,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        2,       true,      1,     true,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        2,       true,      2,     true,    true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
// ######################################################|HDim|    DataType| MaskType|kIsAtomic32|BF16Cvt|kIsSEQPad|kIsHDPad|kIsGroupMode|
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      0,     true,   false,        true>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      1,     true,   false,        true>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        0,       true,      2,     true,   false,        true>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      0,     true,   false,        true>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      1,     true,   false,        true>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdBf16,        1,       true,      2,     true,   false,        true>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        0,       true,      0,     true,   false,        true>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_< 64, FmhaBwdFp16,        1,       true,      0,     true,   false,        true>> { static constexpr int ts_qo = 32; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        1,       true,      0,     true,    true,        true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        1,       true,      0,     true,   false,        true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        0,       true,      0,     true,    true,        true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16,        0,       true,      0,     true,   false,        true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      0,     true,    true,        true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      0,     true,   false,        true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      0,     true,    true,        true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      0,     true,   false,        true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      1,     true,    true,        true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      1,     true,   false,        true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      1,     true,    true,        true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      1,     true,   false,        true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      2,     true,    true,        true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        1,       true,      2,     true,   false,        true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      2,     true,    true,        true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };
template<> struct FmhaBwdV3Ts<fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16,        0,       true,      2,     true,   false,        true>> { static constexpr int ts_qo = 16; static constexpr int ts_kv = 192; };

class fmha_dq_shuffle_kernel
{
    public:
    fmha_dq_shuffle_kernel(const char *name, const char *hsaco)
    {
        int length = strlen(name);
        std::string kernel_func_name = "_ZN5aiter" + std::to_string(length) + name + "E";
        std::string AITER_ASM_DIR = "{F_AITER_ASM_DIR}";
        HIP_CALL(hipModuleLoad(&module, (AITER_ASM_DIR + "fmha_v3_bwd/" + hsaco).c_str()));
        HIP_CALL(hipModuleGetFunction(&kernel_func, module, kernel_func_name.c_str()));
    }

    void
    launch_kernel(fmha_bwd_v3_traits fmha_v3_traits, fmha_bwd_dq_shuffle_args args, const ck_tile::stream_config& s) const
    {
        size_t arg_size = sizeof(args);
        void* config[]  = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                           &args,
                           HIP_LAUNCH_PARAM_BUFFER_SIZE,
                           &arg_size,
                           HIP_LAUNCH_PARAM_END};

        int bdx = 256;
        int gdx = (fmha_v3_traits.s + fmha_v3_traits.ts_dq - 1) / fmha_v3_traits.ts_dq;
        int gdy = fmha_v3_traits.h;
        int gdz = fmha_v3_traits.b;

        HIP_CALL(hipModuleLaunchKernel(kernel_func,
                                       gdx,
                                       gdy,
                                       gdz,
                                       bdx,
                                       1,
                                       1,
                                       0,
                                       s.stream_id_,
                                       NULL,
                                       reinterpret_cast<void**>(&config)));
    }
    private:
    hipModule_t module;
    hipFunction_t kernel_func;
};

class fmha_bwd_v3_kernel
{
    public:
    fmha_bwd_v3_kernel(const char *name, const char *hsaco)
    {
        int length = strlen(name);
        std::string kernel_func_name = "_ZN5aiter" + std::to_string(length) + name + "E";
        std::string AITER_ASM_DIR = "{F_AITER_ASM_DIR}";
        HIP_CALL(hipModuleLoad(&module, (AITER_ASM_DIR + "fmha_v3_bwd/" + hsaco).c_str()));
        HIP_CALL(hipModuleGetFunction(&kernel_func, module, kernel_func_name.c_str()));
    }

    void
    launch_kernel(fmha_bwd_v3_traits fmha_v3_traits, fmha_bwd_v3_args args, const ck_tile::stream_config& s) const
    {
        size_t arg_size = sizeof(args);
        void* config[]  = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                           &args,
                           HIP_LAUNCH_PARAM_BUFFER_SIZE,
                           &arg_size,
                           HIP_LAUNCH_PARAM_END};

        int bdx = 256;
        int gdx = (fmha_v3_traits.s + fmha_v3_traits.ts_kv - 1) / fmha_v3_traits.ts_kv;
        int gdy = fmha_v3_traits.h;
        int gdz = fmha_v3_traits.b;
        if(fmha_v3_traits.mask > 0)
        {
            int num_tg = (fmha_v3_traits.s + fmha_v3_traits.ts_kv - 1) / fmha_v3_traits.ts_kv;
            gdx        = (num_tg % 2) ? (num_tg / 2 + 1) : (num_tg / 2);
        }
        HIP_CALL(hipModuleLaunchKernel(kernel_func,
                                       gdx,
                                       gdy,
                                       gdz,
                                       bdx,
                                       1,
                                       1,
                                       0,
                                       s.stream_id_,
                                       NULL,
                                       reinterpret_cast<void**>(&config)));
    }

    void
    launch_kernel(fmha_bwd_v3_traits fmha_v3_traits, fmha_bwd_v3_gen_args args, const ck_tile::stream_config& s) const
    {
        size_t arg_size = sizeof(args);
        void* config[]  = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                           &args,
                           HIP_LAUNCH_PARAM_BUFFER_SIZE,
                           &arg_size,
                           HIP_LAUNCH_PARAM_END};

        int bdx = 256;
        int gdx = (fmha_v3_traits.s + fmha_v3_traits.ts_kv - 1) / fmha_v3_traits.ts_kv;
        int gdy = fmha_v3_traits.h;
        int gdz = fmha_v3_traits.b;
        if(fmha_v3_traits.mask > 0)
        {
            int num_tg = (fmha_v3_traits.s + fmha_v3_traits.ts_kv - 1) / fmha_v3_traits.ts_kv;
            gdx        = (num_tg % 2) ? (num_tg / 2 + 1) : (num_tg / 2);
        }
        HIP_CALL(hipModuleLaunchKernel(kernel_func,
                                       gdx,
                                       gdy,
                                       gdz,
                                       bdx,
                                       1,
                                       1,
                                       0,
                                       s.stream_id_,
                                       NULL,
                                       reinterpret_cast<void**>(&config)));
    }

    void
    launch_kernel(fmha_bwd_v3_traits fmha_v3_traits, fmha_bwd_v3_genl_args args, const ck_tile::stream_config& s) const
    {
        size_t arg_size = sizeof(args);
        void* config[]  = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                           &args,
                           HIP_LAUNCH_PARAM_BUFFER_SIZE,
                           &arg_size,
                           HIP_LAUNCH_PARAM_END};

        int bdx = 256;
        int gdx = (fmha_v3_traits.s + fmha_v3_traits.ts_kv - 1) / fmha_v3_traits.ts_kv;
        int gdy = fmha_v3_traits.h;
        int gdz = fmha_v3_traits.b;
        if(fmha_v3_traits.mask > 0)
        {
            int num_tg = (fmha_v3_traits.s + fmha_v3_traits.ts_kv - 1) / fmha_v3_traits.ts_kv;
            gdx        = (num_tg % 2) ? (num_tg / 2 + 1) : (num_tg / 2);
        }
        HIP_CALL(hipModuleLaunchKernel(kernel_func,
                                       gdx,
                                       gdy,
                                       gdz,
                                       bdx,
                                       1,
                                       1,
                                       0,
                                       s.stream_id_,
                                       NULL,
                                       reinterpret_cast<void**>(&config)));
    }

    void
    launch_kernel(fmha_bwd_v3_traits fmha_v3_traits, fmha_bwd_v3_group_args args, const ck_tile::stream_config& s) const
    {
        size_t arg_size = sizeof(args);
        void* config[]  = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                           &args,
                           HIP_LAUNCH_PARAM_BUFFER_SIZE,
                           &arg_size,
                           HIP_LAUNCH_PARAM_END};

        int gdx = (fmha_v3_traits.s + fmha_v3_traits.ts_kv - 1) / fmha_v3_traits.ts_kv;
        if(fmha_v3_traits.mask > 0)
        {
            gdx = (gdx % 2) ? (gdx / 2 + 1) : (gdx / 2);
        }
        HIP_CALL(hipModuleLaunchKernel(kernel_func,
                                       gdx,
                                       fmha_v3_traits.h, /*gdy*/
                                       fmha_v3_traits.b, /*gdz*/
                                       256, /*bdx*/
                                       1, /*bdy*/
                                       1, /*bdz*/
                                       0,
                                       s.stream_id_,
                                       NULL,
                                       reinterpret_cast<void**>(&config)));
    }

    void
    launch_kernel(fmha_bwd_v3_traits fmha_v3_traits, fmha_bwd_v3_swa_genl_args args, const ck_tile::stream_config& s) const
    {
        size_t arg_size = sizeof(args);
        void* config[]  = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                           &args,
                           HIP_LAUNCH_PARAM_BUFFER_SIZE,
                           &arg_size,
                           HIP_LAUNCH_PARAM_END};

        int bdx = 256;
        int gdx = (fmha_v3_traits.s + fmha_v3_traits.ts_kv - 1) / fmha_v3_traits.ts_kv;
        int gdy = fmha_v3_traits.h;
        int gdz = fmha_v3_traits.b;

        HIP_CALL(hipModuleLaunchKernel(kernel_func,
                                       gdx,
                                       gdy,
                                       gdz,
                                       bdx,
                                       1,
                                       1,
                                       0,
                                       s.stream_id_,
                                       NULL,
                                       reinterpret_cast<void**>(&config)));
    }
    private:
    hipModule_t module;
    hipFunction_t kernel_func;
};

template <typename dot_do_o_trait_, typename dq_dk_dv_v3_traits_>
float fmha_bwd_v3_(const ck_tile::stream_config& s, fmha_bwd_args a)
{
    if(s.log_level_ > 0)
        std::cout << ", " << fmha_bwd_dot_do_o_get_name_<dot_do_o_trait_>() << ", " << FmhaBwdV3Name<dq_dk_dv_v3_traits_>::bwd_v3_name << std::flush;
    fmha_bwd_v3_args args;
    args.ptr_dq  = a.dq_ptr;
    args.ptr_dk  = a.dk_ptr;
    args.ptr_dv  = a.dv_ptr;
    args.ptr_q   = a.q_ptr;
    args.ptr_k   = a.k_ptr;
    args.ptr_v   = a.v_ptr;
    args.ptr_do  = a.do_ptr;
    args.ptr_lse = a.lse_ptr;
    args.ptr_d   = a.d_ptr;
    args.scalar  = a.scale;
    args.log2e   = ck_tile::log2e_v<float>;
    args.seq_len = a.seqlen_q;

    args.Ts   = FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv * a.stride_k * 2;
    args.Hs   = a.nhead_stride_q * 2;
    args.BAs  = a.batch_stride_q * 2;
    args.Seqs = a.stride_q * 2;

    args.ratio    = a.nhead_q / a.nhead_k;
    args.Hs_kv    = a.nhead_stride_k * 2;
    args.BAs_kv   = a.batch_stride_k * 2;
    args.Seqs_kv  = a.stride_k * 2;
    args.Seqs_dkv = a.stride_dk * 2;
    auto traits = fmha_bwd_v3_traits{a.batch,
                                      a.nhead_q,
                                      a.seqlen_q,
                                      a.hdim_q,
                                      a.mask_type,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_qo,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv};
    {F_dq_shuffle_kernel_define}

    static thread_local fmha_bwd_v3_kernel impl(FmhaBwdV3Name<dq_dk_dv_v3_traits_>::bwd_v3_name, FmhaBwdV3Buf<dq_dk_dv_v3_traits_>::bwd_v3_buf); // static here is for thread safety.
    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){ fmha_bwd_dot_do_o_oneshot_<dot_do_o_trait_>(s_, a); },
        [=](const ck_tile::stream_config& s_){ impl.launch_kernel(traits, args, s_); }{F_dq_shuffle_kernel_call}

    );
}

template <typename dot_do_o_trait_, typename dq_dk_dv_v3_traits_>
float fmha_bwd_v3_gen_(const ck_tile::stream_config& s, fmha_bwd_args a)
{
    if(s.log_level_ > 0)
        std::cout << ", " << fmha_bwd_dot_do_o_get_name_<dot_do_o_trait_>() << ", " << FmhaBwdV3Name<dq_dk_dv_v3_traits_>::bwd_v3_name << std::flush;
    fmha_bwd_v3_gen_args args;
    args.ptr_dq  = a.dq_ptr;
    args.ptr_dk  = a.dk_ptr;
    args.ptr_dv  = a.dv_ptr;
    args.ptr_q   = a.q_ptr;
    args.ptr_k   = a.k_ptr;
    args.ptr_v   = a.v_ptr;
    args.ptr_do  = a.do_ptr;
    args.ptr_lse = a.lse_ptr;
    args.ptr_d   = a.d_ptr;
    args.scalar  = a.scale;
    args.log2e   = ck_tile::log2e_v<float>;
    args.seq_len = a.seqlen_q;

    args.Ts   = FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv * a.stride_k * 2;
    args.Hs   = a.nhead_stride_q * 2;
    args.BAs  = a.batch_stride_q * 2;
    args.Seqs = a.stride_q * 2;

    args.ratio    = a.nhead_q / a.nhead_k;
    args.Hs_kv    = a.nhead_stride_k * 2;
    args.BAs_kv   = a.batch_stride_k * 2;
    args.Seqs_kv  = a.stride_k * 2;
    args.Seqs_dkv = a.stride_dk * 2;
    args.head_dim = a.hdim_q;
    auto traits = fmha_bwd_v3_traits{a.batch,
                                      a.nhead_q,
                                      a.seqlen_q,
                                      a.hdim_q,
                                      a.mask_type,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_qo,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv};
    static thread_local fmha_bwd_v3_kernel impl(FmhaBwdV3Name<dq_dk_dv_v3_traits_>::bwd_v3_name, FmhaBwdV3Buf<dq_dk_dv_v3_traits_>::bwd_v3_buf); // static here is for thread safety.
    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){ fmha_bwd_dot_do_o_oneshot_<dot_do_o_trait_>(s_, a); },
        [=](const ck_tile::stream_config& s_){ impl.launch_kernel(traits, args, s_); }
    );
}

template <typename dot_do_o_trait_, typename dq_dk_dv_v3_traits_, typename convert_dq_trait_>
float fmha_bwd_v3_(const ck_tile::stream_config& s, fmha_bwd_args a)
{
    if(s.log_level_ > 0)
        std::cout << ", " << fmha_bwd_dot_do_o_get_name_<dot_do_o_trait_>() << ", " << FmhaBwdV3Name<dq_dk_dv_v3_traits_>::bwd_v3_name << ", " << fmha_bwd_convert_dq_get_name_<convert_dq_trait_>() << std::flush;
    fmha_bwd_v3_args args;
    args.ptr_dq  = a.dq_acc_ptr;
    args.ptr_dk  = a.dk_ptr;
    args.ptr_dv  = a.dv_ptr;
    args.ptr_q   = a.q_ptr;
    args.ptr_k   = a.k_ptr;
    args.ptr_v   = a.v_ptr;
    args.ptr_do  = a.do_ptr;
    args.ptr_lse = a.lse_ptr;
    args.ptr_d   = a.d_ptr;
    args.scalar  = a.scale;
    args.log2e   = ck_tile::log2e_v<float>;
    args.seq_len = a.seqlen_q;

    args.Ts   = FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv * a.stride_k * 2;
    args.Hs   = a.nhead_stride_q * 2;
    args.BAs  = a.batch_stride_q * 2;
    args.Seqs = a.stride_q * 2;

    args.ratio    = a.nhead_q / a.nhead_k;
    args.Hs_kv    = a.nhead_stride_k * 2;
    args.BAs_kv   = a.batch_stride_k * 2;
    args.Seqs_kv  = a.stride_k * 2;
    args.Seqs_dkv = a.stride_dk * 2;
    auto traits = fmha_bwd_v3_traits{a.batch,
                                      a.nhead_q,
                                      a.seqlen_q,
                                      a.hdim_q,
                                      a.mask_type,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_qo,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv};
    static thread_local fmha_bwd_v3_kernel impl(FmhaBwdV3Name<dq_dk_dv_v3_traits_>::bwd_v3_name, FmhaBwdV3Buf<dq_dk_dv_v3_traits_>::bwd_v3_buf); // static here is for thread safety.
    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){ fmha_bwd_dot_do_o_oneshot_<dot_do_o_trait_>(s_, a); },
        [=](const ck_tile::stream_config& s_){ impl.launch_kernel(traits, args, s_); },
        [=](const ck_tile::stream_config& s_){ fmha_bwd_convert_dq_oneshot_<convert_dq_trait_>(s_, a); }
    );
}

template <typename dot_do_o_trait_, typename dq_dk_dv_v3_traits_, typename convert_dq_trait_>
float fmha_bwd_v3_gen_(const ck_tile::stream_config& s, fmha_bwd_args a)
{
    if(s.log_level_ > 0)
        std::cout << ", " << fmha_bwd_dot_do_o_get_name_<dot_do_o_trait_>() << ", " << FmhaBwdV3Name<dq_dk_dv_v3_traits_>::bwd_v3_name << ", " << fmha_bwd_convert_dq_get_name_<convert_dq_trait_>() << std::flush;
    fmha_bwd_v3_gen_args args;
    args.ptr_dq  = a.dq_acc_ptr;
    args.ptr_dk  = a.dk_ptr;
    args.ptr_dv  = a.dv_ptr;
    args.ptr_q   = a.q_ptr;
    args.ptr_k   = a.k_ptr;
    args.ptr_v   = a.v_ptr;
    args.ptr_do  = a.do_ptr;
    args.ptr_lse = a.lse_ptr;
    args.ptr_d   = a.d_ptr;
    args.scalar  = a.scale;
    args.log2e   = ck_tile::log2e_v<float>;
    args.seq_len = a.seqlen_q;

    args.Ts   = FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv * a.stride_k * 2;
    args.Hs   = a.nhead_stride_q * 2;
    args.BAs  = a.batch_stride_q * 2;
    args.Seqs = a.stride_q * 2;

    args.ratio    = a.nhead_q / a.nhead_k;
    args.Hs_kv    = a.nhead_stride_k * 2;
    args.BAs_kv   = a.batch_stride_k * 2;
    args.Seqs_kv  = a.stride_k * 2;
    args.Seqs_dkv = a.stride_dk * 2;
    args.head_dim = a.hdim_q;
    auto traits = fmha_bwd_v3_traits{a.batch,
                                      a.nhead_q,
                                      a.seqlen_q,
                                      a.hdim_q,
                                      a.mask_type,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_qo,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv};
    static thread_local fmha_bwd_v3_kernel impl(FmhaBwdV3Name<dq_dk_dv_v3_traits_>::bwd_v3_name, FmhaBwdV3Buf<dq_dk_dv_v3_traits_>::bwd_v3_buf); // static here is for thread safety.
    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){ fmha_bwd_dot_do_o_oneshot_<dot_do_o_trait_>(s_, a); },
        [=](const ck_tile::stream_config& s_){ impl.launch_kernel(traits, args, s_); },
        [=](const ck_tile::stream_config& s_){ fmha_bwd_convert_dq_oneshot_<convert_dq_trait_>(s_, a); }
    );
}

template <typename dot_do_o_trait_, typename dq_dk_dv_v3_traits_, typename convert_dq_trait_>
float fmha_bwd_v3_genl_(const ck_tile::stream_config& s, fmha_bwd_args a)
{
    if(s.log_level_ > 0)
        std::cout << ", " << fmha_bwd_dot_do_o_get_name_<dot_do_o_trait_>() << ", " << FmhaBwdV3Name<dq_dk_dv_v3_traits_>::bwd_v3_name << ", " << fmha_bwd_convert_dq_get_name_<convert_dq_trait_>() << std::flush;
    fmha_bwd_v3_genl_args args;
    args.ptr_dq   = a.dq_acc_ptr;
    args.ptr_dk   = a.dk_ptr;
    args.ptr_dv   = a.dv_ptr;
    args.ptr_q    = a.q_ptr;
    args.ptr_k    = a.k_ptr;
    args.ptr_v    = a.v_ptr;
    args.ptr_do   = a.do_ptr;
    args.ptr_lse  = a.lse_ptr;
    args.ptr_d    = a.d_ptr;
    args.scalar   = a.scale;
    args.log2e    = ck_tile::log2e_v<float>;
    args.ratio    = a.nhead_q / a.nhead_k;
    args.seqlen_q = a.seqlen_q;
    args.seqlen_k = a.seqlen_k;
    args.head_dim = a.hdim_q;
    args.nhead_q  = a.nhead_q;
    args.Hs_q     = a.nhead_stride_q * 2;
    args.BAs_q    = a.batch_stride_q * 2;
    args.Seqs_q   = a.stride_q * 2;
    args.Hs_k     = a.nhead_stride_k * 2;
    args.BAs_k    = a.batch_stride_k * 2;
    args.Seqs_k   = a.stride_k * 2;
    args.Hs_v     = a.nhead_stride_v * 2;
    args.BAs_v    = a.batch_stride_v * 2;
    args.Seqs_v   = a.stride_v * 2;
    args.Hs_do    = a.nhead_stride_do * 2;
    args.BAs_do   = a.batch_stride_do * 2;
    args.Seqs_do  = a.stride_do * 2;
    args.Hs_dk    = a.nhead_stride_dk * 2;
    args.BAs_dk   = a.batch_stride_dk * 2;
    args.Seqs_dk  = a.stride_dk * 2;
    args.Hs_dv    = a.nhead_stride_dv * 2;
    args.BAs_dv   = a.batch_stride_dv * 2;
    args.Seqs_dv  = a.stride_dv * 2;

    auto traits = fmha_bwd_v3_traits{a.batch,
                                      a.nhead_q,
                                      a.seqlen_k,
                                      a.hdim_q,
                                      a.mask_type,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_qo,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv};
    static thread_local fmha_bwd_v3_kernel impl(FmhaBwdV3Name<dq_dk_dv_v3_traits_>::bwd_v3_name, FmhaBwdV3Buf<dq_dk_dv_v3_traits_>::bwd_v3_buf); // static here is for thread safety.
    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){ fmha_bwd_dot_do_o_oneshot_<dot_do_o_trait_>(s_, a); },
        [=](const ck_tile::stream_config& s_){ impl.launch_kernel(traits, args, s_); },
        [=](const ck_tile::stream_config& s_){ fmha_bwd_convert_dq_oneshot_<convert_dq_trait_>(s_, a); }
    );
}

template <typename dot_do_o_trait_, typename dq_dk_dv_v3_traits_, typename convert_dq_trait_>
float fmha_bwd_v3_group_(const ck_tile::stream_config& s, fmha_bwd_args a)
{
    if(s.log_level_ > 0)
        std::cout << ", " << fmha_bwd_dot_do_o_get_name_<dot_do_o_trait_>() << ", " << FmhaBwdV3Name<dq_dk_dv_v3_traits_>::bwd_v3_name << ", " << fmha_bwd_convert_dq_get_name_<convert_dq_trait_>() << std::flush;

    fmha_bwd_v3_group_args args;
    auto seqstart_q = reinterpret_cast<const int32_t*>(a.seqstart_q_ptr);
    auto seqstart_k = reinterpret_cast<const int32_t*>(a.seqstart_k_ptr);
    args.ptr_dq   = a.dq_acc_ptr;
    args.ptr_dk   = a.dk_ptr;
    args.ptr_dv   = a.dv_ptr;
    args.ptr_q    = a.q_ptr;
    args.ptr_k    = a.k_ptr;
    args.ptr_v    = a.v_ptr;
    args.ptr_do   = a.do_ptr;
    args.ptr_lse  = a.lse_ptr;
    args.ptr_d    = a.d_ptr;

    args.scalar   = a.scale;
    args.log2e    = ck_tile::log2e_v<float>;
    args.ratio    = a.nhead_q / a.nhead_k;
    args.seqlen_q = seqstart_q[a.batch];
    args.seqlen_k = seqstart_k[a.batch];
    args.Hs_q     = a.nhead_stride_q * 2;
    args.Seqs_q   = a.stride_q * 2;
    args.Hs_k     = a.nhead_stride_k * 2;
    args.Seqs_k   = a.stride_k * 2;
    args.Hs_v     = a.nhead_stride_v * 2;
    args.Seqs_v   = a.stride_v * 2;
    args.Hs_do    = a.nhead_stride_do * 2;
    args.Seqs_do  = a.stride_do * 2;
    args.Hs_dk    = a.nhead_stride_dk * 2;
    args.Seqs_dk  = a.stride_dk * 2;
    args.Hs_dv    = a.nhead_stride_dv * 2;
    args.Seqs_dv  = a.stride_dv * 2;
    args.ptr_qseq = a.seqstart_q_ptr;
    args.ptr_kseq = a.seqstart_k_ptr;
    args.head_dim = a.hdim_q;

    auto traits = fmha_bwd_v3_traits{ a.batch,
                                       a.nhead_q,
                                       a.max_seqlen_k,
                                       a.hdim_q,
                                       a.mask_type,
                                       FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_qo,
                                       FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv };
    static thread_local fmha_bwd_v3_kernel impl(FmhaBwdV3Name<dq_dk_dv_v3_traits_>::bwd_v3_name, FmhaBwdV3Buf<dq_dk_dv_v3_traits_>::bwd_v3_buf); // static here is for thread safety.
    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){ fmha_bwd_dot_do_o_oneshot_<dot_do_o_trait_>(s_, a); },
        [=](const ck_tile::stream_config& s_){ impl.launch_kernel(traits, args, s_); },
        [=](const ck_tile::stream_config& s_){ fmha_bwd_convert_dq_oneshot_<convert_dq_trait_>(s_, a); }
    );
}

// SWA supposes to include following circumstances:
// 1. FA style SWA: t/b: mask_left > 0 or mask_right > 0
// 2. xformer style SWA: xt / xb: window_size > 0
// 3. generic style SWA: g: x, y (TODO: ck doesn't support generic style)
// after preprocessing, 1 & 2 can be unioned into:
// mask_type == mask_top_left or mask_bottom_right
// left > 0 or right > 0
template <typename dot_do_o_trait_, typename dq_dk_dv_v3_traits_, typename convert_dq_trait_>
float fmha_bwd_v3_swa_genl_(const ck_tile::stream_config& s, fmha_bwd_args a)
{
    if(s.log_level_ > 0)
        std::cout << ", " << fmha_bwd_dot_do_o_get_name_<dot_do_o_trait_>() << ", " << FmhaBwdV3Name<dq_dk_dv_v3_traits_>::bwd_v3_name << ", " << fmha_bwd_convert_dq_get_name_<convert_dq_trait_>() << std::flush;
    fmha_bwd_v3_swa_genl_args args;
    args.ptr_dq   = a.dq_acc_ptr;
    args.ptr_dk   = a.dk_ptr;
    args.ptr_dv   = a.dv_ptr;
    args.ptr_q    = a.q_ptr;
    args.ptr_k    = a.k_ptr;
    args.ptr_v    = a.v_ptr;
    args.ptr_do   = a.do_ptr;
    args.ptr_lse  = a.lse_ptr;
    args.ptr_d    = a.d_ptr;
    args.scalar   = a.scale;
    args.log2e    = ck_tile::log2e_v<float>;
    args.ratio    = a.nhead_q / a.nhead_k;
    args.seqlen_q = a.seqlen_q;
    args.seqlen_k = a.seqlen_k;
    args.head_dim = a.hdim_q;
    args.nhead_q = a.nhead_q;
    args.Hs_q     = a.nhead_stride_q * 2;
    args.BAs_q    = a.batch_stride_q * 2;
    args.Seqs_q   = a.stride_q * 2;
    args.Hs_k     = a.nhead_stride_k * 2;
    args.BAs_k    = a.batch_stride_k * 2;
    args.Seqs_k   = a.stride_k * 2;
    args.Hs_v     = a.nhead_stride_v * 2;
    args.BAs_v    = a.batch_stride_v * 2;
    args.Seqs_v   = a.stride_v * 2;
    args.Hs_do    = a.nhead_stride_do * 2;
    args.BAs_do   = a.batch_stride_do * 2;
    args.Seqs_do  = a.stride_do * 2;
    args.Hs_dk    = a.nhead_stride_dk * 2;
    args.BAs_dk   = a.batch_stride_dk * 2;
    args.Seqs_dk  = a.stride_dk * 2;
    args.Hs_dv    = a.nhead_stride_dv * 2;
    args.BAs_dv   = a.batch_stride_dv * 2;
    args.Seqs_dv  = a.stride_dv * 2;

    // convert l/r to x/y HERE
    auto generic_mask = ck_tile::make_generic_attention_mask_coordinates_from_lr_window(a.window_size_left, a.window_size_right, a.seqlen_q, a.seqlen_k, (a.mask_type == static_cast<ck_tile::index_t>(mask_enum::mask_top_left) || a.mask_type == static_cast<ck_tile::index_t>(mask_enum::window_generic)));
    args.mask_y = generic_mask.at(ck_tile::number<0>{});
    args.mask_x = generic_mask.at(ck_tile::number<1>{});

    auto traits = fmha_bwd_v3_traits{a.batch,
                                      a.nhead_q,
                                      a.seqlen_k,
                                      a.hdim_q,
                                      a.mask_type,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_qo,
                                      FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::ts_kv};
    static thread_local fmha_bwd_v3_kernel impl(FmhaBwdV3Name<dq_dk_dv_v3_traits_>::bwd_v3_name, FmhaBwdV3Buf<dq_dk_dv_v3_traits_>::bwd_v3_buf); // static here is for thread safety.
    return ck_tile::launch_kernel(s,
        [=](const ck_tile::stream_config& s_){ fmha_bwd_dot_do_o_oneshot_<dot_do_o_trait_>(s_, a); },
        [=](const ck_tile::stream_config& s_){ impl.launch_kernel(traits, args, s_); },
        [=](const ck_tile::stream_config& s_){ fmha_bwd_convert_dq_oneshot_<convert_dq_trait_>(s_, a); }
    );
}

float fmha_bwd_v3(mha_bwd_traits t, fmha_bwd_args a, const ck_tile::stream_config& s){
    float r = -1;

    // if (t.use_ext_asm == true){
    if ((t.bias_type == bias_enum::no_bias) && (t.has_dbias == false) && (t.has_dropout == false) &&
                (t.is_deterministic == false) && (a.hdim_q == a.hdim_v) && (a.nhead_q % a.nhead_k == 0)) {
        if((t.is_group_mode == false) && (a.hdim_q > 128) && (a.hdim_q <= 192) && (a.hdim_q % 8 == 0)){
            if(t.data_type.compare("fp16") == 0){
                if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                    if(t.mask_type == mask_enum::no_mask){
                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<256, FmhaBwdFp16, false, true, true>;
                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdFp16, false, true, 0, true, true>;
                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<256, FmhaBwdFp16, false, true, true, false>;
                        // const std::string bwd_v3_name = "bwd_v3_hd192_fp16_a32_psskddv";
                        r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                        return r;
                    }
                    else if((((t.mask_type != mask_enum::no_mask) && (a.seqlen_q == a.seqlen_k)) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left))) &&
                            ((a.window_size_left == -1) && (a.window_size_right == 0))){
                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<256, FmhaBwdFp16, false, true, true>;
                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdFp16, true, true, 0, true, true>;
                        using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<256, FmhaBwdFp16, false, true, true, false>;
                        // const std::string bwd_v3_name = "bwd_v3_hd192_fp16_causal_a32_psskddv";
                        r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                        return r;
                    }
                }
            }
            else if(t.data_type.compare("bf16") == 0){
                if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                    if(t.mask_type == mask_enum::no_mask){
                        if(t.how_v3_bf16_cvt == 0){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<256, FmhaBwdBf16, false, true, true>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16, false, true, 0, true, true>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<256, FmhaBwdBf16, false, true, true, false>;
                            // const std::string bwd_v3_name = "bwd_v3_hd192_bf16_a32_rtne_psskddv";
                            r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                            return r;
                        }
                        else if(t.how_v3_bf16_cvt == 1){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<256, FmhaBwdBf16, false, true, true>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16, false, true, 1, true, true>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<256, FmhaBwdBf16, false, true, true, false>;
                            // const std::string bwd_v3_name = "bwd_v3_hd192_bf16_a32_rtna_psskddv";
                            r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                            return r;
                        }
                        else if(t.how_v3_bf16_cvt == 2){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<256, FmhaBwdBf16, false, true, true>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16, false, true, 2, true, true>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<256, FmhaBwdBf16, false, true, true, false>;
                            // const std::string bwd_v3_name = "bwd_v3_hd192_bf16_a32_rtz_psskddv";
                            r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                            return r;
                        }
                    }
                    else if((((t.mask_type != mask_enum::no_mask) && (a.seqlen_q == a.seqlen_k)) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left))) &&
                            ((a.window_size_left == -1) && (a.window_size_right == 0))){
                        if(t.how_v3_bf16_cvt == 0){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<256, FmhaBwdBf16, false, true, true>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16, true, true, 0, true, true>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<256, FmhaBwdBf16, false, true, true, false>;
                            // const std::string bwd_v3_name = "bwd_v3_hd192_bf16_causal_a32_rtne_psskddv";
                            r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                            return r;
                        }
                        else if(t.how_v3_bf16_cvt == 1){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<256, FmhaBwdBf16, false, true, true>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16, true, true, 1, true, true>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<256, FmhaBwdBf16, false, true, true, false>;
                            // const std::string bwd_v3_name = "bwd_v3_hd192_bf16_causal_a32_rtna_psskddv";
                            r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                            return r;
                        }
                        else if(t.how_v3_bf16_cvt == 2){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<256, FmhaBwdBf16, false, true, true>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<192, FmhaBwdBf16, true, true, 2, true, true>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<256, FmhaBwdBf16, false, true, true, false>;
                            // const std::string bwd_v3_name = "bwd_v3_hd192_bf16_causal_a32_rtz_psskddv";
                            r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                            return r;
                        }
                    }
                }
            }
        }
        else if((a.hdim_q > 64) && (a.hdim_q <= 128) && (a.hdim_q % 8 == 0)){
            if(t.data_type.compare("fp16") == 0){
                if((t.is_group_mode == false) && (t.mask_type == mask_enum::no_mask)){
                    if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                        if((a.hdim_q == 128) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                    (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                    (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, false>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, false, true, 0, false, false>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, false, false, false>;
                            // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_a32";
                            r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                            return r;
                        }
                        else if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, false>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, false, true, 0, true, true>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, false, false, false>;
                            // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_a32_psskddv";
                            r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                            return r;
                        }
                        else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, true, false>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, false, true, 0, true, true>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, true, false, false>;
                            // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_a32_psskddv";
                            r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                            return r;
                        }
                        else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, true>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, false, true, 0, true, true>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, false, true, false>;
                            // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_a32_psskddv";
                            r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                            return r;
                        }
                        else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, true, true>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, false, true, 0, true, true>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, true, true, false>;
                            // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_a32_psskddv";
                            r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                            return r;
                        }
                    }
                    else if((t.is_v3_atomic_fp32 == false) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                        if(a.hdim_q == 128){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, false>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, false, false, 0, false, false>;
                            // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_a16";
                            r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                            return r;
                        }
                        else{
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, true>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, false, false, 0, false, true>;
                            // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_a16_pddv";
                            r = fmha_bwd_v3_gen_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                            return r;
                        }
                    }
                }
                else if((t.is_group_mode == true) && (t.mask_type == mask_enum::no_mask)){//group mode
                    if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                        if(a.hdim_q == 128){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, true, true, false>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, false/*causal*/, true/*Atimoc32*/, 0, true/*PadS*/, false/*PadD*/, true/*group*/>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, true, true, false, false>;
                            // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_a32_pssk_group";
                            r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                            return r;
                        }
                        else{
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, true, true, true>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, false/*causal*/, true/*Atimoc32*/, 0, true/*PadS*/, true/*PadD*/, true/*group*/>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, true, true, true, false>;
                            // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_a32_psskddv_group";
                            r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                            return r;
                        }
                    }
                }
                else if((t.is_group_mode == false) && (t.mask_type != mask_enum::no_mask) && ((a.window_size_left == -1) && (a.window_size_right == 0))){
                    if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                        if((a.hdim_q == 128) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                    (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                    (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, false>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, true, true, 0, false, false>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, false, false, false>;
                            // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_causal_a32";
                            r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                            return r;
                        }
                        else if((a.seqlen_q == a.seqlen_k) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left))){
                            if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, true, true, 0, true, true>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, false, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_causal_a32_psskddv";
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, true, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, true, true, 0, true, true>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, true, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_causal_a32_psskddv";
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, true, true, 0, true, true>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, false, true, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_causal_a32_psskddv";
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, true, true, 0, true, true>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, true, true, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_causal_a32_psskddv";
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                        }
                    }
                    else if((t.is_v3_atomic_fp32 == false) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                        if(a.hdim_q == 128){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, false>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, true, false, 0, false, false>;
                            // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_causal_a16";
                            r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                            return r;
                        }
                        else{
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, true>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, true, false, 0, false, true>;
                            // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_causal_a16_pddv";
                            r = fmha_bwd_v3_gen_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                            return r;
                        }
                    }
                }
                else if(((t.mask_type == mask_enum::mask_top_left || t.mask_type == mask_enum::mask_bottom_right) && ((a.window_size_left > 0) || (a.window_size_right > 0))) || (t.mask_type == mask_enum::window_generic)){
                    if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                        if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, false>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, 2, true, 0, true, true>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, false, false, false>;
                            // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_swa_a32_rtne_psskddv";
                            r = fmha_bwd_v3_swa_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                            return r;
                        }
                        else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, true, false>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, 2, true, 0, true, true>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, true, false, false>;
                            // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_swa_a32_rtne_psskddv";
                            r = fmha_bwd_v3_swa_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                            return r;
                        }
                        else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, false, true>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, 2, true, 0, true, true>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, false, true, false>;
                            // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_swa_a32_rtne_psskddv;
                            r = fmha_bwd_v3_swa_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                            return r;
                        }
                        else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, false, true, true>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, 2, true, 0, true, true>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, false, true, true, false>;
                            // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_swa_a32_rtne_psskddv";
                            r = fmha_bwd_v3_swa_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                            return r;
                        }
                    }
                }
                else if((t.is_group_mode == true) && (t.mask_type != mask_enum::no_mask) && ((a.window_size_left == -1) && (a.window_size_right == 0))){//group mode
                    if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/) && (t.mask_type == mask_enum::mask_top_left)){
                        if(a.hdim_q == 128){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, true, true, false>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, true/*causal*/, true/*Atimoc32*/, 0, true/*PadS*/, false/*PadD*/, true/*group*/>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, true, true, false, false>;
                            // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_causal_a32_pssk_group";
                            r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                            return r;
                        }
                        else{
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdFp16, true, true, true>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdFp16, true/*causal*/, true/*Atimoc32*/, 0, true/*PadS*/, true/*PadD*/, true/*group*/>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdFp16, true, true, true, false>;
                            // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_causal_a32_psskddv_group";
                            r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                            return r;
                        }
                    }
                }
            }
            else if(t.data_type.compare("bf16") == 0){
                if((t.is_group_mode == false) && (t.mask_type == mask_enum::no_mask)){
                    if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                        if(t.how_v3_bf16_cvt == 0){
                            if((a.hdim_q == 128) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % {F_seqlen_limit} == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                        (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                        (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 0, false, false>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtne";
                                r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 0, true, true>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtne_psskddv";
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 0, true, true>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtne_psskddv";
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 0, true, true>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, true, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtne_psskddv";
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 0, true, true>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, true, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtne_psskddv";
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                        }
                        else if(t.how_v3_bf16_cvt == 1){
                            if((a.hdim_q == 128) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % {F_seqlen_limit} == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                        (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                        (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 1, false, false>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtna";
                                r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 1, true, true>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtna_psskddv";
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 1, true, true>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtna_psskddv";
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 1, true, true>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, true, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtna_psskddv";
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 1, true, true>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, true, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtna_psskddv";
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                        }
                        else if(t.how_v3_bf16_cvt == 2){
                            if((a.hdim_q == 128) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % {F_seqlen_limit} == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                        (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                        (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 2, false, false>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtz";
                                r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 2, true, true>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtz_psskddv";
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 2, true, true>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtz_psskddv";
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 2, true, true>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, true, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtz_psskddv";
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, true, 2, true, true>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, true, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtz_psskddv";
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                        }
                    }
                    else if((t.is_v3_atomic_fp32 == false) && (a.seqlen_q == a.seqlen_k) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                        if(t.how_v3_bf16_cvt == 0){
                            if(a.hdim_q == 128 && (a.seqlen_k % {F_seqlen_limit} == 0)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, false, 0, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a16_rtne";
                                r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                return r;
                            }
                            else if(a.hdim_q != 128 && (a.seqlen_k % 64 == 0)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, false, 0, false, true>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a16_rtne_pddv";
                                r = fmha_bwd_v3_gen_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                return r;
                            }
                        }
                        else if(t.how_v3_bf16_cvt == 1){
                            if(a.hdim_q == 128 && (a.seqlen_k % {F_seqlen_limit} == 0)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, false, 1, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a16_rtna";
                                r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                return r;
                            }
                            else if(a.hdim_q != 128 && (a.seqlen_k % 64 == 0)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, false, 1, false, true>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a16_rtna_pddv";
                                r = fmha_bwd_v3_gen_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                return r;
                            }
                        }
                        else if(t.how_v3_bf16_cvt == 2){
                            if(a.hdim_q == 128 && (a.seqlen_k % {F_seqlen_limit} == 0)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, false, 2, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a16_rtz";
                                r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                return r;
                            }
                            else if(a.hdim_q != 128 && (a.seqlen_k % 64 == 0)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false, false, 2, false, true>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a16_rtz_pddv";
                                r = fmha_bwd_v3_gen_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                return r;
                            }
                        }
                    }
                }
                else if((t.is_group_mode == true) && (t.mask_type == mask_enum::no_mask)){ //group mode
                    if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                        if(t.how_v3_bf16_cvt == 0){
                            if(a.hdim_q == 128){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, true, true, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false/*causal*/, true/*Atimoc32*/, 0, true/*PadS*/, false/*PadD*/, true/*group*/>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, true, true, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtne_pssk_group";
                                r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else{
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, true, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false/*causal*/, true/*Atimoc32*/, 0, true/*PadS*/, true/*PadD*/, true/*group*/>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, true, true, true, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_a32_rtne_psskddv_group";
                                r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                        }
                        else if(t.how_v3_bf16_cvt == 1){
                            if(a.hdim_q == 128){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, true, true, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false/*causal*/, true/*Atimoc32*/, 1, true/*PadS*/, false/*PadD*/, true/*group*/>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, true, true, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_a32_rtna_pssk_group";
                                r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else{
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, true, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false/*causal*/, true/*Atimoc32*/, 1, true/*PadS*/, true/*PadD*/, true/*group*/>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, true, true, true, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_a32_rtna_psskddv_group";
                                r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                        }
                        else if(t.how_v3_bf16_cvt == 2){
                            if(a.hdim_q == 128){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, true, true, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false/*causal*/, true/*Atimoc32*/, 2, true/*PadS*/, false/*PadD*/, true/*group*/>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, true, true, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_a32_rtz_pssk_group";
                                r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else{
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, true, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, false/*causal*/, true/*Atimoc32*/, 2, true/*PadS*/, true/*PadD*/, true/*group*/>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, true, true, true, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_a32_rtz_psskddv_group";
                                r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                        }
                    }
                }
                else if((t.is_group_mode == false) && (t.mask_type != mask_enum::no_mask) && ((a.window_size_left == -1) && (a.window_size_right == 0))){
                    if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                        if(t.how_v3_bf16_cvt == 0){
                            if((a.hdim_q == 128) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % {F_seqlen_limit} == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                        (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                        (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 0, false, false>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtne";
                                r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if((a.seqlen_q == a.seqlen_k) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left))){
                                if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 0, true, true>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtne_psskddv";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 0, true, true>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtne_psskddv";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 0, true, true>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, true, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtne_psskddv";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 0, true, true>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, true, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtne_psskddv";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                            }
                        }
                        else if(t.how_v3_bf16_cvt == 1){
                            if((a.hdim_q == 128) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % {F_seqlen_limit} == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                        (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                        (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 1, false, false>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtna";
                                r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if((a.seqlen_q == a.seqlen_k) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left))){
                                if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 1, true, true>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtna_psskddv";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 1, true, true>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtna_psskddv";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 1, true, true>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, true, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtna_psskddv";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 1, true, true>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, true, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtna_psskddv";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                            }
                        }
                        else if(t.how_v3_bf16_cvt == 2){
                            if((a.hdim_q == 128) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % {F_seqlen_limit} == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                        (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                        (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 2, false, false>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtz";
                                r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if((a.seqlen_q == a.seqlen_k) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left))){
                                if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 2, true, true>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtz_psskddv";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 2, true, true>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtz_psskddv";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 2, true, true>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, true, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtz_psskddv";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, true>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, true, 2, true, true>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, true, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtz_psskddv";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                            }
                        }
                    }
                    else if((t.is_v3_atomic_fp32 == false) && (a.seqlen_q == a.seqlen_k) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                        if(t.how_v3_bf16_cvt == 0){
                            if(a.hdim_q == 128  && (a.seqlen_k % {F_seqlen_limit} == 0)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, false, 0, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a16_rtne";
                                r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                return r;
                            }
                            else if(a.hdim_q != 128  && (a.seqlen_k % 64 == 0)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, false, 0, false, true>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a16_rtne_pddv";
                                r = fmha_bwd_v3_gen_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                return r;
                            }
                        }
                        else if(t.how_v3_bf16_cvt == 1){
                            if(a.hdim_q == 128  && (a.seqlen_k % {F_seqlen_limit} == 0)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, false, 1, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a16_rtna";
                                r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                return r;
                            }
                            else if(a.hdim_q != 128  && (a.seqlen_k % 64 == 0)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, false, 1, false, true>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a16_rtna_pddv";
                                r = fmha_bwd_v3_gen_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                return r;
                            }
                        }
                        else if(t.how_v3_bf16_cvt == 2){
                            if(a.hdim_q == 128  && (a.seqlen_k % {F_seqlen_limit} == 0)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, false, 2, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a16_rtz";
                                r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                return r;
                            }
                            else if(a.hdim_q != 128  && (a.seqlen_k % 64 == 0)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true, false, 2, false, true>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a16_rtz_pddv";
                                r = fmha_bwd_v3_gen_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                                return r;
                            }
                        }
                    }
                }
                else if(((t.mask_type == mask_enum::mask_top_left || t.mask_type == mask_enum::mask_bottom_right) && ((a.window_size_left > 0) || (a.window_size_right > 0))) || (t.mask_type == mask_enum::window_generic)){
                    if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                        if(t.how_v3_bf16_cvt == 0){
                            if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 2, true, 0, true, true>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_swa_a32_rtne_psskddv";
                                r = fmha_bwd_v3_swa_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 2, true, 0, true, true>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_swa_a32_rtne_psskddv";
                                r = fmha_bwd_v3_swa_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 2, true, 0, true, true>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, true, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_swa_a32_rtne_psskddv;
                                r = fmha_bwd_v3_swa_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 2, true, 0, true, true>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, true, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_swa_a32_rtne_psskddv";
                                r = fmha_bwd_v3_swa_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                        }
                        else if(t.how_v3_bf16_cvt == 1){
                            if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 2, true, 1, true, true>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_swa_a32_rtna_psskddv";
                                r = fmha_bwd_v3_swa_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 2, true, 1, true, true>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_swa_a32_rtna_psskddv";
                                r = fmha_bwd_v3_swa_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 2, true, 1, true, true>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, true, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_swa_a32_rtna_psskddv;
                                r = fmha_bwd_v3_swa_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 2, true, 1, true, true>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, true, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_swa_a32_rtna_psskddv";
                                r = fmha_bwd_v3_swa_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                        }
                        else if(t.how_v3_bf16_cvt == 2){
                            if((a.seqlen_q % 64 == 0) && (a.hdim_q == 128)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 2, true, 2, true, true>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_swa_a32_rtz_psskddv";
                                r = fmha_bwd_v3_swa_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if((a.seqlen_q % 64 != 0) && (a.hdim_q == 128)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 2, true, 2, true, true>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_swa_a32_rtz_psskddv";
                                r = fmha_bwd_v3_swa_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if((a.seqlen_q % 64 == 0) && (a.hdim_q != 128)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, false, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 2, true, 2, true, true>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, false, true, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_swa_a32_rtz_psskddv;
                                r = fmha_bwd_v3_swa_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else if((a.seqlen_q % 64 != 0) && (a.hdim_q != 128)){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, false, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, 2, true, 2, true, true>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, false, true, true, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_swa_a32_rtz_psskddv";
                                r = fmha_bwd_v3_swa_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                        }
                    }
                }
                else if((t.is_group_mode == true) && (t.mask_type != mask_enum::no_mask) && ((a.window_size_left == -1) && (a.window_size_right == 0))){//group mode
                    if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/) && (t.mask_type == mask_enum::mask_top_left)){
                        if(t.how_v3_bf16_cvt == 0){
                            if(a.hdim_q == 128){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, true, true, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true/*causal*/, true/*Atimoc32*/, 0, true/*PadS*/, false/*PadD*/, true/*group*/>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, true, true, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtne_pssk_group";
                                r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else{
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, true, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true/*causal*/, true/*Atimoc32*/, 0, true/*PadS*/, true/*PadD*/, true/*group*/>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, true, true, true, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_bf16_causal_a32_rtne_psskddv_group";
                                r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                        }
                        else if(t.how_v3_bf16_cvt == 1){
                            if(a.hdim_q == 128){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, true, true, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true/*causal*/, true/*Atimoc32*/, 1, true/*PadS*/, false/*PadD*/, true/*group*/>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, true, true, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_causal_a32_rtna_pssk_group";
                                r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else{
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, true, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true/*causal*/, true/*Atimoc32*/, 1, true/*PadS*/, true/*PadD*/, true/*group*/>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, true, true, true, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_causal_a32_rtna_psskddv_group";
                                r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                        }
                        else if(t.how_v3_bf16_cvt == 2){
                            if(a.hdim_q == 128){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, true, true, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true/*causal*/, true/*Atimoc32*/, 2, true/*PadS*/, false/*PadD*/, true/*group*/>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, true, true, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_causal_a32_rtz_pssk_group";
                                r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else{
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<128, FmhaBwdBf16, true, true, true>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<128, FmhaBwdBf16, true/*causal*/, true/*Atimoc32*/, 2, true/*PadS*/, true/*PadD*/, true/*group*/>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<128, FmhaBwdBf16, true, true, true, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd128_fp16_causal_a32_rtz_psskddv_group";
                                r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                        }
                    }
                }
            }
        }
        else if(a.hdim_q == 64){
            if(t.data_type.compare("fp16") == 0){
                if(t.mask_type == mask_enum::no_mask){
                    if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                        if(t.is_group_mode == false){
                            if(a.seqlen_q % 64 == 0){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdFp16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdFp16, false, true, 0, true, false>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdFp16, false, false, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd64_fp16_a32_pssk";
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else{
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdFp16, false, true, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdFp16, false, true, 0, true, false>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdFp16, false, true, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd64_fp16_a32_pssk";
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                        }
                        else{
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdFp16, true, true, false>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdFp16, false, true, 0, true, false, true>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdFp16, true, true, false, false>;
                            // const std::string bwd_v3_name = "bwd_v3_hd64_fp16_a32_pssk_group";
                            r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                            return r;
                        }
                    }
                    else if((t.is_v3_atomic_fp32 == false) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdFp16, false, false, false>;
                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdFp16, false, false, 0, false, false>;
                        // const std::string bwd_v3_name = "bwd_v3_hd64_fp16_a16";
                        r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                        return r;
                    }
                }
                else if((t.mask_type != mask_enum::no_mask) && ((a.window_size_left == -1) && (a.window_size_right == 0))){
                    if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                        if((t.is_group_mode == false) && ((a.seqlen_q == a.seqlen_k) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left)))){
                            if(a.seqlen_q % 64 == 0){
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdFp16, false, false, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdFp16, true, true, 0, true, false>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdFp16, false, false, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd64_fp16_causal_a32_pssk";
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                            else{
                                using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdFp16, false, true, false>;
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdFp16, true, true, 0, true, false>;
                                using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdFp16, false, true, false, false>;
                                // const std::string bwd_v3_name = "bwd_v3_hd64_fp16_causal_a32_pssk";
                                r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                return r;
                            }
                        }
                        else if((t.is_group_mode == true) && (t.mask_type == mask_enum::mask_top_left)){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdFp16, true, true, false>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdFp16, true, true, 0, true, false, true>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdFp16, true, true, false, false>;
                            // const std::string bwd_v3_name = "bwd_v3_hd64_fp16_causal_a32_pssk_group";
                            r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                            return r;
                        }
                    }
                    else if((t.is_v3_atomic_fp32 == false) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                        using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdFp16, false, false, false>;
                        using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdFp16, true, false, 0, false, false>;
                        // const std::string bwd_v3_name = "bwd_v3_hd64_fp16_causal_a16";
                        r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                        return r;
                    }
                }
            }
            else if(t.data_type.compare("bf16") == 0){
                if(t.mask_type == mask_enum::no_mask){
                    if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                        if(t.is_group_mode == false){
                            if(t.how_v3_bf16_cvt == 0){
                                if(a.seqlen_q % 64 == 0){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, true, 0, true, false>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, false, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a32_rtne_pssk";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else{
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, true, 0, true, false>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, true, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a32_rtne_pssk";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                if(a.seqlen_q % 64 == 0){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, true, 1, true, false>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, false, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a32_rtna_pssk";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else{
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, true, 1, true, false>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, true, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a32_rtna_pssk";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                if(a.seqlen_q % 64 == 0){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, true, 2, true, false>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, false, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a32_rtz_pssk";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else{
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, true, 2, true, false>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, true, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a32_rtz_pssk";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                            }
                        }
                        else{
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, true, true, false>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, true, true, false, false>;
                            if(t.how_v3_bf16_cvt == 0){
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, true, 0, true, false, true>;
                                r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, true, 1, true, false, true>;
                                r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                            }
                            else{
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, true, 2, true, false, true>;
                                r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                            }
                            return r;
                        }
                    }
                    else if((t.is_v3_atomic_fp32 == false) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                        if(t.how_v3_bf16_cvt == 0){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, false, 0, false, false>;
                            // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a16_rtne";
                            r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                            return r;
                        }
                        else if(t.how_v3_bf16_cvt == 1){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, false, 1, false, false>;
                            // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a16_rtna";
                            r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                            return r;
                        }
                        else if(t.how_v3_bf16_cvt == 2){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, false, false, 2, false, false>;
                            // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_a16_rtz";
                            r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                            return r;
                        }
                    }
                }
                else if((t.mask_type != mask_enum::no_mask) && ((a.window_size_left == -1) && (a.window_size_right == 0))){
                    if((t.is_v3_atomic_fp32 == true) && (a.nhead_stride_dq_acc >= a.stride_dq_acc /*dq_acc only support BHSD*/)){
                        if((t.is_group_mode == false) && ((a.seqlen_q == a.seqlen_k) || ((a.seqlen_q != a.seqlen_k) && (t.mask_type == mask_enum::mask_top_left)))){
                            if(t.how_v3_bf16_cvt == 0){
                                if(a.seqlen_q % 64 == 0){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, true, 0, true, false>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, false, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a32_rtne_pssk";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else{
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, true, 0, true, false>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, true, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a32_rtne_pssk";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                if(a.seqlen_q % 64 == 0){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, true, 1, true, false>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, false, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a32_rtna_pssk";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else{
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, true, 1, true, false>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, true, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a32_rtna_pssk";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                            }
                            else if(t.how_v3_bf16_cvt == 2){
                                if(a.seqlen_q % 64 == 0){
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, true, 2, true, false>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, false, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a32_rtz_pssk";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                                else{
                                    using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, true, false>;
                                    using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, true, 2, true, false>;
                                    using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, false, true, false, false>;
                                    // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a32_rtz_pssk";
                                    r = fmha_bwd_v3_genl_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                                    return r;
                                }
                            }
                        }
                        else if((t.is_group_mode == true) && (t.mask_type == mask_enum::mask_top_left)){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, true, true, false>;
                            using convert_dq_trait_ = fmha_bwd_convert_dq_traits_<64, FmhaBwdBf16, true, true, false, false>;
                            if(t.how_v3_bf16_cvt == 0){
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, true, 0, true, false, true>;
                                r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                            }
                            else if(t.how_v3_bf16_cvt == 1){
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, true, 1, true, false, true>;
                                r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                            }
                            else{
                                using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, true, 2, true, false, true>;
                                r = fmha_bwd_v3_group_<dot_do_o_trait_, dq_dk_dv_v3_traits_, convert_dq_trait_>(s, a);
                            }
                            return r;
                        }
                    }
                    else if((t.is_v3_atomic_fp32 == false) && (a.seqlen_q == a.seqlen_k) && (a.seqlen_k % 64 == 0) && (a.stride_q == a.stride_do) && (a.nhead_stride_q == a.nhead_stride_do) && (a.batch_stride_q == a.batch_stride_do) &&
                                (a.stride_k == a.stride_v) && (a.nhead_stride_k == a.nhead_stride_v) && (a.batch_stride_k == a.batch_stride_v) && (a.nhead_stride_k == a.nhead_stride_dk) && (a.nhead_stride_v == a.nhead_stride_dv) &&
                                (a.batch_stride_q >= a.stride_q) && (a.batch_stride_do >= a.stride_do) && ((a.batch_stride_dk / a.batch_stride_k) == (a.nhead_q / a.nhead_k)) && ((a.batch_stride_dv / a.batch_stride_v) == (a.nhead_q / a.nhead_k))){
                        if(t.how_v3_bf16_cvt == 0){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, false, 0, false, false>;
                            const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a16_rtne";
                            r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                            return r;
                        }
                        else if(t.how_v3_bf16_cvt == 1){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, false, 1, false, false>;
                            // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a16_rtna";
                            r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                            return r;
                        }
                        else if(t.how_v3_bf16_cvt == 2){
                            using dot_do_o_trait_ = fmha_bwd_dot_do_o_traits_<64, FmhaBwdBf16, false, false, false>;
                            using dq_dk_dv_v3_traits_ = fmha_bwd_dq_dk_dv_v3_traits_<64, FmhaBwdBf16, true, false, 2, false, false>;
                            // const std::string bwd_v3_name = "bwd_v3_hd64_bf16_causal_a16_rtz";
                            r = fmha_bwd_v3_<dot_do_o_trait_, dq_dk_dv_v3_traits_>(s, a);
                            return r;
                        }
                    }
                }
            }
        }
    }
    // } // if (t.use_ext_asm == true)

    return r;
}

} // namespace AOTRITON_NS::v3::flash::aiter
