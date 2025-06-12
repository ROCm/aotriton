// Copyright © 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/config.h>
#include <aotriton/_internal/util.h>
#include <aotriton/flash.h>
#include <aotriton/util.h>
#include <flash/iface.op_attn_bwd.h>
#include <flash/affine.bwd_dq_dk_dv_v3.h>
#include <iostream>
#include <algorithm>
#include <numeric_limits>

namespace AOTRITON_NS::v3::flash {

namespace ck_tile {
template <typename T>
struct log2e;

template <>
struct log2e<double>
{
    static constexpr double value = 1.44269504088896340736;
};

template <>
struct log2e<float>
{
    static constexpr float value = float(log2e<double>::value);
};

template <typename T = double>
constexpr T log2e_v = log2e<T>::value;

template <typename T = double>
constexpr T log2e_rcp_v = 1. / log2e<T>::value;
};

bool BwdDqDkDvV3Context::check_inputs_are_supported() {
  const auto& args = *params;
  // No bias support
  if (args.BIAS_TYPE) return false;
  // No Varlen support
  if (args.cu_seqlens_q) return false;
  if (args.cu_seqlens_k) return false;
  // Only support hdim <= 192
  if (args.head_dim > 192) return false;
  // TODO: support dropout kernel. fwd and bwd should have identical PRNG
  if (args.ENABLE_DROPOUT) return false;
  // AITER ASM kernel only reads u32 strides.
#define CHECK_STRIDE(T)                                               \
  do {                                                                \
    auto strides = T->strides();                                      \
    auto max_e = std::max_element(strides.begin(), strides.end());    \
    if (max_e * 2 > numeric_limits<uint32_t>::max) {                  \
      return false;                                                   \
    } while(0)
  CHECK_STRIDE(args.Q);
  CHECK_STRIDE(args.K);
  CHECK_STRIDE(args.V);
  CHECK_STRIDE(args.Out);
  CHECK_STRIDE(args.DO);
  CHECK_STRIDE(args.DK);
  CHECK_STRIDE(args.DV);
  CHECK_STRIDE(args.DQ);
  CHECK_STRIDE(args.DB);
#undef CHECK_STRIDE

  return true;
}

void BwdDqDkDvV3Context::calculate_residual_func_fields() {
    const auto& args = *params;
    auto check_if_uniform = [&]() -> bool {
        // Reject varlen
        if (args.cu_seqlens_q || args.cu_seqlens_k) return false;
        // TODO: GQA support
        if (args.num_head_q != args.num_head_k) return false;
#define CMP_TENSOR(X, Y)                                                \
        do {                                                            \
            if (args.X->strides() != args.Y->strides()) return false;   \
        } while(0)
        CMP_TENSOR(Q, K);
        CMP_TENSOR(Q, DO);
        CMP_TENSOR(K, V);
        // This is more restrict than AITER kernel
        CMP_TENSOR(K, DK);
        CMP_TENSOR(V, DV);
#undef CMP_TENSOR
        // Tensor Memory layout must be BHSD or BSHD
        // D-last is ensured by caller
        if (args.Q->stride(0) < args.Q->stride(2)) return false;

        if (args.max_seqlen_q != args.max_seqlen_k) return false;
        if (args.max_seqlen_q % 64) return false;
        return true;
    };
    auto check_hdim_regular = [&]() -> bool {
        if (args.head_dim == 64)
            return true;
        if (args.head_dim == 128)
            return true;
        if (args.head_dim == 192)
            return true;
        return false;
    };
    residual_args.kIsUniformStride = check_if_uniform();
    residual_args.kIsSEQPad = (args.max_seqlen_q % 64 != 0);
    residual_args.kIsHDPad = !check_hdim_regular();
    residual_args.kIsGroupMode = (args.num_head_q != args.num_head_k);
    auto check_mask_type = [&]() -> int8_t {
      if (args.CAUSAL_TYPE == CausalType::None)
        return 0;
      if (args.Window_left == WindowValue::TopLeftAligned &&
          args.Window_right == WindowValue::TopLeftAligned)
        return 1;
      return 2;
    };
    residual_args.MaskType = check_mask_type();
}

fmha_bwd_args
construct_fmha_bwd_args(const BwdDqDkDvV3Context& ctx) {
  const auto& args = *ctx.params;
  auto batch = args.Q->size(0);
  auto hdim_q = args.Q->size(3);
  auto hdim_v = args.V->size(3);
  auto nhead = args.Q->size(1);
  auto nhead_k = args.K->size(1);
  auto scale = args.sm_scale;
  auto stride_q = args.Q->stride(2);
  auto stride_k = args.K->stride(2);
  auto stride_v = args.V->stride(2);
  auto stride_o = args.O->stride(2);
  auto stride_do = args.DO->stride(2);
  auto stride_dq_acc = args.DQ_ACC->stride(2);
  auto stride_dq = args.DQ->stride(2);
  auto stride_dk = args.DK->stride(2);
  auto stride_dv = args.DV->stride(2);

  auto nhead_stride_q = args.Q->stride(1);
  auto nhead_stride_k = args.K->stride(1);
  auto nhead_stride_v = args.V->stride(1);
  auto nhead_stride_o = args.O->stride(1);
  auto nhead_stride_do = args.DO->stride(1);
  auto nhead_stride_dq_acc = args.DQ_ACC->stride(1);
  auto nhead_stride_dq = args.DQ->stride(1);
  auto nhead_stride_dk = args.DK->stride(1);
  auto nhead_stride_dv = args.DV->stride(1);
  auto nhead_stride_lsed = args.L->stride(1);

  auto batch_stride_q = args.Q->stride(0);
  auto batch_stride_k = args.K->stride(0);
  auto batch_stride_v = args.V->stride(0);
  auto batch_stride_o = args.O->stride(0);
  auto batch_stride_do = args.DO->stride(0);
  auto batch_stride_dq_acc = args.DQ_ACC->stride(0);
  auto batch_stride_dq = args.DQ->stride(0);
  auto batch_stride_dk = args.DK->stride(0);
  auto batch_stride_dv = args.DV->stride(0);
  auto batch_stride_lsed = args.L->stride(0);
  float p_drop = 0.0;
  float p_undrop = 0.0;
  auto drop_seed_offset = std::make_pair<uint64_t, uint64_t>(0, 0); // placeholder
  return {
    args.Q->data_ptr(),
    args.K->data_ptr(),
    args.V->data_ptr(),
    nullptr,    // unused, was handling bias
    args.Out->data_ptr(),
    args.L->data_ptr(),
    args.DO->data_ptr(),
    args.D->data_ptr(),
    nullptr,    // unused, was: randval_buf->data_ptr(),
    args.DQ->data_ptr(),
    args.DK->data_ptr(),
    args.DV->data_ptr(),
    nullptr,    // dbias_buf->data_ptr(),
    dq_acc_buf->data_ptr(),
    nullptr,    // was for varlen
    nullptr,    // was for varlen
    nullptr,
    args.max_seqlen_q,
    args.max_seqlen_k,
    batch,
    args.max_seqlen_q,
    args.max_seqlen_k,
    hdim_q,
    hdim_v,
    nhead,
    nhead_k,
    scale,
    stride_q,
    stride_k,
    stride_v,
    0,            // stride_b
    stride_o,
    0,            // stride_randval,
    stride_do,
    stride_dq_acc,
    stride_dq,     // stride_dq
    stride_dk,
    stride_dv,
    stride_dbias,
    nhead_stride_q,
    nhead_stride_k,
    nhead_stride_v,
    nhead_stride_bias,
    nhead_stride_o,
    0,            // nhead_stride_randval,
    nhead_stride_do,
    nhead_stride_lsed,
    nhead_stride_dq_acc,
    nhead_stride_dq,  // nhead_stride_dq
    nhead_stride_dk,  // nhead_stride_dk
    nhead_stride_dv,  // nhead_stride_dv
    0,                // nhead_stride_dbias,
    batch_stride_q,
    batch_stride_k,
    batch_stride_v,
    0,                    // batch_stride_bias,
    batch_stride_o,
    0,                    // batch_stride_randval,
    batch_stride_do,
    batch_stride_lsed,
    batch_stride_dq_acc,  // batch_stride_dq_acc
    batch_stride_dq,      // batch_stride_dq
    batch_stride_dk,
    batch_stride_dv,
    0,                    // batch_stride_dbias,
    0,                    // split_stride_dq_acc, but unused in AITER ASM
    args.Window_left,
    args.Window_right,
    ctx.residual_args.MaskType,
    p_drop,
    p_undrop,
    drop_seed_offset};
};

}

void BwdDqDkDvV3Context::pp_direct_kernel_args_for_fmha_bwd_v3_args(DirectKernelArguments& union_of_args) const {
  auto& args = &union_of_args.fmha_bwd_v3_args;
  auto a = construct_fmha_bwd_args(*this);
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
}


void BwdDqDkDvV3Context::pp_direct_kernel_args_for_fmha_bwd_v3_gen_args(DirectKernelArguments& union_of_args) const {
  auto& args = &union_of_args.fmha_bwd_v3_gen_args;
  auto a = construct_fmha_bwd_args(*this);
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
}


void BwdDqDkDvV3Context::pp_direct_kernel_args_for_fmha_bwd_v3_genl_args(DirectKernelArguments& union_of_args) const {
  auto& args = &union_of_args.fmha_bwd_v3_genl_args;
  auto a = construct_fmha_bwd_args(*this);
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
}


void BwdDqDkDvV3Context::pp_direct_kernel_args_for_fmha_bwd_v3_group_args(DirectKernelArguments& union_of_args) const {
  auto& args = &union_of_args.fmha_bwd_v3_group_args;
  auto a = construct_fmha_bwd_args(*this);
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
}

#if 0 // TODO auto generic_mask = ck_tile::make_generic_attention_mask_coordinates_from_lr_window
void BwdDqDkDvV3Context::pp_direct_kernel_args_for_fmha_bwd_v3_swa_genl_args(DirectKernelArguments& union_of_args) const {
  auto& args = &union_of_args.fmha_bwd_v3_swa_genl_args;
  auto a = construct_fmha_bwd_args(*this);
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
}
#endif

dim3 BwdDqDkDvV3Context::grid_calculator() const {
  return {0, 0, 0};
}

hipError_t AOTRITON_API
aiter_bwd(const attn_bwd_params& in,
          int32_t params_version,
          AOTRITON_NS::Stream stream_wrap,
          const attn_options* options) {
  if (params_version != attn_bwd_params::kVersion) {
    return hipErrorInvalidSymbol; // params_version mismatch
  }
  if (!in.DQ_ACC) {
    return hipErrorInvalidMemcpyDirection;
  }
  hipError_t err;
  auto stream = stream_wrap.native();
  auto gpu = getGpuFromStream(stream);
  int batch = in.Q.size(0);
  int head_dim = in.Q.size(3);
  int num_head_q = in.Q.size(1);
  int num_head_k = in.K.size(1);
  int max_seqlen_q = in.Q.size(2);
  int max_seqlen_k = in.K.size(2);
  int num_seqlens = 0;
  if (in.cu_seqlens_q) {
    // Compact varlen, num_seqlens > 0
    num_seqlens = in.cu_seqlens_q.size(0) - 1;
    max_seqlen_q = in.Max_seqlen_q;
  }
  if (in.cu_seqlens_k) {
    max_seqlen_k = in.Max_seqlen_k;
  }
  static std::vector<int> compiled_head_dims {64, 128, 192};
  // const auto& compiled_head_dims = BwdKernelDkDvMetadata::get_BLOCK_DMODEL_choices();
  int16_t head_dim_rounded = round_value(head_dim, compiled_head_dims);
  OpAttnBwdParams params = {
    .Q = &in.Q,
    .K = &in.K,
    .V = &in.V,
    .B = &in.B,
    .sm_scale = in.Sm_scale,
    .Out = &in.Out,
    .DO = &in.DO,
    .DK = &in.DK,
    .DV = &in.DV,
    .DQ = &in.DQ,
    .DB = &in.DB,
    .DQ_ACC = &in.DQ_ACC,
    .L = &in.L,
    .D = &in.D,
    .num_head_q = num_head_q,
    .num_head_k = num_head_k,
    .cu_seqlens_q = &in.cu_seqlens_q,
    .cu_seqlens_k = &in.cu_seqlens_k,
    .num_seqlens = num_seqlens,
    .max_seqlen_q = max_seqlen_q,
    .max_seqlen_k = max_seqlen_k,
    .head_dim = head_dim,
    .dropout_p = in.dropout_p,
    .philox_seed_ptr  = &in.philox_seed_ptr,
    .philox_offset1   = &in.philox_offset1,
    .philox_offset2   = in.philox_offset2,
    .Window_left = in.window_left,
    .Window_right = in.window_left,
    .BLOCK_DMODEL = head_dim_rounded,
    .CAUSAL_TYPE = in.causal_type,
    .ENABLE_DROPOUT = in.dropout_p > 0.0,
    .PADDED_HEAD = head_dim != head_dim_rounded,
    .BIAS_TYPE = int8_t(bool(in.B) ? 1 : 0),
  };
  BwdDqDkDvV3Context context;
  context.params = &params;
  err = context.lookup_optimal(gpu);
  if (err != hipSuccess) {
    return err;
  }
  return context.launch(stream);
}

}
