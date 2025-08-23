// Copyright © 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/config.h>
#include <aotriton/_internal/util.h>
#include <aotriton/flash.h>
#include <aotriton/util.h>
#include <aotriton/_internal/lazy_tensor_internal.h>
#include <flash/iface.op_attn_bwd.h>
#include <flash/affine.bwd_dq_dk_dv_v3.h>
#include <flash/shim.bwd_preprocess.h>
#include <flash/shim.bwd_preprocess_varlen.h>
#include <flash/shim.bwd_postprocess.h>
#include <algorithm>
#include <limits>
#ifndef NDEBUG
#include <iostream>
#include <stdio.h>
#endif

#define STRINGIFICATION(s) STRINGIFICATION_I(s)
#define STRINGIFICATION_I(s) #s

namespace AOTRITON_NS::v2::flash {

extern hipError_t
bwd_preprocess(T4 out, T4 dout, T2 delta, AOTRITON_NS::Stream stream_wrap);

extern hipError_t
bwd_preprocess_varlen(T4 out,
                      T4 dout,
                      T2 delta,
                      T1 cu_seqlens_q,
                      int32_t max_seqlen_q,
                      AOTRITON_NS::Stream stream_wrap);

}

namespace AOTRITON_NS::v3::flash {

// AITER/CK Compatitility code
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
// End of AITER/CK Compatitility code

const char* BwdDqDkDvV3Context::check_inputs_are_supported() {
  const auto& args = *params;
#define RETURN_IF(COND)                                               \
  do {                                                                \
    if (COND) {                                                       \
      return "Input unsupported due to " STRINGIFICATION(COND);       \
    }                                                                 \
  } while(0)
  // No bias support
  RETURN_IF(args.BIAS_TYPE);
  // No Varlen support
  RETURN_IF(args.cu_seqlens_q && *args.cu_seqlens_q);
  RETURN_IF(args.cu_seqlens_k && *args.cu_seqlens_k);
  // Only support hdim <= 192
  RETURN_IF(args.head_dim > 192);
  // TODO: support dropout kernel. fwd and bwd should have identical PRNG
  RETURN_IF(args.ENABLE_DROPOUT);
  RETURN_IF(args.num_head_q != args.num_head_k);
  RETURN_IF(!args.DQ_ACC);
#undef RETURN_IF
  // We do not have test suite to validate SWA at the moment.
  if (args.CAUSAL_TYPE != CausalType::None) {
      if (args.Window_left != WindowValue::TopLeftAligned ||
          args.Window_right != WindowValue::TopLeftAligned) {
#ifndef NDEBUG
        std::cerr << "Input unsupported due to args.CAUSAL_TYPE = " << int(args.CAUSAL_TYPE) << " and "
                  << " args.Window_left = " << args.Window_left
                  << " args.Window_right = " << args.Window_right
                  << std::endl;
#endif
        return "Input unsupported due to SWA";
      }
  }
  // AITER ASM kernel only reads u32 strides.
#define CHECK_STRIDE(T)                                               \
  do {                                                                \
    auto strides = T->strides();                                      \
    size_t max_e = *std::max_element(strides.begin(), strides.end()); \
    if (max_e * 2 > std::numeric_limits<uint32_t>::max()) {           \
      return "Input unsupported due to large tensor " STRINGIFICATION(T);           \
    }                                                                 \
  } while(0)
#if 0
      std::cerr << "Input unsupported due to large tensor " << #T << std::endl;
      std::cerr << "strides: "; for (auto s : strides) std::cerr << s << " "; std::cerr << std::endl;
      std::cerr << "max_e * 2: " << max_e * 2 << std::endl;
#endif
  CHECK_STRIDE(args.Q);
  CHECK_STRIDE(args.K);
  CHECK_STRIDE(args.V);
  CHECK_STRIDE(args.Out);
  CHECK_STRIDE(args.DO);
  CHECK_STRIDE(args.DK);
  CHECK_STRIDE(args.DV);
  CHECK_STRIDE(args.DB);
#undef CHECK_STRIDE

  return nullptr;
}

void BwdDqDkDvV3Context::calculate_residual_func_fields() {
    const auto& args = *params;
    auto check_if_uniform = [&]() -> bool {
        // Reject varlen
        if (args.cu_seqlens_q && *args.cu_seqlens_q) return false;
        if (args.cu_seqlens_k && *args.cu_seqlens_k) return false;
        // TODO: GQA support
        if (args.num_head_q != args.num_head_k) return false;
#ifdef NDEBUG
#define CMP_TENSOR(X, Y)                                                \
        do {                                                            \
            if (args.X->strides() != args.Y->strides()) return false;   \
        } while(0)
#else
#define CMP_TENSOR(X, Y)                                                \
        do {                                                            \
            std::cerr << #X << " strides: ";                            \
            for (auto e : args.X->strides())                            \
              std::cerr << e << " ";                                    \
            std::cerr << std::endl;                                     \
            std::cerr << #Y << " strides: ";                            \
            for (auto e : args.Y->strides())                            \
              std::cerr << e << " ";                                    \
            std::cerr << std::endl;                                     \
            if (args.X->strides() != args.Y->strides()) return false;   \
        } while(0)
#endif
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
    residual_args.kIsUniformStride = check_if_uniform();
    residual_args.kIsSEQPad = (args.max_seqlen_q % 64 != 0);
    // residual_args.kIsHDPad = !check_hdim_regular();
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

// Too many narrowing warning here.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"

aiter::fmha_bwd_args
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
  auto stride_o = args.Out->stride(2);
  auto stride_do = args.DO->stride(2);
  auto stride_dq_acc = args.DQ_ACC->stride(2);
  auto stride_dq = args.DQ->stride(2);
  auto stride_dk = args.DK->stride(2);
  auto stride_dv = args.DV->stride(2);

  auto nhead_stride_q = args.Q->stride(1);
  auto nhead_stride_k = args.K->stride(1);
  auto nhead_stride_v = args.V->stride(1);
  auto nhead_stride_o = args.Out->stride(1);
  auto nhead_stride_do = args.DO->stride(1);
  auto nhead_stride_dq_acc = args.DQ_ACC->stride(1);
  auto nhead_stride_dq = args.DQ->stride(1);
  auto nhead_stride_dk = args.DK->stride(1);
  auto nhead_stride_dv = args.DV->stride(1);
  auto nhead_stride_lsed = args.L->stride(1);

  auto batch_stride_q = args.Q->stride(0);
  auto batch_stride_k = args.K->stride(0);
  auto batch_stride_v = args.V->stride(0);
  auto batch_stride_o = args.Out->stride(0);
  auto batch_stride_do = args.DO->stride(0);
  auto batch_stride_dq_acc = args.DQ_ACC->stride(0);
  auto batch_stride_dq = args.DQ->stride(0);
  auto batch_stride_dk = args.DK->stride(0);
  auto batch_stride_dv = args.DV->stride(0);
  auto batch_stride_lsed = args.L->stride(0);
  float p_drop = 0.0;
  float p_undrop = 0.0;
  auto drop_seed_offset = std::make_pair<uint64_t, uint64_t>(0, 0); // placeholder
  aiter::fmha_bwd_args ret = {
    .q_ptr = args.Q->data_ptr(),
    .k_ptr = args.K->data_ptr(),
    .v_ptr = args.V->data_ptr(),
    .bias_ptr = nullptr,    // unused, was handling bias
    .o_ptr = args.Out->data_ptr(),
    .lse_ptr = args.L->data_ptr(),
    .do_ptr = args.DO->data_ptr(),
    .d_ptr = args.D->data_ptr(),
    .rand_val_ptr = nullptr,    // unused, was: randval_buf->data_ptr(),
    .dq_ptr = args.DQ->data_ptr(),
    .dk_ptr = args.DK->data_ptr(),
    .dv_ptr = args.DV->data_ptr(),
    .dbias_ptr = nullptr,    // dbias_buf->data_ptr(),
    .dq_acc_ptr = args.DQ_ACC->data_ptr(),
    .seqstart_q_ptr = args.cu_seqlens_q->data_ptr(),    // was for varlen/group (why needed for group?)
    .seqstart_k_ptr = args.cu_seqlens_k->data_ptr(),    // was for varlen/group
    .seqlen_k_ptr = nullptr,
    .seqlen_q = args.max_seqlen_q,
    .seqlen_k = args.max_seqlen_k,
    .batch = batch,
    .max_seqlen_q = args.max_seqlen_q,
    .max_seqlen_k = args.max_seqlen_k,
    .hdim_q = hdim_q,
    .hdim_v = hdim_v,
    .nhead_q = nhead,
    .nhead_k = nhead_k,
    .scale = scale,
    .stride_q = stride_q,
    .stride_k = stride_k,
    .stride_v = stride_v,
    .stride_bias = 0,            // if alibi, b*h need set this to h, 1*h need set this to 0
    .stride_o = stride_o,
    .stride_randval = 0,            // stride_randval,
    .stride_do = stride_do,
    .stride_dq_acc = stride_dq_acc,
    .stride_dq = stride_dq,     // stride_dq
    .stride_dk = stride_dk,
    .stride_dv = stride_dv,
    .stride_dbias = 0,            // stride_dbias,
    .nhead_stride_q = nhead_stride_q,
    .nhead_stride_k = nhead_stride_k,
    .nhead_stride_v = nhead_stride_v,
    .nhead_stride_bias = 0,            // nhead_stride_bias,
    .nhead_stride_o = nhead_stride_o,
    .nhead_stride_randval = 0,            // nhead_stride_randval,
    .nhead_stride_do = nhead_stride_do,
    .nhead_stride_lsed = nhead_stride_lsed,
    .nhead_stride_dq_acc = nhead_stride_dq_acc,
    .nhead_stride_dq = nhead_stride_dq,  // nhead_stride_dq
    .nhead_stride_dk = nhead_stride_dk,  // nhead_stride_dk
    .nhead_stride_dv = nhead_stride_dv,  // nhead_stride_dv
    .nhead_stride_dbias = 0,                // nhead_stride_dbias,
    .batch_stride_q = batch_stride_q,
    .batch_stride_k = batch_stride_k,
    .batch_stride_v = batch_stride_v,
    .batch_stride_bias = 0,                    // batch_stride_bias,
    .batch_stride_o = batch_stride_o,
    .batch_stride_randval = 0,                    // batch_stride_randval,
    .batch_stride_do = batch_stride_do,
    .batch_stride_lsed = batch_stride_lsed,
    .batch_stride_dq_acc = batch_stride_dq_acc,  // batch_stride_dq_acc
    .batch_stride_dq = batch_stride_dq,      // batch_stride_dq
    .batch_stride_dk = batch_stride_dk,
    .batch_stride_dv = batch_stride_dv,
    .batch_stride_dbias = 0,                    // batch_stride_dbias,
    .split_stride_dq_acc = 0,                    // split_stride_dq_acc, but unused in AITER ASM
    .window_size_left = args.Window_left,
    .window_size_right = args.Window_right,
    .mask_type = ctx.residual_args.MaskType,
    .p_drop = p_drop,
    .p_undrop = p_undrop,
    .drop_seed_offset = drop_seed_offset,
  };
  return ret;
}

using DirectKernelArguments = BwdDqDkDvV3Context::DirectKernelArguments;

/*
 * s/\<fmha_v3_traits\>/traits/g
 * %s/FmhaBwdV3Ts<dq_dk_dv_v3_traits_>::/perf_args./g
 */
std::tuple<dim3, dim3>
BwdDqDkDvV3Context::pp_direct_kernel_args_for_fmha_bwd_v3_args(DirectKernelArguments& union_of_args) const {
#ifndef NDEBUG
  std::cerr << "Calling " << __func__ << std::endl;
#endif
  auto& args = union_of_args.fmha_bwd_v3_args;
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

  args.Ts   = perf_args.ts_kv * a.stride_k * 2;
  args.Hs   = a.nhead_stride_q * 2;
  args.BAs  = a.batch_stride_q * 2;
  args.Seqs = a.stride_q * 2;

  args.ratio    = a.nhead_q / a.nhead_k;
  args.Hs_kv    = a.nhead_stride_k * 2;
  args.BAs_kv   = a.batch_stride_k * 2;
  args.Seqs_kv  = a.stride_k * 2;
  args.Seqs_dkv = a.stride_dk * 2;

  using aiter::fmha_bwd_v3_traits;
  auto traits = fmha_bwd_v3_traits {a.batch,
                                    a.nhead_q,
                                    a.seqlen_q,
                                    a.hdim_q,
                                    a.mask_type,
                                    perf_args.ts_qo,
                                    perf_args.ts_kv};
  int gdx = (traits.s + traits.ts_kv - 1) / traits.ts_kv;
  int gdy = traits.h;
  int gdz = traits.b;
  if (residual_args.MaskType > 0) {
    int num_tg = (traits.s + traits.ts_kv - 1) / traits.ts_kv;
    gdx        = (num_tg % 2) ? (num_tg / 2 + 1) : (num_tg / 2);
  }
  dim3 grid {gdx, gdy, gdz};
  dim3 block { 256, 1, 1 };
  return std::make_tuple(grid, block);
}


std::tuple<dim3, dim3>
BwdDqDkDvV3Context::pp_direct_kernel_args_for_fmha_bwd_v3_gen_args(DirectKernelArguments& union_of_args) const {
#ifndef NDEBUG
  std::cerr << "Calling " << __func__ << std::endl;
#endif
  auto& args = union_of_args.fmha_bwd_v3_gen_args;
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

  args.Ts   = perf_args.ts_kv * a.stride_k * 2;
  args.Hs   = a.nhead_stride_q * 2;
  args.BAs  = a.batch_stride_q * 2;
  args.Seqs = a.stride_q * 2;

  args.ratio    = a.nhead_q / a.nhead_k;
  args.Hs_kv    = a.nhead_stride_k * 2;
  args.BAs_kv   = a.batch_stride_k * 2;
  args.Seqs_kv  = a.stride_k * 2;
  args.Seqs_dkv = a.stride_dk * 2;
  args.head_dim = a.hdim_q;

  using aiter::fmha_bwd_v3_traits;
  auto traits = fmha_bwd_v3_traits {a.batch,
                                    a.nhead_q,
                                    a.seqlen_q,
                                    a.hdim_q,
                                    a.mask_type,
                                    perf_args.ts_qo,
                                    perf_args.ts_kv};
  int gdx = (traits.s + traits.ts_kv - 1) / traits.ts_kv;
  int gdy = traits.h;
  int gdz = traits.b;
  if (residual_args.MaskType > 0) {
    int num_tg = (traits.s + traits.ts_kv - 1) / traits.ts_kv;
    gdx        = (num_tg % 2) ? (num_tg / 2 + 1) : (num_tg / 2);
  }
  dim3 grid {gdx, gdy, gdz};
  dim3 block { 256, 1, 1 };
  return std::make_tuple(grid, block);
}


std::tuple<dim3, dim3>
BwdDqDkDvV3Context::pp_direct_kernel_args_for_fmha_bwd_v3_genl_args(DirectKernelArguments& union_of_args) const {
#ifndef NDEBUG
  std::cerr << "Calling " << __func__ << std::endl;
#endif
  auto& args = union_of_args.fmha_bwd_v3_genl_args;
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

  using aiter::fmha_bwd_v3_traits;
  auto traits = fmha_bwd_v3_traits {a.batch,
                                    a.nhead_q,
                                    a.seqlen_k,
                                    a.hdim_q,
                                    a.mask_type,
                                    perf_args.ts_qo,
                                    perf_args.ts_kv};
  int gdx = (traits.s + traits.ts_kv - 1) / traits.ts_kv;
  int gdy = traits.h;
  int gdz = traits.b;
  if (residual_args.MaskType > 0) {
    int num_tg = (traits.s + traits.ts_kv - 1) / traits.ts_kv;
    gdx        = (num_tg % 2) ? (num_tg / 2 + 1) : (num_tg / 2);
  }
  dim3 grid {gdx, gdy, gdz};
  dim3 block { 256, 1, 1 };
#ifndef NDEBUG
  std::cerr << "Inside " << __func__ << std::endl
            << "args.ptr_dq = " << args.ptr_dq << std::endl
            << "args.ptr_dk = " << args.ptr_dk << std::endl
            << "args.ptr_dv = " << args.ptr_dv << std::endl
            << "args.ptr_q = " << args.ptr_q << std::endl
            << "args.ptr_k = " << args.ptr_k << std::endl
            << "args.ptr_v = " << args.ptr_v << std::endl
            << std::endl;
  auto hexdump = [](void *ptr, int buflen) {
    unsigned char *buf = (unsigned char*)ptr;
    int i, j;
    fprintf(stderr, "hexdump: %08p\n", buf);
    for (i=0; i<buflen; i+=16) {
      fprintf(stderr, "%06x: ", i);
      for (j=0; j<16; j++)
        if (i+j < buflen)
          fprintf(stderr, "%02x ", buf[i+j]);
        else
          fprintf(stderr, "   ");
      fprintf(stderr, " ");
      for (j=0; j<16; j++)
        if (i+j < buflen)
          fprintf(stderr, "%c", isprint(buf[i+j]) ? buf[i+j] : '.');
      fprintf(stderr, "\n");
    }
  };
  hexdump(&args, sizeof(args));
  fprintf(stderr, "Union %p\n", &union_of_args);
  fprintf(stderr, "Union %p\n", &union_of_args.fmha_bwd_v3_args);
  fprintf(stderr, "Union %p\n", &union_of_args.fmha_bwd_v3_gen_args);
  fprintf(stderr, "Union %p\n", &union_of_args.fmha_bwd_v3_genl_args);
#endif
  return std::make_tuple(grid, block);
}


std::tuple<dim3, dim3>
BwdDqDkDvV3Context::pp_direct_kernel_args_for_fmha_bwd_v3_group_args(DirectKernelArguments& union_of_args) const {
  auto& args = union_of_args.fmha_bwd_v3_group_args;
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
  // args.seqlen_q = seqstart_q[a.batch];
  // args.seqlen_k = seqstart_k[a.batch];
  args.seqlen_q = a.max_seqlen_q;
  args.seqlen_k = a.max_seqlen_k;
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

  using aiter::fmha_bwd_v3_traits;
  auto traits = fmha_bwd_v3_traits {a.batch,
                                    a.nhead_q,
                                    a.max_seqlen_k,
                                    a.hdim_q,
                                    a.mask_type,
                                    perf_args.ts_qo,
                                    perf_args.ts_kv };
  int gdx = (traits.s + traits.ts_kv - 1) / traits.ts_kv;
  if (traits.mask > 0) {
    gdx = (gdx % 2) ? (gdx / 2 + 1) : (gdx / 2);
  }
  int gdy = traits.h;
  int gdz = traits.b;

  dim3 grid {gdx, gdy, gdz};
  dim3 block { 256, 1, 1 };
  return std::make_tuple(grid, block);
}

std::tuple<dim3, dim3>
BwdDqDkDvV3Context::pp_direct_kernel_args_for_fmha_bwd_v3_swa_genl_args(DirectKernelArguments& union_of_args) const {
  // As stated above, we do not have proper test suite to validate SWA support ATM
  dim3 grid { 0, 0, 0};
  dim3 block { 0, 1, 1 };
  return std::make_tuple(grid, block);
  // Leave the code for further development.
#if 0 // TODO auto generic_mask = ck_tile::make_generic_attention_mask_coordinates_from_lr_window
  auto& args = union_of_args.fmha_bwd_v3_swa_genl_args;
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

  using aiter::fmha_bwd_v3_traits;
  auto traits = fmha_bwd_v3_traits{a.batch,
                                    a.nhead_q,
                                    a.seqlen_k,
                                    a.hdim_q,
                                    a.mask_type,
                                    perf_args.ts_qo,
                                    perf_args.ts_kv};
  int gdx = (traits.s + traits.ts_kv - 1) / traits.ts_kv;
  int gdy = traits.h;
  int gdz = traits.b;
  dim3 grid {gdx, gdy, gdz};
  dim3 block { 256, 1, 1 };
  return std::make_tuple(grid, block);
#endif
}

#pragma GCC diagnostic pop

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
  LazyTensorInternal<2> lazy_delta(in.D);
  LazyTensorInternal<4> lazy_dq_acc(in.DQ_ACC);
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
    .DQ_ACC = &lazy_dq_acc,
    .L = &in.L,
    .D = &lazy_delta,
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
  // Invoke context.lookup_optimal to confirm the input works
  // TODO: this API should call Metro Kernel instead
  // TODO: Metro kernel should call lookup_optimal for all context before invoking anything
  BwdDqDkDvV3Context context;
  context.params = &params;
  err = context.lookup_optimal(gpu);
  if (err != hipSuccess) {
    return err;
  }
  using AOTRITON_NS::v2::flash::bwd_preprocess;
  using AOTRITON_NS::v2::flash::bwd_preprocess_varlen;
  if (num_seqlens == 0)
    err = bwd_preprocess(in.Out, in.DO, lazy_delta.make_concrete(), stream);
  else
    err = bwd_preprocess_varlen(in.Out, in.DO, lazy_delta.make_concrete(),
                                in.cu_seqlens_q, max_seqlen_q, stream);
  if (err != hipSuccess)
    return err;
  err = context.launch(stream);
  if (err != hipSuccess)
    return err;

  {
    BwdPostprocessContext context;
    context.params = &params;
    err = context.lookup_optimal(gpu);
    if (err != hipSuccess)
      return err;
    err = context.launch(stream);
    if (err != hipSuccess)
      return err;
  }
  return err;
}

dim3 BwdPostprocessContext::grid_calculator() const {
  dim3 grid {
    AOTRITON_NS::cdiv<uint32_t>(params->DQ->size(2), this->BLOCK_M),
    uint32_t(params->Out->size(1)),
    uint32_t(params->Out->size(0)),
  };
  return grid;
}

}
