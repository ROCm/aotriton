// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_API_FLASH_ATTN_AITER_H
#define AOTRITON_V2_API_FLASH_ATTN_AITER_H

#include <aotriton/config.h>
#include <aotriton/_internal/aiter_hip_common.h>
#include <variant>

namespace AOTRITON_NS::v3::flash::aiter {

using namespace AOTRITON_NS::v3::aiter;

struct __attribute__((packed)) fmha_bwd_v3_args
{
    void* ptr_dq;
    p2 _p0;
    void* ptr_dk;
    p2 _p1;
    void* ptr_dv;
    p2 _p2;
    const void* ptr_q;
    p2 _p3;
    const void* ptr_k;
    p2 _p4;
    const void* ptr_v;
    p2 _p5;
    const void* ptr_do;
    p2 _p6;
    const void* ptr_lse;
    p2 _p7;
    const void* ptr_d;
    p2 _p8;
    float scalar;
    p3 _p9;
    float log2e;
    p3 _p10;
    unsigned int seq_len;
    p3 _p11;
    unsigned int Ts;
    p3 _p12;
    unsigned int Hs;
    p3 _p13;
    unsigned int BAs;
    p3 _p14;
    unsigned int Seqs;
    p3 _p15;
    unsigned int ratio;
    p3 _p16;
    unsigned int Hs_kv;
    p3 _p17;
    unsigned int BAs_kv;
    p3 _p18;
    unsigned int Seqs_kv;
    p3 _p19;
    unsigned int Seqs_dkv;
    p3 _p20;
};

struct __attribute__((packed)) fmha_bwd_v3_gen_args
{
    void* ptr_dq;
    p2 _p0;
    void* ptr_dk;
    p2 _p1;
    void* ptr_dv;
    p2 _p2;
    const void* ptr_q;
    p2 _p3;
    const void* ptr_k;
    p2 _p4;
    const void* ptr_v;
    p2 _p5;
    const void* ptr_do;
    p2 _p6;
    const void* ptr_lse;
    p2 _p7;
    const void* ptr_d;
    p2 _p8;
    float scalar;
    p3 _p9;
    float log2e;
    p3 _p10;
    unsigned int seq_len;
    p3 _p11;
    unsigned int Ts;
    p3 _p12;
    unsigned int Hs;
    p3 _p13;
    unsigned int BAs;
    p3 _p14;
    unsigned int Seqs;
    p3 _p15;
    unsigned int ratio;
    p3 _p16;
    unsigned int Hs_kv;
    p3 _p17;
    unsigned int BAs_kv;
    p3 _p18;
    unsigned int Seqs_kv;
    p3 _p19;
    unsigned int Seqs_dkv;
    p3 _p20;
    unsigned int head_dim;
    p3 _p21;
};

struct __attribute__((packed)) fmha_bwd_v3_genl_args
{
    void* ptr_dq;
    void* ptr_dk;
    void* ptr_dv;
    const void* ptr_q;
    const void* ptr_k;
    const void* ptr_v;
    const void* ptr_do;
    const void* ptr_lse;
    const void* ptr_d;
    float scalar;
    p1 _p0;
    float log2e;
    p1 _p1;
    unsigned int ratio;
    p1 _p2;
    unsigned int seqlen_q;
    p1 _p3;
    unsigned int seqlen_k;
    p1 _p4;
    unsigned int head_dim;
    p1 _p5;
    unsigned int nhead_q;
    p1 _p6;
    unsigned int Hs_q;
    p1 _p7;
    unsigned int BAs_q;
    p1 _p8;
    unsigned int Seqs_q;
    p1 _p9;
    unsigned int Hs_k;
    p1 _p10;
    unsigned int BAs_k;
    p1 _p11;
    unsigned int Seqs_k;
    p1 _p12;
    unsigned int Hs_v;
    p1 _p13;
    unsigned int BAs_v;
    p1 _p14;
    unsigned int Seqs_v;
    p1 _p15;
    unsigned int Hs_do;
    p1 _p16;
    unsigned int BAs_do;
    p1 _p17;
    unsigned int Seqs_do;
    p1 _p18;
    unsigned int Hs_dk;
    p1 _p19;
    unsigned int BAs_dk;
    p1 _p20;
    unsigned int Seqs_dk;
    p1 _p21;
    unsigned int Hs_dv;
    p1 _p22;
    unsigned int BAs_dv;
    p1 _p23;
    unsigned int Seqs_dv;
    p1 _p24;
};

struct __attribute__((packed)) fmha_bwd_v3_group_args
{
    void* ptr_dq;
    void* ptr_dk;
    void* ptr_dv;
    const void* ptr_q;
    const void* ptr_k;
    const void* ptr_v;
    const void* ptr_do;
    const void* ptr_lse;
    const void* ptr_d;
    const void* ptr_qseq;
    const void* ptr_kseq;
    float scalar;
    p1 _p0;
    float log2e;
    p1 _p1;
    unsigned int ratio;
    p1 _p2;
    unsigned int seqlen_q; //total length of q sequences
    p1 _p3;
    unsigned int seqlen_k; //total length of k sequences
    p1 _p4;
    unsigned int Hs_q;
    p1 _p5;
    unsigned int Seqs_q;
    p1 _p6;
    unsigned int Hs_k;
    p1 _p7;
    unsigned int Seqs_k;
    p1 _p8;
    unsigned int Hs_v;
    p1 _p9;
    unsigned int Seqs_v;
    p1 _p10;
    unsigned int Hs_do;
    p1 _p11;
    unsigned int Seqs_do;
    p1 _p12;
    unsigned int Hs_dk;
    p1 _p13;
    unsigned int Seqs_dk;
    p1 _p14;
    unsigned int Hs_dv;
    p1 _p15;
    unsigned int Seqs_dv;
    p1 _p16;
    unsigned int head_dim;
    p1 _p17;
};

struct __attribute__((packed)) fmha_bwd_v3_swa_genl_args
{
    void* ptr_dq;
    void* ptr_dk;
    void* ptr_dv;
    const void* ptr_q;
    const void* ptr_k;
    const void* ptr_v;
    const void* ptr_do;
    const void* ptr_lse;
    const void* ptr_d;
    float scalar;
    p1 _p0;
    float log2e;
    p1 _p1;
    unsigned int ratio;
    p1 _p2;
    unsigned int seqlen_q;
    p1 _p3;
    unsigned int seqlen_k;
    p1 _p4;
    unsigned int head_dim;
    p1 _p5;
    unsigned int nhead_q;
    p1 _p6;
    unsigned int Hs_q;
    p1 _p7;
    unsigned int BAs_q;
    p1 _p8;
    unsigned int Seqs_q;
    p1 _p9;
    unsigned int Hs_k;
    p1 _p10;
    unsigned int BAs_k;
    p1 _p11;
    unsigned int Seqs_k;
    p1 _p12;
    unsigned int Hs_v;
    p1 _p13;
    unsigned int BAs_v;
    p1 _p14;
    unsigned int Seqs_v;
    p1 _p15;
    unsigned int Hs_do;
    p1 _p16;
    unsigned int BAs_do;
    p1 _p17;
    unsigned int Seqs_do;
    p1 _p18;
    unsigned int Hs_dk;
    p1 _p19;
    unsigned int BAs_dk;
    p1 _p20;
    unsigned int Seqs_dk;
    p1 _p21;
    unsigned int Hs_dv;
    p1 _p22;
    unsigned int BAs_dv;
    p1 _p23;
    unsigned int Seqs_dv;
    p1 _p24;
    int mask_x;
    p1 _p25;
    int mask_y;
    p1 _p26;
};

struct __attribute__((packed)) fmha_bwd_dq_shuffle_args
{
    void *ptr_dq;
    p2 _p0;
    unsigned int Ts;
    p3 _p1;
    unsigned int Hs;
    p3 _p2;
    unsigned int BAs;
    p3 _p3;
    unsigned int Seqs;
    p3 _p4;
};

struct fmha_bwd_v3_traits
{
    int b;
    int h;
    int s;
    int d;

    int mask;
    int ts_qo;
    int ts_kv;
    int ts_dq = 64;
};

namespace ck_tile {
  using index_t = int32_t;
}

struct fmha_bwd_args
{
    const void* q_ptr;
    const void* k_ptr;
    const void* v_ptr;
    const void* bias_ptr; // bias or alibi_slope pointer
    const void* o_ptr;
    const void* lse_ptr;
    const void* do_ptr;
    void* d_ptr;
    void* rand_val_ptr;
    void* dq_ptr;
    void* dk_ptr;
    void* dv_ptr;
    void* dbias_ptr;
    void* dq_acc_ptr;
    const void* seqstart_q_ptr;
    const void* seqstart_k_ptr;
    const void* seqlen_k_ptr;
    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t batch;
    ck_tile::index_t max_seqlen_q;
    ck_tile::index_t max_seqlen_k;
    ck_tile::index_t hdim_q;
    ck_tile::index_t hdim_v;
    ck_tile::index_t nhead_q;
    ck_tile::index_t nhead_k;
    float scale;
    ck_tile::index_t stride_q;
    ck_tile::index_t stride_k;
    ck_tile::index_t stride_v;
    ck_tile::index_t stride_bias; // if alibi, b*h need set this to h, 1*h need set this to 0
    ck_tile::index_t stride_o;
    ck_tile::index_t stride_randval;
    ck_tile::index_t stride_do;
    ck_tile::index_t stride_dq_acc;
    ck_tile::index_t stride_dq;
    ck_tile::index_t stride_dk;
    ck_tile::index_t stride_dv;
    ck_tile::index_t stride_dbias;
    ck_tile::index_t nhead_stride_q;
    ck_tile::index_t nhead_stride_k;
    ck_tile::index_t nhead_stride_v;
    ck_tile::index_t nhead_stride_bias;
    ck_tile::index_t nhead_stride_o;
    ck_tile::index_t nhead_stride_randval;
    ck_tile::index_t nhead_stride_do;
    ck_tile::index_t nhead_stride_lsed;
    ck_tile::index_t nhead_stride_dq_acc;
    ck_tile::index_t nhead_stride_dq;
    ck_tile::index_t nhead_stride_dk;
    ck_tile::index_t nhead_stride_dv;
    ck_tile::index_t nhead_stride_dbias;
    ck_tile::index_t batch_stride_q;
    ck_tile::index_t batch_stride_k;
    ck_tile::index_t batch_stride_v;
    ck_tile::index_t batch_stride_bias;
    ck_tile::index_t batch_stride_o;
    ck_tile::index_t batch_stride_randval;
    ck_tile::index_t batch_stride_do;
    ck_tile::index_t batch_stride_lsed;
    ck_tile::index_t batch_stride_dq_acc;
    ck_tile::index_t batch_stride_dq;
    ck_tile::index_t batch_stride_dk;
    ck_tile::index_t batch_stride_dv;
    ck_tile::index_t batch_stride_dbias;
    ck_tile::index_t split_stride_dq_acc;
    ck_tile::index_t window_size_left;
    ck_tile::index_t window_size_right;
    ck_tile::index_t mask_type;
    float p_drop;
    float p_undrop;
    std::variant<std::pair<uint64_t, uint64_t>, std::pair<const void*, const void*>>
        drop_seed_offset;
};

} // AOTRITON_NS::v3::flash::aiter

#endif
