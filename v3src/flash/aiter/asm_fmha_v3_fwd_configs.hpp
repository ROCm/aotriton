// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include <unordered_map>


#define ADD_CFG(dtype, hdim_q, hdim_v, mask, mode, bf16_cvt, ts_qo, ts_kv, arch, path, knl_name, co_name)         \
    {                                         \
        arch knl_name, { knl_name, path co_name, arch, dtype, hdim_q, hdim_v, mask, mode, bf16_cvt, ts_qo, ts_kv }         \
    }

namespace AOTRITON_NS::v3::flash::aiter {

struct fmha_v3_fwdConfig
{
    std::string knl_name;
    std::string co_name;
    std::string arch;
    std::string dtype;
    int hdim_q;
    int hdim_v;
    int mask;
    int mode;
    int bf16_cvt;
    int ts_qo;
    int ts_kv;
};

using CFG = std::unordered_map<std::string, fmha_v3_fwdConfig>;

static CFG cfg_fmha_fwd = {
    ADD_CFG("bf16",  128,  128,    0,    0,    0,  256,   32, "gfx942", "fmha_v3_fwd/", "_ZN5aiter24fmha_fwd_hd128_bf16_rtneE", "fwd_hd128_bf16_rtne.co"),
    ADD_CFG("bf16",  128,  128,    0,    0,    1,  256,   32, "gfx942", "fmha_v3_fwd/", "_ZN5aiter24fmha_fwd_hd128_bf16_rtnaE", "fwd_hd128_bf16_rtna.co"),
    ADD_CFG("bf16",  128,  128,    0,    0,    2,  256,   32, "gfx942", "fmha_v3_fwd/", "_ZN5aiter23fmha_fwd_hd128_bf16_rtzE", "fwd_hd128_bf16_rtz.co"),
    ADD_CFG("bf16",  128,  128,    2,    0,    0,  256,   32, "gfx942", "fmha_v3_fwd/", "_ZN5aiter31fmha_fwd_hd128_bf16_causal_rtneE", "fwd_hd128_bf16_causal_rtne.co"),
    ADD_CFG("bf16",  128,  128,    2,    0,    1,  256,   32, "gfx942", "fmha_v3_fwd/", "_ZN5aiter31fmha_fwd_hd128_bf16_causal_rtnaE", "fwd_hd128_bf16_causal_rtna.co"),
    ADD_CFG("bf16",  128,  128,    2,    0,    2,  256,   32, "gfx942", "fmha_v3_fwd/", "_ZN5aiter30fmha_fwd_hd128_bf16_causal_rtzE", "fwd_hd128_bf16_causal_rtz.co"),
    ADD_CFG("bf16",  128,  128,    0,    1,    0,  256,   32, "gfx942", "fmha_v3_fwd/", "_ZN5aiter30fmha_fwd_hd128_bf16_rtne_groupE", "fwd_hd128_bf16_rtne_group.co"),
    ADD_CFG("bf16",  128,  128,    0,    1,    1,  256,   32, "gfx942", "fmha_v3_fwd/", "_ZN5aiter30fmha_fwd_hd128_bf16_rtna_groupE", "fwd_hd128_bf16_rtna_group.co"),
    ADD_CFG("bf16",  128,  128,    0,    1,    2,  256,   32, "gfx942", "fmha_v3_fwd/", "_ZN5aiter29fmha_fwd_hd128_bf16_rtz_groupE", "fwd_hd128_bf16_rtz_group.co"),
    ADD_CFG("bf16",  128,  128,    2,    1,    0,  256,   32, "gfx942", "fmha_v3_fwd/", "_ZN5aiter37fmha_fwd_hd128_bf16_causal_rtne_groupE", "fwd_hd128_bf16_causal_rtne_group.co"),
    ADD_CFG("bf16",  128,  128,    2,    1,    1,  256,   32, "gfx942", "fmha_v3_fwd/", "_ZN5aiter37fmha_fwd_hd128_bf16_causal_rtna_groupE", "fwd_hd128_bf16_causal_rtna_group.co"),
    ADD_CFG("bf16",  128,  128,    2,    1,    2,  256,   32, "gfx942", "fmha_v3_fwd/", "_ZN5aiter36fmha_fwd_hd128_bf16_causal_rtz_groupE", "fwd_hd128_bf16_causal_rtz_group.co"),
    ADD_CFG("bf16",  192,  128,    0,    0,    0,  128,   32, "gfx942", "fmha_v3_fwd/", "_ZN5aiter28fmha_fwd_hd192x128_bf16_rtneE", "fwd_hd192x128_bf16_rtne.co"),
    ADD_CFG("bf16",  192,  128,    0,    0,    1,  128,   32, "gfx942", "fmha_v3_fwd/", "_ZN5aiter28fmha_fwd_hd192x128_bf16_rtnaE", "fwd_hd192x128_bf16_rtna.co"),
    ADD_CFG("bf16",  192,  128,    0,    0,    2,  128,   32, "gfx942", "fmha_v3_fwd/", "_ZN5aiter27fmha_fwd_hd192x128_bf16_rtzE", "fwd_hd192x128_bf16_rtz.co"),
    ADD_CFG("bf16",  192,  128,    2,    0,    0,  128,   32, "gfx942", "fmha_v3_fwd/", "_ZN5aiter35fmha_fwd_hd192x128_bf16_causal_rtneE", "fwd_hd192x128_bf16_causal_rtne.co"),
    ADD_CFG("bf16",  192,  128,    2,    0,    1,  128,   32, "gfx942", "fmha_v3_fwd/", "_ZN5aiter35fmha_fwd_hd192x128_bf16_causal_rtnaE", "fwd_hd192x128_bf16_causal_rtna.co"),
    ADD_CFG("bf16",  192,  128,    2,    0,    2,  128,   32, "gfx942", "fmha_v3_fwd/", "_ZN5aiter34fmha_fwd_hd192x128_bf16_causal_rtzE", "fwd_hd192x128_bf16_causal_rtz.co"),
    ADD_CFG("bf16",  192,  128,    0,    1,    0,  128,   32, "gfx942", "fmha_v3_fwd/", "_ZN5aiter34fmha_fwd_hd192x128_bf16_rtne_groupE", "fwd_hd192x128_bf16_rtne_group.co"),
    ADD_CFG("bf16",  192,  128,    0,    1,    1,  128,   32, "gfx942", "fmha_v3_fwd/", "_ZN5aiter34fmha_fwd_hd192x128_bf16_rtna_groupE", "fwd_hd192x128_bf16_rtna_group.co"),
    ADD_CFG("bf16",  192,  128,    0,    1,    2,  128,   32, "gfx942", "fmha_v3_fwd/", "_ZN5aiter33fmha_fwd_hd192x128_bf16_rtz_groupE", "fwd_hd192x128_bf16_rtz_group.co"),
    ADD_CFG("bf16",  192,  128,    2,    1,    0,  128,   32, "gfx942", "fmha_v3_fwd/", "_ZN5aiter41fmha_fwd_hd192x128_bf16_causal_rtne_groupE", "fwd_hd192x128_bf16_causal_rtne_group.co"),
    ADD_CFG("bf16",  192,  128,    2,    1,    1,  128,   32, "gfx942", "fmha_v3_fwd/", "_ZN5aiter41fmha_fwd_hd192x128_bf16_causal_rtna_groupE", "fwd_hd192x128_bf16_causal_rtna_group.co"),
    ADD_CFG("bf16",  192,  128,    2,    1,    2,  128,   32, "gfx942", "fmha_v3_fwd/", "_ZN5aiter40fmha_fwd_hd192x128_bf16_causal_rtz_groupE", "fwd_hd192x128_bf16_causal_rtz_group.co"),
    ADD_CFG("bf16",  128,  128,    0,    0,    0,  256,   64, "gfx950", "fmha_v3_fwd/", "_ZN5aiter19fmha_fwd_hd128_bf16E", "fwd_hd128_bf16.co"),
    ADD_CFG("bf16",  128,  128,    2,    0,    0,  256,   64, "gfx950", "fmha_v3_fwd/", "_ZN5aiter26fmha_fwd_hd128_bf16_causalE", "fwd_hd128_bf16_causal.co"),
    ADD_CFG("bf16",  128,  128,    0,    1,    0,  256,   64, "gfx950", "fmha_v3_fwd/", "_ZN5aiter25fmha_fwd_hd128_bf16_groupE", "fwd_hd128_bf16_group.co"),
    ADD_CFG("bf16",  128,  128,    2,    1,    0,  256,   64, "gfx950", "fmha_v3_fwd/", "_ZN5aiter32fmha_fwd_hd128_bf16_causal_groupE", "fwd_hd128_bf16_causal_group.co"),
    ADD_CFG("bf16",  192,  128,    0,    0,    0,  128,  128, "gfx950", "fmha_v3_fwd/", "_ZN5aiter25fmha_fwd_hd192_hd128_bf16E", "fwd_hd192_hd128_bf16.co"),
    ADD_CFG("bf16",  192,  128,    2,    0,    0,  128,  128, "gfx950", "fmha_v3_fwd/", "_ZN5aiter32fmha_fwd_hd192_hd128_bf16_causalE", "fwd_hd192_hd128_bf16_causal.co"),
    ADD_CFG("bf16",  192,  128,    0,    1,    0,  128,  128, "gfx950", "fmha_v3_fwd/", "_ZN5aiter31fmha_fwd_hd192_hd128_bf16_groupE", "fwd_hd192_hd128_bf16_group.co"),
    ADD_CFG("bf16",  192,  128,    2,    1,    0,  128,  128, "gfx950", "fmha_v3_fwd/", "_ZN5aiter38fmha_fwd_hd192_hd128_bf16_causal_groupE", "fwd_hd192_hd128_bf16_causal_group.co"),
};

} // namespace AOTRITON_NS::v3::flash::aiter
