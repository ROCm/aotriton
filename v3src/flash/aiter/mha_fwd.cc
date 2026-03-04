#include <aotriton/_internal/aiter_hip_common.h>
#include <aotriton/_internal/flash/aiter.h>
#include <aotriton/util.h>
#include <memory>
#include <string>
#include "asm_fmha_v3_fwd_configs.hpp"

// Copied from AITER

namespace AOTRITON_NS::v3::flash::aiter {

int get_cfg_mask_type(const mha_fwd_args& a)
{
    if(a.mask_type == 0)
    {
        return 0;
    }
    if((a.mask_type == 2 || (a.mask_type == 1 && a.seqlen_q == a.seqlen_k)) &&
       a.window_size_left == -1 && a.window_size_right == 0)
    {
        return 2;
    }
    return -1;
}

std::string get_kernel_name_key(std::string_view arch_id,
                                const std::string& data_type,
                                int hdim_q,
                                int hdim_v,
                                int mask_type,
                                int bf16_cvt,
                                bool mode,
                                const CFG* cfgs)
{
    std::string kernel_name_key{};
    for(const auto& el : *cfgs)
    {
        const auto& cfg = el.second;
        if(cfg.arch != arch_id)
        {
            continue;
        }

        if(cfg.dtype == data_type && cfg.hdim_q == hdim_q && cfg.hdim_v == hdim_v &&
           cfg.mask == mask_type && cfg.mode == mode)
        {
            if(arch_id == "gfx950")
            {
                kernel_name_key = el.first;
                break;
            }
            else if(arch_id == "gfx942" && cfg.bf16_cvt == bf16_cvt)
            {
                kernel_name_key = el.first;
                break;
            }
        }
    }

    return kernel_name_key;
}

std::string get_kernel_co_name(const std::string& cfg_co_name,
                               std::string_view arch_id,
                               AOTRITON_NS::Gpu gpu)
{
    std::string co_name = cfg_co_name;
    if(arch_id == "gfx942")
    {
        auto pos = cfg_co_name.rfind('/');
        if (gpu == GPU_AMD_ARCH_GFX942_MOD2) {
            co_name = cfg_co_name.substr(0, pos + 1) + "MI308/" + cfg_co_name.substr(pos + 1);
        } else {
            co_name = cfg_co_name.substr(0, pos + 1) + "MI300/" + cfg_co_name.substr(pos + 1);
        }
    }
    return co_name;
}

void init_fmha_fwd_v3_args(fmha_fwd_v3_args& args,
                           const mha_fwd_args& a,
                           int ts_qo,
                           std::string_view arch_id)
{
    int tune_opt = 5;
    // if num_head is not 8N, or seqlen is bigger than 16K, downgrade to 2and3
    if(a.mask_type != 0 && ((a.nhead_q % 8 != 0) || (a.seqlen_q > 16384)))
    {
        tune_opt -= 2;
    }
    if(a.hdim_q == 192 && a.hdim_v == 128 && arch_id == "gfx942")
    {
        tune_opt = 0;
    }
    args.ptr_o   = a.o_ptr;
    args.ptr_q   = a.q_ptr;
    args.ptr_k   = a.k_ptr;
    args.ptr_v   = a.v_ptr;
    args.ptr_lse = a.lse_ptr;

    args.scalar           = a.scale_s;
    args.s_seq_len        = a.seqlen_q;
    args.s_Seqs           = a.stride_q * 2;
    args.s_Ts             = ts_qo * a.stride_q * 2;
    args.s_Hs             = a.nhead_stride_q * 2;
    args.s_Bs             = a.batch_stride_q * 2;
    args.s_gqa            = a.nhead_q / a.nhead_k;
    args.s_k_Seqs         = a.stride_k * 2;
    args.s_k_Hs           = a.nhead_stride_k * 2;
    args.s_k_Bs           = a.batch_stride_k * 2;
    args.s_opt            = tune_opt;
    args.s_lse            = a.has_lse ? 1 : 0;
    args.s_kv_seq_len     = a.seqlen_k;
    args.s_qk_head_dim    = a.hdim_q;
    args.s_v_head_dim     = a.hdim_v;
    args.s_q_head_num     = a.nhead_q;
    args.s_v_Seqs         = a.stride_v * 2;
    args.s_v_Hs           = a.nhead_stride_v * 2;
    args.s_v_Bs           = a.batch_stride_v * 2;
    args.s_o_Seqs         = a.stride_o * 2;
    args.s_o_Hs           = a.nhead_stride_o * 2;
    args.s_o_Bs           = a.batch_stride_o * 2;
    args.s_lse_Hs         = a.nhead_stride_lse * 4;
    args.ptr_qseq         = nullptr;
    args.ptr_kseq         = nullptr;
    args.ptr_qseq_padding = nullptr;
    args.ptr_kseq_padding = nullptr;
    // batch mode does not support padded
    if(a.is_group_mode)
    {
        args.ptr_kseq_padding = a.seqstart_k_ptr;
        if(a.cu_seqlen_k_ptr && a.seqstart_k_ptr)
        {
            args.ptr_kseq = a.cu_seqlen_k_ptr;
        }
        else
        {
            args.ptr_kseq = a.seqstart_k_ptr;
        }
        args.ptr_qseq_padding = a.seqstart_q_ptr;
        if(a.cu_seqlen_q_ptr && a.seqstart_q_ptr)
        {
            args.ptr_qseq = a.cu_seqlen_q_ptr;
        }
        else
        {
            args.ptr_qseq = a.seqstart_q_ptr;
        }
    }
}

std::tuple<int, int, int> get_grid_dim(const mha_fwd_args& a, int ts_qo, std::string_view arch_id)
{

    int tg_div = (a.mask_type != 0) ? 2 : 1;
    if(arch_id == "gfx942" && a.is_group_mode && a.hdim_q == 192 && a.hdim_v == 128)
    {
        tg_div = 1; // do not merge the head and tail in seqlen_q direction
    }
    // batch
    int gdx = ((a.seqlen_q + ts_qo - 1) / ts_qo + tg_div - 1) / tg_div;
    int gdy = a.nhead_q;
    int gdz = a.batch;
    if(arch_id == "gfx942" && a.hdim_q == 192 && a.hdim_v == 128)
    {
        gdx = a.nhead_q;
        gdy = (a.seqlen_q + ts_qo - 1) /
              ts_qo; // do not merge the head and tail in seqlen_q direction
        gdz = a.batch;
    }
    // group
    if(a.is_group_mode)
    {
        gdx = a.nhead_q;
        gdy = a.batch;
        gdz = ((a.seqlen_q + ts_qo - 1) / ts_qo + tg_div - 1) / tg_div;
    }

    return std::make_tuple(gdx, gdy, gdz);
}

float fmha_fwd_v3(mha_fwd_args a, const ck_tile::stream_config& s)
{
    auto [gpu, arch_id] = get_gpu_arch(s.stream_id_);

    if((!a.use_asm_v3) || (a.hdim_q != 192 && a.hdim_q != 128) || (a.hdim_v != 128) ||
       (a.data_type != "bf16") || (a.bias_type != 0) || (a.p_drop > 0.f) ||
       ((arch_id != "gfx942") && (arch_id != "gfx950")))
    {
        // std::cout << "[Warning]unsupported condition in fwd_v3!!!" << std::endl;
        return -1;
    }

    auto fwd_cfgs               = &cfg_fmha_fwd;
    int cfg_mask_type           = get_cfg_mask_type(a);
    std::string kernel_name_key = get_kernel_name_key(arch_id,
                                                      a.data_type,
                                                      a.hdim_q,
                                                      a.hdim_v,
                                                      cfg_mask_type,
                                                      a.how_v3_bf16_cvt,
                                                      a.is_group_mode,
                                                      fwd_cfgs);
    auto it                     = fwd_cfgs->find(kernel_name_key);
    if(it == fwd_cfgs->end())
    {
        return -1;
    };

    if(a.v3_api_check)
    {
        return 1;
    };

    AiterAsmKernel* impl_ptr = nullptr;
    static thread_local std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>>
        impl_ptr_map;

    const auto& cfg     = it->second;
    const char* name    = cfg.knl_name.c_str();
    std::string co_name = get_kernel_co_name(cfg.co_name, arch_id, gpu);

    auto result = impl_ptr_map.emplace(name, nullptr);
    if(result.second)
    {
        result.first->second = std::make_unique<AiterAsmKernel>(name, co_name.c_str());
    }
    impl_ptr = result.first->second.get();

    fmha_fwd_v3_args args;
    int arg_size = sizeof(args);
    init_fmha_fwd_v3_args(args, a, cfg.ts_qo, arch_id);

    int bdx              = (a.hdim_q == 192 && a.hdim_v == 128) ? 256 : 512;
    auto [gdx, gdy, gdz] = get_grid_dim(a, cfg.ts_qo, arch_id);

    return ck_tile::launch_kernel(s, [=](const ck_tile::stream_config& s_) mutable {
        // Explicit assignment forces evaluation order and prevents compiler from
        // reordering operations that could lead to accessing uninitialized args
        void* args_ptr     = &args;
        void* arg_size_ptr = &arg_size;
        impl_ptr->launch_kernel({args_ptr, arg_size_ptr, gdx, gdy, gdz, bdx, 1, 1, s_.stream_id_});
    });
}

} // namespace aiter
