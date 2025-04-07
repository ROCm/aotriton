# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

AOTRITON_SUPPORTED_GPUS = (
    'gfx90a_mod0',
    'gfx942_mod0',
    # 'gfx942_mod1',
    # 'gfx942_mod2',
    'gfx950_mod0',
    'gfx1100_mod0',
    'gfx1201_mod0',
)
#     'MI200'     : 'GPU_ARCH_AMD_GFX90A',
#     'MI300X'    : 'GPU_ARCH_AMD_GFX942',
#     'Navi31'    : 'GPU_ARCH_AMD_GFX1100',
#     'Navi32'    : 'GPU_ARCH_AMD_GFX1101',
#     'Unidentified'    : 'GPU_ARCH_AMD_GFX950',
#     'RX9070XT'  : 'GPU_ARCH_AMD_GFX1201',


# AOTRITON_GPU_ARCH_TUNING_STRING = {
#     'MI200'     : 'gfx90a',
#     'MI300X'    : 'gfx942',
#     'Navi31'    : 'gfx1100',
#     'Navi32'    : 'gfx1101',
#     'Unidentified'    : 'gfx950',
#     'RX9070XT'  : 'gfx1201',
# }

AOTRITON_TUNING_DATABASE_REUSE = {
    'gfx950_mod0'  : 'gfx942_mod0',
    'gfx1201_mod0' : 'gfx1100_mod0',
}

AOTRITON_ARCH_WARPSIZE = {
    'gfx90a'     : 64,
    'gfx942'     : 64,
    'gfx950'     : 64,
    'gfx1100'    : 32,
    'gfx1201'    : 32,
}

def gpu2arch(gpu : str) -> str:
    return gpu.split('_mod')[0]

def cluster_gpus(gpus : list[str]) -> dict[str : list[str]]:
    ret = {}
    for gpu in gpus:
        arch = gpu2arch(gpu)
        ret[arch].append(gpu)
    return ret

def select_gpus(target_arch, target_gpus) -> list[str]:
    if target_gpus:
        return target_gpus
    ret = []
    for gpu in AOTRITON_SUPPORTED_GPUS:
        arch = gpu2arch(gpu)
        if arch in target_arch:
            ret.append(gpu)
    return ret

def main():
    import argparse
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--target_arch", type=str, default=None, nargs='*', choices=AOTRITON_ARCH_WARPSIZE.keys(),
                   help="Select specific list of architectures and related GPU tuning information.")
    p.add_argument("--target_gpus", type=str, default=None, nargs='*', choices=AOTRITON_SUPPORTED_GPUS,
                   help="Select specific list of GPUs. Overrides --target_arch.")
    args = p.parse_args()
    gpus = select_gpus(args.target_arch, args.target_gpus)
    print(";".join(gpus))

if __name__ == '__main__':
    main()
