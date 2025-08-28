# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from collections import defaultdict

AOTRITON_SUPPORTED_GPUS = (
    'gfx90a_mod0',
    'gfx942_mod0',
    # 'gfx942_mod1',
    # 'gfx942_mod2',
    'gfx950_mod0',
    'gfx1100_mod0',
    'gfx1101_mod0',
    'gfx1102_mod0',
    'gfx1151_mod0',
    'gfx1150_mod0',
    'gfx1201_mod0',
    'gfx1200_mod0',
    'gfx1250_mod0',
)

# TODO: AOTRITON_TUNING_DATABASE_REUSE -> AOTRITON_TUNING_DATABASE_FALLBACK
# Load fallback entries first, and override with "patching" entries from real GPU
AOTRITON_TUNING_DATABASE_REUSE = {
    'gfx1101_mod0' : 'gfx1100_mod0',
    'gfx1102_mod0' : 'gfx1100_mod0',
    'gfx1200_mod0' : 'gfx1201_mod0',
    'gfx1151_mod0' : 'gfx1100_mod0',
    'gfx1150_mod0' : 'gfx1100_mod0',
    'gfx1250_mod0' : 'gfx942_mod0',
}

AOTRITON_ARCH_TO_PACK = {
    'gfx90a'    : 'gfx90a',
    'gfx942'    : 'gfx942',
    'gfx950'    : 'gfx950',
    'gfx1100'   : 'gfx11xx',
    'gfx1101'   : 'gfx11xx',
    'gfx1102'   : 'gfx11xx',
    'gfx1151'   : 'gfx11xx',
    'gfx1150'   : 'gfx11xx',
    'gfx1201'   : 'gfx120x',
    'gfx1200'   : 'gfx120x',
    'gfx1250'   : 'gfx1250',
}

AOTRITON_ARCH_TO_DIRECTORY = {
    k : f'amd-{v}' for k, v in AOTRITON_ARCH_TO_PACK.items()
}

AOTRITON_ARCH_WARPSIZE = {
    'gfx90a'     : 64,
    'gfx942'     : 64,
    'gfx950'     : 64,
    'gfx1100'    : 32,
    'gfx1101'    : 32,
    'gfx1102'    : 32,
    'gfx1151'    : 32,
    'gfx1150'    : 32,
    'gfx1201'    : 32,
    'gfx1200'    : 32,
    'gfx1250'    : 32,
}

AOTRITON_ARCH_PRODUCTION_LINE = {
    'gfx90a'     : 'CDNA',
    'gfx942'     : 'CDNA',
    'gfx950'     : 'CDNA',
    'gfx1100'    : 'RDNA',
    'gfx1101'    : 'RDNA',
    'gfx1102'    : 'RDNA',
    'gfx1151'    : 'RDNA',
    'gfx1150'    : 'RDNA',
    'gfx1201'    : 'RDNA',
    'gfx1200'    : 'RDNA',
    'gfx1250'    : 'CDNA',
}

def gpu2arch(gpu : str) -> str:
    return gpu.split('_mod')[0]

def cluster_gpus(gpus : list[str]) -> dict[str : list[str]]:
    ret = defaultdict(list)
    for gpu in gpus:
        arch = gpu2arch(gpu)
        ret[arch].append(gpu)
    for k, v in ret.items():
        ret[k] = sorted(v)
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
    arch_metavar = '{' + ','.join(AOTRITON_ARCH_WARPSIZE.keys()) + '}'
    p.add_argument("--target_arch", type=str, default=None, nargs='*', metavar=arch_metavar,
                   help=f"Select architectures and related GPU tuning information. Unsupported ones will be ignored.")
    p.add_argument("--target_gpus", type=str, default=None, nargs='*', choices=AOTRITON_SUPPORTED_GPUS,
                   help="Select specific list of GPUs. Overrides --target_arch.")
    args = p.parse_args()
    gpus = select_gpus(args.target_arch, args.target_gpus)
    print(";".join(gpus))

if __name__ == '__main__':
    main()
