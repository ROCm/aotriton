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
    'gfx1103_mod0',
    'gfx1150_mod0',
    'gfx1151_mod0',
    'gfx1152_mod0',
    'gfx1153_mod0',
    'gfx1201_mod0',
    'gfx1200_mod0',
    'gfx1250_mod0',
)

# TODO: AOTRITON_TUNING_DATABASE_REUSE -> AOTRITON_TUNING_DATABASE_FALLBACK
# Load fallback entries first, and override with "patching" entries from real GPU
AOTRITON_TUNING_DATABASE_REUSE = {
    'gfx1101_mod0' : 'gfx1100_mod0',
    'gfx1102_mod0' : 'gfx1100_mod0',
    'gfx1103_mod0' : 'gfx1100_mod0',
    'gfx1200_mod0' : 'gfx1201_mod0',
    'gfx1150_mod0' : 'gfx1100_mod0',
    'gfx1151_mod0' : 'gfx1100_mod0',
    'gfx1152_mod0' : 'gfx1100_mod0',
    'gfx1153_mod0' : 'gfx1100_mod0',
    'gfx1250_mod0' : 'gfx942_mod0',
}

AOTRITON_ARCH_TO_PACK = {
    'gfx90a'    : 'gfx90a',
    'gfx942'    : 'gfx942',
    'gfx950'    : 'gfx950',
    'gfx1100'   : 'gfx11xx',
    'gfx1101'   : 'gfx11xx',
    'gfx1102'   : 'gfx11xx',
    'gfx1103'   : 'gfx11xx',
    'gfx1150'   : 'gfx11xx',
    'gfx1151'   : 'gfx11xx',
    'gfx1152'   : 'gfx11xx',
    'gfx1153'   : 'gfx11xx',
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
    'gfx1103'    : 32,
    'gfx1150'    : 32,
    'gfx1151'    : 32,
    'gfx1152'    : 32,
    'gfx1153'    : 32,
    'gfx1201'    : 32,
    'gfx1200'    : 32,
    'gfx1250'    : 32,
    'gfx1251'    : 32,
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
    import sys
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    arch_metavar = '{' + ','.join(AOTRITON_ARCH_WARPSIZE.keys()) + '}'
    p.add_argument("--target_arch", type=str, default=None, nargs='*', metavar=arch_metavar,
                   help="Select architectures and related GPU tuning information. Unsupported ones will be ignored.")
    p.add_argument("--target_gpus", type=str, default=None, nargs='*', choices=AOTRITON_SUPPORTED_GPUS,
                   help="Select specific list of GPUs. Overrides --target_arch.")
    args = p.parse_args()
    gpus = select_gpus(args.target_arch, args.target_gpus)
    if not gpus:
        supported = sorted(AOTRITON_ARCH_WARPSIZE.keys())
        requested = list(args.target_gpus or args.target_arch or [])
        unsupported = [a for a in (args.target_arch or []) if a not in AOTRITON_ARCH_WARPSIZE]
        print(
            "aotriton: no supported target arch in input list\n"
            f"  requested: {', '.join(requested) if requested else '(none)'}\n"
            f"  unsupported (filtered out): {', '.join(unsupported) if unsupported else '(none)'}\n"
            f"  supported: {', '.join(supported)}\n"
            "hint: disable aotriton in your build system when no requested arch matches\n"
            "      the supported set. See https://github.com/ROCm/aotriton/issues/169.",
            file=sys.stderr,
        )
        sys.exit(2)
    print(";".join(gpus))

if __name__ == '__main__':
    main()
