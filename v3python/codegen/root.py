# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Root of the Generation process

from pathlib import Path
from collections import defaultdict
from ..rules import (
    kernels as triton_kernels,
    operators as dispatcher_operators,
    affine_kernels,
)
from .kernel import KernelShimGenerator
from .affine import AffineGenerator
from .operator import OperatorGenerator
from ..utils import (
    LazyFile,
    RegistryRepository,
    log,
)
from .common import (
    hsaco_dir,
    hsaco_filename,
)
from ..gpu_targets import AOTRITON_ARCH_TO_DIRECTORY

class RootGenerator(object):
    def __init__(self, args):
        self._args = args

    def generate(self):
        args = self._args
        hsaco_for_kernels = []
        asms_for_kernels = []
        shims = []
        for op in dispatcher_operators:
            opg = OperatorGenerator(self._args, op, parent_repo=None)
            opg.generate()
            shims += opg.shim_files
        for k in triton_kernels:
            ksg = KernelShimGenerator(self._args, k, parent_repo=None)
            ksg.generate()
            hsacos = ksg.this_repo.get_data('hsaco')
            hsaco_for_kernels.append((k, hsacos))
            shims += ksg.shim_files
        # print(f'{affine_kernels=}')
        for ak in affine_kernels:
            log(lambda : f'{ak.__class__=}')
            aksg = AffineGenerator(self._args, ak, parent_repo=None)
            aksg.generate()
            asms = aksg.this_repo.get_data('asms')
            asms_for_kernels.append((ak, asms))
            shims += aksg.shim_files

        if args.build_for_tuning_second_pass:
            return

        with LazyFile(args.build_dir / 'Bare.shim') as shimfile:
            for shim in shims:
                print(str(shim.absolute()), file=shimfile)
        if args.noimage_mode:
            return
        # TODO: Support Cluter Functionals
        #       Implemented this in
        #       Functional.filepack_signature (used by Functional.full_filepack_path)
        cluster_dict = defaultdict(list)
        with LazyFile(args.build_dir / 'Bare.compile') as rulefile:
            for kdesc, hsacos in hsaco_for_kernels:
                image_path = hsaco_dir(args.build_dir, kdesc)
                image_path.mkdir(parents=True, exist_ok=True)
                for functional, signatures in hsacos.items():
                    log(lambda : f'{signatures=}')
                    for ksig in signatures:
                        # TODO: Add sanity check to ensure
                        # k == functional.meta_object and
                        # functional == ksig._functional ?
                        self.write_hsaco(kdesc, image_path, functional, ksig, rulefile)
                    ffp = functional.full_filepack_path
                    aol = [self._absobjfn(image_path, kdesc, ksig) for ksig in signatures]
                    cluster_dict[ffp] += aol
        with LazyFile(args.build_dir / 'Bare.cluster') as clusterfile:
            for ffp, aol in cluster_dict.items():
                self.write_cluster(ffp, aol, clusterfile)
        '''
        Note: Affine kernel's functionals have residual choices, so it is not
        completely the same with Triton kernel/Interface's functionals

        However, Affine kernel's functionals do not use residual choices to
        compute full_filepack_path, so eventually .hsaco and .co files will be
        consolated into the same .aks2 file, which is intentional.
        '''
        affine_dict = defaultdict(list)
        for akdesc, asm_registry in asms_for_kernels:
            for package_path, asms in asm_registry.items():
                affine_dict[Path(package_path)] += [ self._absasmfn(asm) for asm in asms ]
        with LazyFile(args.build_dir / 'Affine.cluster') as clusterfile:
            for ffp, aol in affine_dict.items():
                self.write_cluster(ffp, list(set(aol)), clusterfile)

    def _absobjfn(self, path, kdesc, ksig):
        full = path / hsaco_filename(kdesc, ksig)
        return str(full.absolute())

    def write_hsaco(self, kdesc, path, functional, ksig, rulefile):
        log(lambda : f'{ksig=}')
        srcfn = kdesc.triton_source_path
        print(self._absobjfn(path, kdesc, ksig),
              str(srcfn.absolute()),
              kdesc.triton_kernel_name,
              ksig.num_warps,
              ksig.num_stages,
              ksig.waves_per_eu,
              functional.arch,
              ksig.triton_signature_string,  # Functional is not Triton-specific
              sep=';', file=rulefile)

    def write_cluster(self, ffp, aol, clusterfile):
        print(*ffp.parts, end=';', sep=';', file=clusterfile)
        print(*aol, sep=';', file=clusterfile)

    def _absasmfn(self, asm_path):
        full = self._args.root_dir / asm_path
        return str(full.absolute())

    # def write_cluster(self, kdesc, path, functional, signatures, clusterfile):
    #     full_filepack_path = functional.full_filepack_path
    #     print(*full_filepack_path.parts,
    #           end=';', sep=';', file=clusterfile)
    #     path_list = [self._absobjfn(path, kdesc, ksig) for ksig in signatures]
    #     print(*path_list, sep=';', file=clusterfile)
