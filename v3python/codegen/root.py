# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Root of the Generation process

from ..rules import (
    kernels as triton_kernels,
    operators as dispatcher_operators,
)
from .kernel import KernelShimGenerator
from .operator import OperatorGenerator
from ..utils import (
    LazyFile,
    RegistryRepository,
    log,
)
from ..gpu_targets import AOTRITON_ARCH_TO_DIRECTORY

class RootGenerator(object):
    def __init__(self, args):
        self._args = args

    def generate(self):
        args = self._args
        hsaco_for_kernels = []
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

        with LazyFile(args.build_dir / 'Bare.shim') as shimfile:
            for shim in shims:
                print(str(shim.absolute()), file=shimfile)
        if args.noimage_mode:
            return
        # TODO: Support Cluter Functionals
        #       Implemented this in
        #       Functional.filepack_signature (used by Functional.full_filepack_path)
        with (
            LazyFile(args.build_dir / 'Bare.compile') as rulefile,
            LazyFile(args.build_dir / 'Bare.cluster') as clusterfile,
        ):
            for k, hsacos in hsaco_for_kernels:
                image_path = args.build_dir / k.FAMILY / f'gpu_kernel_image.{k.NAME}'
                image_path.mkdir(parents=True, exist_ok=True)
                for functional, signatures in hsacos.items():
                    log(lambda : f'{signatures=}')
                    for ksig in signatures:
                        # TODO: Add sanity check to ensure
                        # k == functional.meta_object and
                        # functional == ksig._functional ?
                        self.write_hsaco(k, image_path, functional, ksig, rulefile)
                    self.write_cluster(k, image_path, functional, signatures, clusterfile)

    def _objfn(self, kdesc, ksig):
        return kdesc.NAME + '-Sig-' + ksig.full_compact_signature + '.hsaco'

    def _absobjfn(self, path, kdesc, ksig):
        full = path / self._objfn(kdesc, ksig)
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

    def write_cluster(self, kdesc, path, functional, signatures, clusterfile):
        full_filepack_path = functional.full_filepack_path
        print(*full_filepack_path.parts,
              end=';', sep=';', file=clusterfile)
        path_list = [self._absobjfn(path, kdesc, ksig) for ksig in signatures]
        print(*path_list, sep=';', file=clusterfile)
