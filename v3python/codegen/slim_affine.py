from pathlib import Path
from argparse import Namespace

from ..base import (
    Interface,
    Functional,
)
from ..utils import (
    RegistryRepository,
    log,
)
from .template import get_template
from .interface import InterfaceGenerator
from .common import codegen_includes

class SlimAffineGenerator(InterfaceGenerator):
    HEADER_TEMPLATE = get_template('slim_affine.h')
    SOURCE_TEMPLATE = get_template('slim_affine.cc')
    PFX = 'affine'

    def __init__(self, args, iface : Interface, parent_repo : RegistryRepository):
        super().__init__(args, iface, parent_repo)
        akdesc = iface
        # Patch _target_arch since affine kernel may not support all arches.
        self._target_arch = { arch: gpus for arch, gpus in self._target_arch.items() if arch in akdesc.SUPPORTED_ARCH }
        del self._target_gpus  # For safety
        self._target_arch_keys = list(self._target_arch.keys())

    def create_sub_generator(self, functional : Functional, df : 'pandas.DataFrame', sql : str):
        raise RuntimeError("There should be no calls to SlimAffineGenerator.create_sub_generator"
                           " since slim affine kernel has vendored dispatcher.")

    # TODO: generate rules to package .co files into .aks2 files
    # gfx950/fmha_v3_bwd/*.co -> amd-gfx950/flash/fmha_v3_bwd.aks2
    def generate(self):
        super().generate()
        asm_registry = self.this_repo.get_hsaco_registry('asms')
        akdesc = self._iface
        for aks2_path, inarchive_path, kernel_co in akdesc.co_gen(self._args.build_dir, self._target_arch):
            relocate_rule = '|' + inarchive_path + '|' + kernel_co.absolute().as_posix()
            asm_registry.register(aks2_path, relocate_rule, append=True)

    def write_shim_header(self, functionals, fout):
        akdesc = self._iface
        shared_iface = akdesc.SHARED_IFACE is not None
        assert shared_iface, 'Slim affine kernel class {akdesc} must define SHARED_IFACE field.'
        shared_iface_family = akdesc.SHARED_IFACE.FAMILY if shared_iface else akdesc.FAMILY
        if shared_iface:
            self._add_iface_for_source(akdesc.SHARED_IFACE)
        has_cookie_object = 1 if akdesc.COOKIE_CLASS is not None else 0
        cookie_class = '// optional cookie class' if not has_cookie_object else akdesc.COOKIE_CLASS
        d = {
            'shared_iface_family'   : shared_iface_family,
            'shared_iface'          : 1 if shared_iface else 0,
            'kernel_family_name'    : akdesc.FAMILY,
            'affine_kernel_name'      : akdesc.NAME,
            'param_class_name'      : akdesc.param_class_name,
            'context_class_name'    : akdesc.context_class_name,
            'has_cookie_object'     : has_cookie_object,
            'cookie_class'          : cookie_class,
        }
        d['includes'] = codegen_includes(self._hdr_include_repo.get_data())
        print(self.HEADER_TEMPLATE.format_map(d), file=fout)

    def write_shim_source(self, functionals, fout):
        akdesc = self._iface
        shared_iface = akdesc.SHARED_IFACE is not None
        shared_iface_family = akdesc.SHARED_IFACE.FAMILY if shared_iface else akdesc.FAMILY
        d = {
            'shared_iface'        : 1 if shared_iface else 0,
            'shared_iface_family' : shared_iface_family,
            'kernel_family_name'  : akdesc.FAMILY,
            'affine_kernel_name'  : akdesc.NAME,  # TODO: use signature so AMD_LOG_LEVEL=3 is more meaningful
            'param_class_name'    : akdesc.param_class_name,
            'context_class_name'  : akdesc.context_class_name,
            'get_archmod_number_body'               : self.codegen_archmod_number_body(),
        }
        d['includes'] = codegen_includes(self._src_include_repo.get_data())
        print(self.SOURCE_TEMPLATE.format_map(d), file=fout)

