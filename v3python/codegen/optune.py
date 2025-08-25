# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Generate <family>/optune.<kernel_name>/<functional>.cc

from .template import get_template
from ..base import (
    typed_choice as TC,
    Functional,
    Interface,
)
from .basetune import BaseTuneCodeGenerator
from ..utils import (
    LazyFile,
    dict2json,
    log,
)

class OptuneCodeGenerator(BaseTuneCodeGenerator):
    OPTUNE_TEMPLATE = get_template('optune_table_entry.cc')

    def __init__(self,
                 args,
                 f : Functional,
                 dataframe_for_tuning : 'pandas.DataFrame | None',
                 parent_repo):
        super().__init__(args, f, dataframe_for_tuning, parent_repo)
        iface = self._f.meta_object
        if self._df is None or self._df.empty:
            self._lut_tensor, self._backend_names, self._binning_dict = iface.translate_empty_dataframe(f)
        else:
            self._lut_tensor, self._backend_names, self._binning_dict = iface.translate_dataframe(f, self._df)

    @property
    def is_trivial(self):
        return len(self._backend_names) <= 1

    def generate_trivial(self):
        functional = self._f
        iface = functional.meta_object
        mono_backend = str(self._backend_names[0])
        repo = self._parent_repo.get_dict_registry('trivial_tunes')
        repo.register((functional.arch_number, functional.godel_number), mono_backend)

    def generate(self):
        with LazyFile(self._cc_file) as fout:
            self.write_optune_src(fout)

    def write_optune_src(self, fout):
        f = self._f
        iface = f.meta_object
        lut_ctype, lut_cshape, lut_cdata = self.codegen_format_lut(self._lut_tensor)
        # gpu_kernel_image_dir = args.build_dir / f.FAMILY / f'gpu_kernel_image.{f.NAME}'
        package_path = str(f.full_filepack_path)
        d = {
            'op_family_name'        : iface.FAMILY,
            'op_name'               : iface.NAME,
            'arch_number'           : f.arch_number,
            'godel_number'          : f.godel_number,
            'lut_ctype'             : lut_ctype,
            'lut_cshape'            : lut_cshape,
            'lut_data'              : lut_cdata,
            'context_class_name'    : iface.context_class_name,
            'op_param_class_name'   : iface.param_class_name,
            'deduplicated_lut_function' : self.codegen_deduplicated_lut_function(lut_ctype, lut_cshape),
            'human_readable_signature' : f.human_readable_signature,
        }
        print(self.OPTUNE_TEMPLATE.format_map(d), file=fout)

