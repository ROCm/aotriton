# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from ..base import (
    typed_choice as TC,
    Functional,
    Interface,
)
import numpy as np

class BaseTuneCodeGenerator(ABC):
    BIN_INDEX_SUFFIX = '_binned_index'

    def __init__(self,
                 args,
                 f : Functional,
                 dataframe_for_tuning : 'pandas.DataFrame | None',
                 parent_repo):
        self._args = args
        self._f = f
        self._df = dataframe_for_tuning
        self._parent_repo = parent_repo
        self._cc_file = self.get_cc_file(f)

    def get_cc_file(self, f):
        iface = self._f.meta_object
        tune_dir = self._args.build_dir / iface.FAMILY / f'{iface.TUNE_NAME}.{iface.NAME}'
        tune_dir.mkdir(parents=True, exist_ok=True)
        return tune_dir / (f.tunecc_signature + '.cc')

    @property
    def cc_file(self):
        return self._cc_file

    @abstractmethod
    def generate(self):
        pass

    def codegen_deduplicated_lut_function(self, lut_ctype, lut_cshape):
        iface = self._f.meta_object
        d = {
            'param_class_name'      : iface.param_class_name,
            'lut_ctype'             : lut_ctype,
            'lut_shape'             : lut_cshape,
            'binning_autotune_keys' : self.codegen_binning_code(),
            'binned_indices'        : self.codegen_binned_indices(),
        }
        lambda_params = '(const {param_class_name}& params, int mod_number, {lut_ctype} lut{lut_shape})'
        stmt = []
        stmt.append(lambda_params + ' {{')
        stmt.append('    {binning_autotune_keys}')
        stmt.append('    return lut[mod_number]{binned_indices};')
        stmt.append('}}')
        ALIGN = '\n'
        lambda_src = ALIGN.join(stmt).format_map(d)
        lut_registry = self._parent_repo.get_function_registry('lut_function')
        lut_params = lambda_params.format_map(d)
        lut_function_pfx = f'{iface.NAME}__lut_lambda'
        lut_function_name = lut_registry.register(lambda_src, 'int', lut_function_pfx, lut_params)
        return lut_function_name

    def codegen_binning_code(self):
        if self._binning_dict is None:
            return ''
        ALIGN = '\n' + 4 * ' '  # Note codegen_binning_lambda already contains ';'
        stmt = []
        for key, algo in self._binning_dict.items():
            stmt += algo.codegen_binning_lambda(key, out_suffix=self.BIN_INDEX_SUFFIX)
        return ALIGN.join(stmt)

    def codegen_binned_indices(self):
        if self._binning_dict is None:
            return '[0]'
        return ''.join([f'[{key}{self.BIN_INDEX_SUFFIX}]' for key in self._binning_dict.keys()])

    def codegen_format_lut(self, lut_tensor):
        f = self._f
        max_value = np.max(lut_tensor)
        for dtype in [np.int8, np.int16, np.int32, np.int64]:
            if max_value < np.iinfo(dtype).max:
                break
        ctype =  f'int{np.iinfo(dtype).bits}_t'
        cshape = ''.join([f'[{s}]' for s in lut_tensor.shape])
        def fmt(t):
            return np.array2string(t, separator=',').replace('[', '{').replace(']', '}')
        tensor_text_list = []
        for i, gpu in enumerate(f.optimized_for):
            text  = f'\n// GPU {gpu}\n'
            text += fmt(lut_tensor[i])
            text += f'\n// End of GPU {gpu}\n'
            tensor_text_list.append(text)
        cdata = '{' + '\n,\n'.join(tensor_text_list) + '}'
        return ctype, cshape, cdata
