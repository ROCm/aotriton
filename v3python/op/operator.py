# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from ..base import (
    Interface,
    Functional,
)
import numpy as np

class Operator(Interface):
    TUNE_NAME = 'optune'

    def __init__(self, backends : list[Interface]):
        super().__init__()
        self._late_init()
        assert len(backends) > 0, f'Operator {self.UNTYPED_FULL_NAME} needs at least one backend'
        self._backends = backends
        self._backend_dict = { b.enum_name : b for b in self._backends }

    @property
    def enum_name(self):
        # CamelName = self.NAME.replace('_', ' ').title().replace(' ', '')
        return f'kOp_{self.class_name_base}'

    def list_backends(self):
        return self._backends

    @property
    def fallback_backend(self):
        return self._backends[0]

    @property
    def nbackends(self):
        return len(self._backends)

    '''
    Operator is backed KernelDescription/MetroKernel/AffineKernel
    and has no additional params
    '''
    def list_non_functional_params(self):
        return []

    # TODO: Unify with KernelDescription.translate_dataframe?
    def translate_dataframe(self, f : Functional, df : 'pandas.DataFrame'):
        sparse_keys = [ f'inputs${key}' for key in self.OPTUNE_KEYS.keys() ]
        nkeys = len(sparse_keys)
        def sorted_unique_key(key):
            return np.unique(df[key].to_numpy()).tolist()
        sparse_key_possible_values = { key : sorted_unique_key(key) for key in sparse_keys }
        binning_dict = { key : algo(sparse_key_possible_values[f'inputs${key}']) for key, algo in self.OPTUNE_KEYS.items() }
        lut_shape = [f.noptimized_for] + [ len(sparse_key_possible_values[key]) for key in sparse_keys ]
        lut_tensor = np.empty(lut_shape, dtype=object)
        backend_key = 'op$backend'
        '''
        Bucketing autotune indices
        '''
        for i, ind_key in enumerate(sparse_keys):
            bucket = sparse_key_possible_values[ind_key]
            def discretization(v):
                return bucket.index(v)
            df[f'$$ind_{i}'] = df[ind_key].apply(discretization)
        for i, gpu in enumerate(f.optimized_for):
            if i > 0:
                lut_tensor[i] = lut_tensor[0]
            df_i = df[df['gpu'] == gpu]
            inds = tuple([df_i[f'$$ind_{i}'] for i in range(nkeys)])
            backends = df_i[backend_key]
            lut_tensor[i][inds] = backends
        '''
        LUT tensor for Optune stores string directly.
        '''
        return lut_tensor, np.unique(lut_tensor).tolist(), binning_dict

    def translate_empty_dataframe(self, f : Functional):
        lut_tensor = np.zeros([f.noptimized_for, 1], dtype=np.int8)
        return lut_tensor, [self.fallback_backend.enum_name], None
