# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Generate <family>/optune.<kernel_name>/<functional>.cc

from ..base import (
    typed_choice as TC,
    Functional,
    Interface,
)
from .basetune import BaseTuneCodeGenerator

class OptuneCodeGenerator(BaseTuneCodeGenerator):
    def __init__(self,
                 args,
                 f : Functional,
                 dataframe_for_tuning : 'pandas.DataFrame | None',
                 parent_repo):
        super().__init__(args, f, dataframe_for_tuning, parent_repo)
        iface = self._f.meta_object
        if self._df is None or self._df.empty:
            self._lut_tensor, self._backends, self._binning_dict = iface.translate_empty_dataframe(f)
        else:
            self._lut_tensor, self._backends, self._binning_dict = iface.translate_dataframe(f, self._df)

    @property
    def is_trivial(self):
        return len(self._backends) <= 1

    def generate_trivial(self):
        functional = self._f
        iface = functional.meta_object
        mono_backend = str(self._backends[0])
        repo = self._parent_repo.get_dict_registry('trivial_tunes')
        repo.register((functional.arch_number, functional.godel_number), mono_backend)

    def generate(self):
        raise RuntimeError("TODO: Implement non-trivial optune")

