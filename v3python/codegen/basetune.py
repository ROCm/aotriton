# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from ..base import (
    typed_choice as TC,
    Functional,
    Interface,
)

class BaseTuneCodeGenerator(ABC):
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
