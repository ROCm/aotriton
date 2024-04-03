# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod

'''
Note: unlike KernelDescription, whose constants will be specialized for EVERY kernel.
      KernelTuningDatabase(ForArch) should work for all KernelDescription instances.

      Therefore the index of the database can only be built when seeing the
      first set of ArgumentSelection objects, because the json object itself
      has zero information about the triton kernel.
'''
class CommonKernelTuningDatabaseForArch(ABC):
    def __init__(self, k : 'KernelDescription', arch : str, downgrader=None):
        self._kdesc = k
        self._arch = arch
        self._gpu = None
        self._downgrader = downgrader

    @property
    def arch(self):
        return self._arch

    @property
    @abstractmethod
    def empty(self):
        pass

    def set_gpu(self, gpu, index):
        self._gpu = gpu
        self._arch_number = index
        return self

    @property
    def gpu(self):
        return self._gpu

    @property
    def arch_number(self):
        return self._arch_number

    '''
    Create db index, and also initialize _fsel_positions so that _extract_keys_from_fsels can use it
    '''
    @abstractmethod
    def _build_db_index(self, fsels):
        pass

    def select(self, fsels : 'list[ArgumentSelection]', perf_meta : 'list[ArgumentMetadata]') -> 'list[ArgumentSelection], dict[str,str]':
        if self.empty:
            yield [], None
        self._build_db_index(fsels)
        yield from self._select_from_db(fsels, perf_meta)

    @abstractmethod
    def _select_from_db(self,
                        fsels : 'list[ArgumentSelection]',
                        perf_meta : 'list[ArgumentMetadata]',
                        no_duplicate=True):
        pass

    '''
    tinfo: one piece of tuning information, can be a json object, or a row in SQLite database
    '''
    @abstractmethod
    def craft_perf_selection(self, tinfo, perf_meta: 'list[ArgumentSelection]'):
        pass

    @abstractmethod
    def get_lut(self,
                kdesc : 'KernelDescription',
                autotune_keys : 'list[tuple[str, Binning]]',
                fsels : 'list[ArgumentSelection]',
                perf_meta : 'list[ArgumentMetadata]'):
        pass

    def _downgrade(self, fallback_applied_fsels, tuning_info):
        if self._downgrader is None:
            return tuning_info
        patcher = self._downgrader.lookup_patcher(fallback_applied_fsels)
        if patcher is None:
            return tuning_info
        return [patcher(deepcopy(tune)) for tune in tuning_info]
