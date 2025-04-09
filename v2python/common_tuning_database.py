# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from .gpu_targets import AOTRITON_TUNING_DATABASE_REUSE, gpu2arch

'''
Note: unlike KernelDescription, whose constants will be specialized for EVERY kernel.
      KernelTuningDatabase(ForArch) should work for all KernelDescription instances.

      Therefore the index of the database can only be built when seeing the
      first set of ArgumentSelection objects, because the json object itself
      has zero information about the triton kernel.
'''
class CommonKernelTuningDatabaseForArch(ABC):
    def __init__(self, k : 'KernelDescription',
                 for_gpus : list[str],
                 db_gpus : list[str],
                 downgrader=None):
        self._for_gpus = for_gpus
        self._db_gpus = db_gpus
        self._kdesc = k
        self._downgrader = downgrader
        self._arch = gpu2arch(for_gpus[0])
        self._arch_number = self._kdesc.get_arch_number(self._arch)
        print(f'CommonKernelTuningDatabaseForArch.__init__ {k.FULL_KERNEL_NAME} {db_gpus=}')

    @property
    def arch(self):
        return self._arch

    @property
    @abstractmethod
    def empty(self):
        pass

    @property
    def for_gpus(self):
        return self._for_gpus

    @property
    def db_gpus(self):
        return self._db_gpus

    @property
    def arch_number(self):
        return self._arch_number

    @classmethod
    def is_passthrough_tuning(klass):
        return False

    '''
    Create db index, and also initialize _fsel_positions so that _extract_keys_from_fsels can use it
    '''
    @abstractmethod
    def _build_db_index(self, fsels):
        pass

    '''
    Callgraph: select -> _select_from_db -> _lookup_tuning_info
                                         <-
                         _select_from_db -> craft_perf_selection
               <--------------------------- craft_perf_selection
    Called by KernelDescription.gen_all_object_files to narrow down kernels to build for fsels
    '''
    def select(self, fsels : 'list[ArgumentSelection]', perf_meta : 'list[ArgumentMetadata]') -> 'list[ArgumentSelection], dict[str,str]':
        if self.empty:
            for gpu in self.for_gpus:
                yield gpu, [], None
        self._build_db_index(fsels)
        yield from self._select_from_db(fsels, perf_meta)

    @abstractmethod
    def _select_from_db(self,
                        fsels : 'list[ArgumentSelection]',
                        perf_meta : 'list[ArgumentMetadata]',
                        no_duplicate=True):
        pass

    '''
    Translate row into dict that only contains maps from input keys to values
    '''
    @abstractmethod
    def extract_inputs(self, columns, row):
        pass

    '''
    columns, row: one piece of tuning information, can be a json object, or a
                  single row in SQLite database.
                  For json database, columns is None (schemaless, metadata included in rows)
    Called by select -> _select_from_db
              or
              KernelTuningEntryForFunctionalOnGPU
    '''
    @abstractmethod
    def craft_perf_selection(self,
                             columns,
                             row,
                             perf_meta: 'list[ArgumentSelection]') -> 'list[TunedArgument], compiler_options':
        pass

    '''
    Callgraph: get_lut -> (Extract tuning info for kdesc+fsels)
               <--------- Construct KernelTuningEntryForFunctionalOnGPU fro tuning info
    '''
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
