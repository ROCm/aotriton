# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pathlib
import sqlite3
from .kernel_argument import ArgumentSelection, TunedArgument
from .gpu_targets import gpu2arch, AOTRITON_TUNING_DATABASE_REUSE
from .common_tuning_database import CommonKernelTuningDatabaseForArch
from .sqlite_tuning_database import SQLiteKernelTuningDatabaseForArch
from .downgrader import TuningDowngrader
from .tuning_lut import (
    KernelTuningEntryForFunctionalOnGPU,
    ARCH_TO_DIRECTORY,
)

class EmptyKernelTuningDatabaseForArch(CommonKernelTuningDatabaseForArch):
    def __init__(self, k, for_gpus):
        super().__init__(k, for_gpus, db_gpus=None)

    @property
    def empty(self):
        return True

    def _build_db_index(self, fsels):
        assert False

    def _select_from_db(self,
                        fsels : 'list[ArgumentSelection]',
                        perf_meta : 'list[ArgumentMetadata]',
                        no_duplicate=True):
        assert False

    def extract_inputs(self, columns, row):
        assert False

    def craft_perf_selection(self,
                             columns,
                             row,
                             perf_meta: 'list[ArgumentSelection]') -> 'list[TunedArgument], compiler_options':
        return [TunedArgument(meta, meta.default_value) for meta in perf_meta], None

    def get_lut(self,
                kdesc : 'KernelDescription',
                autotune_keys : 'list[tuple[str, Binning]]',
                fsels : 'list[ArgumentSelection]',
                perf_meta : 'list[ArgumentMetadata]'):
        return KernelTuningEntryForFunctionalOnGPU(kdesc, self, fsels,
                                                   columns=None, rows=None,
                                                   autotune_keys=None,
                                                   perf_meta=perf_meta)

class BootstrapTuningDatabaseForArch(EmptyKernelTuningDatabaseForArch):

    @classmethod
    def is_passthrough_tuning(klass):
        return True

    def extract_inputs(self, columns, row):
        assert False

    def craft_perf_selection(self,
                             columns,
                             row,
                             perf_meta: 'list[ArgumentSelection]') -> 'list[TunedArgument], compiler_options':
        if row is None:
            return [TunedArgument(meta, meta.default_value) for meta in perf_meta], None
        return row

    def get_lut(self,
                kdesc : 'KernelDescription',
                autotune_keys : 'list[tuple[str, Binning]]',
                fsels : 'list[ArgumentSelection]',
                perf_meta : 'list[ArgumentMetadata]'):
        fsel_dict = ArgumentSelection.build_fsel_dict(fsels)
        rows = []
        for cfg in kdesc.gen_autotune_configs(self._gpu, fsel_dict):
            psels, compiler_options = cfg.translate_to_psel_and_co(perf_meta)
            rows.append((psels, compiler_options))
        # print(f'get_lut {len(rows)=}')
        return KernelTuningEntryForFunctionalOnGPU(kdesc, self, fsels,
                                                   columns=None, rows=rows,
                                                   autotune_keys=None,
                                                   perf_meta=perf_meta)

class KernelTuningDatabase(object):
    MONOLITHIC_TUNING_DATABASE_FILE = 'tuning_database.sqlite3'

    def __init__(self, tune_info_dir : pathlib.Path, k : 'KernelDescription', build_for_tuning=False):
        self._kdesc = k
        self.gpu_dict = {}
        self._build_for_tuning = build_for_tuning and hasattr(k, 'gen_autotune_configs')
        self._cached_dba = {}
        self._gpu_set = set()
        if self._build_for_tuning:
            return
        td = pathlib.Path(tune_info_dir) / self.MONOLITHIC_TUNING_DATABASE_FILE # in case tune_info_dir is str
        # print(f"Tryint to probe KernelTuningDatabase inside {td}")
        self._downgrader = TuningDowngrader.create_from_kdesc(k)
        self._conn = sqlite3.connect(td)
        self._table_name = k.KERNEL_FAMILY.upper() + '$' + k._triton_kernel_name
        tup = self._conn.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{self._table_name}';").fetchone()
        if tup is None:  # Table not exists
            return
        res = self._conn.execute(f"SELECT DISTINCT gpu FROM {self._table_name};")
        self._gpu_set = set([gpu for gpu, in res.fetchall()])
        # Due to the complexity of GPU selections, the creation of
        # DatabaseForArch is deferred to select_gpus and acached in _cached_dba

    def select_gpus(self, gpus):
        arches = set(map(gpu2arch, gpus))
        assert len(arches) == 1, f'KernelTuningDatabase.select_gpus only accept gpus with the same arch, but receives {gpus=}, which becomes {arches=}'
        cache_key = tuple(sorted(gpus))
        if cache_key not in self._cached_dba:
            self._cached_dba[cache_key] = self.__create_dba(cache_key)
        return self._cached_dba[cache_key]

    #     if arch not in self.gpu_dict:
    #         if not self._build_for_tuning:
    #             self.arch_dict[arch] = EmptyKernelTuningDatabaseForArch(self._kdesc, arch)
    #         else:
    #             self.arch_dict[arch] = BootstrapTuningDatabaseForArch(self._kdesc, arch)
    #     return self.arch_dict[arch].set_gpu(gpu, index)

    def __create_dba(self, gpus):
        gpu404 = None
        reals = []
        for gpu in gpus:
            if gpu in self._gpu_set:
                reals.append(gpu)
                continue
            if gpu not in AOTRITON_TUNING_DATABASE_REUSE:
                gpu404 = gpu
                break
            real = AOTRITON_TUNING_DATABASE_REUSE[gpu]
            if real not in self._gpu_set:
                gpu404 = gpu
                break
            reals.append(real)
        if gpu404 is not None:
            if not self._build_for_tuning:
                print(f'For kernel {self._kdesc.KERNEL_FAMILY}.{self._kdesc.name}, GPU {gpu} from list {gpus} was not found in tuning database, using dummy one instead')
                return EmptyKernelTuningDatabaseForArch(self._kdesc, gpus)
            else:
                return BootstrapTuningDatabaseForArch(self._kdesc, gpus)
        return SQLiteKernelTuningDatabaseForArch(self._kdesc,
                                                 gpus,
                                                 reals,
                                                 self._conn,
                                                 self._table_name,
                                                 self._downgrader)

    @property
    def empty(self):
        return not self.arch_dict or self.build_for_tuning

    @property
    def build_for_tuning(self):
        return self._build_for_tuning
