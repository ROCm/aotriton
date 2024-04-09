# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pathlib
import sqlite3
from .kernel_argument import TunedArgument
from .gpu_targets import AOTRITON_GPU_ARCH_TUNING_STRING
from .common_tuning_database import CommonKernelTuningDatabaseForArch
from .sqlite_tuning_database import SQLiteKernelTuningDatabaseForArch
from .downgrader import TuningDowngrader
from .tuning_lut import KernelTuningEntryForFunctionalOnGPU

class EmptyKernelTuningDatabaseForArch(CommonKernelTuningDatabaseForArch):
    def __init__(self, k, arch):
        super().__init__(k, arch)

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

    def craft_perf_selection(self, tinfo, perf_meta: 'list[ArgumentSelection]'):
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

class KernelTuningDatabase(object):
    MONOLITHIC_TUNING_DATABASE_FILE = 'tuning_database.sqlite3'

    def __init__(self, tune_info_dir : pathlib.Path, k : 'KernelDescription'):
        self._kdesc = k
        self.arch_dict = {}
        td = pathlib.Path(tune_info_dir) / self.MONOLITHIC_TUNING_DATABASE_FILE # in case tune_info_dir is str
        # print(f"Tryint to probe KernelTuningDatabase inside {td}")
        downgrader = TuningDowngrader.create_from_kdesc(k)
        self._conn = sqlite3.connect(td)
        table_name = k.KERNEL_FAMILY.upper() + '$' + k._triton_kernel_name
        tup = self._conn.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';").fetchone()
        if tup is None:  # Table not exists
            return
        res = self._conn.execute(f"SELECT DISTINCT arch FROM {table_name};")
        for arch, in res.fetchall():
            dba = SQLiteKernelTuningDatabaseForArch(k, arch, self._conn, table_name, downgrader)
            self.arch_dict[arch] = dba

    def select_gpu(self, gpu, index):
        arch = AOTRITON_GPU_ARCH_TUNING_STRING[gpu]
        if arch not in self.arch_dict:
            print('For kernel {self._kdesc.KERNEL_FAMILY}.{self._kdesc.name}, Architecture {arch} was not found in tuning database, using dummy one instead')
            self.arch_dict[arch] = EmptyKernelTuningDatabaseForArch(self._kdesc, arch)
        return self.arch_dict[arch].set_gpu(gpu, index)

    @property
    def empty(self):
        return not self.arch_dict
