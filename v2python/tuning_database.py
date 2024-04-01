# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
import pathlib
import sqlite3
from abc import ABC, abstractmethod
from copy import deepcopy
from collections import defaultdict
from .kernel_argument import TunedArgument
from .gpu_targets import AOTRITON_GPU_ARCH_TUNING_STRING
from .tuning_lut import KernelTuningEntryForFunctionalOnGPU

'''
Used in conjunction with PARTIALLY_TUNED_FUNCTIONALS

Commonly enabling functionals will cost extra resources,
and thus make the fallback turing information unusable
'''
class TuningDowngrader(object):
    def __init__(self, matching_list):
        self._matching_list = matching_list

    @staticmethod
    def create_from_kdesc(k : 'KernelDescription'):
        if not hasattr(k, 'DOWNGRADER'):
            return None
        return TuningDowngrader(k.DOWNGRADER)

    def match(self, matching, fallback_applied_fsels):
        iterator = iter(matching)
        while True:
            key = next(iterator, None)
            value = next(iterator, None)
            if key is None or value is None:
                break
            all_matched = True
            for fsel in fallback_applied_fsels:
                if not fsel.meta.has_argument(key):
                    all_matched = False
                    break
                if fsel.argument_value != value:
                    all_matched = False
                    break
            if all_matched:
                return True
        return False

    def lookup_patcher(self, fallback_applied_fsels):
        for matching, tuned_kernel_patcher in self._matching_list:
            if self.match(matching, fallback_applied_fsels):
                def patcher(tinfo):
                    print(f"Downgrade kernel from {tinfo['tuned_kernel']} {tinfo['compiler_options']}", end=' ')
                    tuned_kernel_patcher(tinfo['tuned_kernel'], tinfo['compiler_options'])
                    print(f"into {tinfo['tuned_kernel']} {tinfo['compiler_options']}")
                    return tinfo
                return patcher
        return None

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

class JsonKernelTuningDatabaseForArch(CommonKernelTuningDatabaseForArch):
    def _load_json_with_filter(self, f, kernel_name):
        j = json.load(f)
        tune_info = [ ti for ti in j['tune_info'] if ti['kernel_name'] == kernel_name]
        j['tune_info'] = tune_info
        return j

    def __init__(self, k, f, downgrader=None):
        self._j = self._load_json_with_filter(f, k.SHIM_KERNEL_NAME)
        super().__init__(k, self._j['arch'], downgrader=downgrader)
        self._index_matching_keys = None
        self._lut = {}
        self._index = None
        self._fsel_positions = None

    def _init_matching_keys(self, fsels):
        '''
        Translate functional selections (fsels) to keys for further extraction in database

        Result cached in self._index_matching_keys, reverse mapping cached in
        self._fsel_positions

        Note:
        fsel does not always have corresponding information in the database.
        Possible causation is the database is old, or the user simply did not
        store the key when running the benchmark. In either case, a "None"
        record must present in KernelDescription.PARTIALLY_TUNED_FUNCTIONALS.

        On the other hand, the reverse mapping must always present because
        _extract_keys_from_fsels requires this information to select
        functionals, and it always has a valid value because all fsels come
        from the KernelDescription class, and must have a position in the
        arugment list.
        '''
        self._index_matching_keys = []
        self._fsel_positions = []
        tinput = self._j['tune_info'][0]['inputs']
        for fsel in fsels:
            mfsel = fsel.meta
            if mfsel.nchoices <= 1:
                continue
            key_detected = None
            if mfsel.is_tensor:
                for tensor_name in fsel.argument_names:
                    tensor_key = f'{tensor_name}.dtype'
                    if tensor_key in tinput:
                        key_detected = tensor_key
                        break
            elif mfsel.is_type:
                key_detected = None # TODO
            elif mfsel.is_feature:
                for aname in fsel.argument_names:
                    if aname in tinput:
                        key_detected = aname
                        break
                if key_detected is None:
                    key_detected = '__UNDETECTED_{mfsel.argument_names[0]}'
                # Disable the assertion to allow old tuning database being used on newer kernels
                # assert key_detected is not None, f'Functional(s) {mfsel.argument_names} are not found in the database'
            self._index_matching_keys.append(key_detected)
            self._fsel_positions.append(fsel.meta.first_apperance)
        # print(f'{self._index_matching_keys=}')

    def _extract_keys_from_json(self, ti):
        keys = [ti['inputs'].get(k, None) for k in self._index_matching_keys]
        def convert(value):
            if isinstance(value, str) and value.startswith('torch.'):
                if value == 'torch.float16':
                    return '*fp16:16'
                elif value == 'torch.bfloat16':
                    return '*bf16:16'
                else:
                    assert False, f'Unknown datatype {value}'
            return value
        return tuple(map(convert, keys))

    def _build_db_index(self, fsels):
        if self._index is not None:
            return
        self._init_matching_keys(fsels)
        self._index = defaultdict(list)
        self._index_dedup = defaultdict(list)
        for ti in self._j['tune_info']:
            tup = self._extract_keys_from_json(ti)
            # print(f'_build_db_index {tup}')
            self._index[tup].append(ti)
            is_dup = False
            for eti in self._index_dedup[tup]:
                # print(f'{eti=}')
                if eti['tuned_kernel'] == ti['tuned_kernel'] and eti['compiler_options'] == ti['compiler_options']:
                    is_dup = True
                    break
            if not is_dup:
                self._index_dedup[tup].append(ti)
        if False:  # debug
            tup=('*fp16:16', 1, 16, True, True)
            print(f'_build_db_index {self._index[tup]=} {self._index_dedup[tup]=}')

    def _extract_keys_from_fsels(self, fsels, use_fallback_for_partially_tuned=False):
        keys = {}
        fallback_applied = []
        # print(f'{len(self._fsel_positions)=}')
        for fsel in fsels:
            try:
                # print(f'{fsel=}')
                # print(f'_extract_keys_from_fsels {fsel.argument_names} {fsel.meta.first_apperance}')
                offset = self._fsel_positions.index(fsel.meta.first_apperance)
                if use_fallback_for_partially_tuned and fsel.meta.incomplete_tuning:
                    value = fsel.meta.fallback_tuning_value
                    fallback_applied.append(fsel)
                else:
                    value = fsel.argument_value

                keys[offset] = value
                if value is None:
                    assert use_fallback_for_partially_tuned
                    assert fsel.meta.incomplete_tuning
                # print(f'keys[{offset}] = {value} {fsel=}')
            except ValueError:
                pass
        l = [keys[offset] for offset in range(len(self._fsel_positions))]
        # print(f'{l=}')
        return tuple(l), fallback_applied

    def _lookup_tuning_info(self, fsels, with_duplicates=True):
        tup, _ = self._extract_keys_from_fsels(fsels)
        if tup in self._index:
            return self._index[tup] if with_duplicates else self._index_dedup[tup]
        fallback_tup, fallback_applied_fsels = self._extract_keys_from_fsels(fsels, use_fallback_for_partially_tuned=True)
        print(f'Functionals {tup} cannot be found in tuning db, use {fallback_tup} instead')
        assert fallback_tup in self._index
        tuning_info = self._index[fallback_tup] if with_duplicates else self._index_dedup[fallback_tup]
        return self._downgrade(fallback_applied_fsels, tuning_info)

    def craft_perf_selection(self, tinfo, perf_meta: 'list[ArgumentSelection]'):
        if tinfo is None:  # default value when tuning db does not contain the kernel
            return [TunedArgument(meta, meta.default_value) for meta in perf_meta], None
        ps = dict(tinfo['tuned_kernel'])
        co = dict(tinfo['compiler_options'])
        if 'waves_per_eu' in ps:
            co['waves_per_eu'] = ps['waves_per_eu']
            # co['_debug'] = dict(tinfo)
            del ps['waves_per_eu']
        return [TunedArgument(meta, ps[meta.argument_names[0]]) for meta in perf_meta], co

    def _select_from_db(self,
                        fsels : 'list[ArgumentSelection]',
                        perf_meta : 'list[ArgumentMetadata]',
                        no_duplicate=True):
        indexed = self._lookup_tuning_info(fsels, with_duplicates=not no_duplicate)
        assert indexed
        for tinfo in indexed:
            yield self.craft_perf_selection(tinfo, perf_meta)

    def get_lut(self,
                kdesc : 'KernelDescription',
                autotune_keys : 'list[tuple[str, Binning]]',
                fsels : 'list[ArgumentSelection]',
                perf_meta : 'list[ArgumentMetadata]'):
        if self.empty:
            # Null Lut
            return KernelTuningEntryForFunctionalOnGPU(kdesc, self, fsels,
                                                       indexed=None, autotune_keys=None,
                                                       perf_meta=perf_meta)
        self._build_db_index(fsels)
        tup, _  = self._extract_keys_from_fsels(fsels)
        if tup not in self._lut:
            indexed = self._lookup_tuning_info(fsels)
            # print(f'{tup=}')
            assert indexed
            self._lut[tup] = KernelTuningEntryForFunctionalOnGPU(kdesc, self, fsels, indexed,
                                                                 autotune_keys, perf_meta)
        return self._lut[tup]

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
                                                   indexed=None, autotune_keys=None,
                                                   perf_meta=perf_meta)

class SQLiteKernelTuningDatabaseForArch(CommonKernelTuningDatabaseForArch):
    def __init__(self, k, arch, conn, table_name, downgrader=None):
        super().__init__(k, arch, downgrader=downgrader)
        self._conn = conn
        self._kdesc = k
        self._table_name = table_name
        self._input_column_names = None
        self._column_names = None
        self._column_name_to_index = None
        self._fsel_index_to_column_name = None
        self._lut = {}
        self._select_stmt_base = None

    @property
    def empty(self):
        stmt = f'SELECT COUNT(id) FROM {self._table_name} WHERE arch = ?'
        nitem, = self._conn.execute(stmt, (self._arch,)).fetchone()
        return nitem > 0

    def _build_db_index(self, fsels):
        if self._fsel_index_to_column_name is not None:
            return
        self._input_column_names = []
        self._fsel_index_to_column_name = {}
        cursor = self._conn.cursor()
        cursor.execute(f"SELECT * FROM {self._table_name};")
        self._column_name_to_index = { tup[0] : index for index, tup in enumerate(cursor.description) }
        self._column_names = set(self._column_name_to_index.keys())
        for fsel in fsels:
            mfsel = fsel.meta
            if mfsel.nchoices <= 1:
                continue
            key_detected = None
            if mfsel.is_tensor:
                for tensor_name in fsel.argument_names:
                    tensor_key = f'inputs${tensor_name}_dtype'
                    if tensor_key in column_names:
                        key_detected = tensor_key
                        break
            elif mfsel.is_type:
                key_detected = None # TODO
            elif mfsel.is_feature:
                for aname in fsel.argument_names:
                    if f'inputs${aname}' in column_names:
                        key_detected = aname
                        break
                if key_detected is None:
                    key_detected = '__UNDETECTED_{mfsel.argument_names[0]}'
            if key_detected is not None:
                self._input_column_names.append(key_detected)
                self._fsel_index_to_column_name[fsel.meta.first_apperance] = key_detected
        stmt_all_columns = ', '.join(self._input_column_names)
        self._select_stmt_base = f'SELECT {stmt_all_columns} from {self._table_name} '

    def _lookup_tuning_info(self, fsels, with_duplicates=True):
        columns, values = self._extract_colunm_and_values(fsels)
        rows = self._select_from_table(columns, values)
        if not rows:
            columns, values = self._extract_colunm_and_values(fsels, use_fallback_for_partially_tuned=True)
            rows = self._select_from_table(columns, values)
        assert rows
        return self._downgrade(rows)

    def _select_from_db(self,
                        fsels : 'list[ArgumentSelection]',
                        perf_meta : 'list[ArgumentMetadata]',
                        no_duplicate=True):
        rows = self._lookup_tuning_info(fsels, with_duplicates=not no_duplicate)
        assert rows
        for row in rows:
            yield self.craft_perf_selection(row, perf_meta)

    def craft_perf_selection(self, row, perf_meta: 'list[ArgumentSelection]'):
        if row is None:  # default value when tuning db does not contain the kernel
            return [TunedArgument(meta, meta.default_value) for meta in perf_meta], None
        ps = self._row_to_dict(row, prefix='tuned_kernel')
        co = self._row_to_dict(row, prefix='compiler_options')
        if 'waves_per_eu' in ps:
            co['waves_per_eu'] = ps['waves_per_eu']
            del ps['waves_per_eu']
        return [TunedArgument(meta, ps[meta.argument_names[0]]) for meta in perf_meta], co

    def get_lut(self,
                kdesc : 'KernelDescription',
                autotune_keys : 'list[tuple[str, Binning]]',
                fsels : 'list[ArgumentSelection]',
                perf_meta : 'list[ArgumentMetadata]'):
        if self.empty:
            # Null Lut
            return KernelTuningEntryForFunctionalOnGPU(kdesc, self, fsels,
                                                       indexed=None, autotune_keys=None,
                                                       perf_meta=perf_meta)
        self._build_db_index(fsels)
        columns, values  = self._extract_colunm_and_values(fsels)
        if values not in self._lut:
            rows = self._lookup_tuning_info(fsels)
            # print(f'{tup=}')
            assert rows
            self._lut[tup] = KernelTuningEntryForFunctionalOnGPU(kdesc, self, fsels, rows,
                                                                 autotune_keys, perf_meta)
        return self._lut[tup]

    def _cast_argument_selection_to_sqlite3(self, mfsel, value):
        if mfsel.is_tensor:
            if value.startswith('*fp16'):
                return 'torch.float16'
            if value.startswith('*bf16'):
                return 'torch.bfloat16'
            if value.startswith('*fp32'):
                return 'torch.float32'
            # TODO: Add more tensor types here
            assert False
        elif mfsel.is_type:
            # Should not hit here since functional argument do not need multiple types
            # For examples, it is not necessary to compiler kernels with offset:i32 and offset:i64
            assert False
            return value
        else:
            return value


    def _extract_colunm_and_values(self,
                                   fsels : 'list[ArgumentSelection]',
                                   use_fallback_for_partially_tuned=False):
        columns = []
        values = []
        # print(f'{len(self._fsel_positions)=}')
        for fsel in fsels:
            try:
                # print(f'{fsel=}')
                # print(f'_extract_keys_from_fsels {fsel.argument_names} {fsel.meta.first_apperance}')

                colunm_name = self._fsel_index_to_column_name[fsel.meta.first_apperance]
                if use_fallback_for_partially_tuned and fsel.meta.incomplete_tuning:
                    value = fsel.meta.fallback_tuning_value
                else:
                    value = fsel.argument_value
                value = self._cast_argument_selection_to_sqlite3(fsel.meta, value)

                columns.append(colunm_name)
                values.append(value)
                if value is None:
                    assert use_fallback_for_partially_tuned
                    assert fsel.meta.incomplete_tuning
                # print(f'keys[{offset}] = {value} {fsel=}')
            except ValueError:
                pass
        return columns, values

    def _select_from_table(self, columns, values):
        assert False # TODO
        conds = [ 'arch = ?' ]
        # Check value is not None in case falling back to any value
        conds += [f'{column} = ?' for colunm, v in zip(columns, v) if v is not None]
        select_vals = [self._arch]
        select_vals = [v for v in values if v is not None]
        select_stmt = self._select_stmt_base + ' WHERE ' + ' AND '.join(conds)
        return self._conn.execute(select_stmt, select_vals).fetchall()

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
