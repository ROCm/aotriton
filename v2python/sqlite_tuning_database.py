# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


import sqlite3
from abc import ABC, abstractmethod
from .kernel_argument import TunedArgument
from .tuning_lut import KernelTuningEntryForFunctionalOnGPU
from .common_tuning_database import CommonKernelTuningDatabaseForArch
from .downgrader import TuningDowngrader

class SQLiteKernelTuningDatabaseForArch(CommonKernelTuningDatabaseForArch):
    UNDETECTED_COLUMN_PREFIX = '$$__UNDETECTED_'

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
        return nitem == 0

    def _build_db_index(self, fsels):
        if self._fsel_index_to_column_name is not None:
            return
        self._input_column_names = []
        self._fsel_index_to_column_name = {}
        cursor = self._conn.cursor()
        cursor.execute(f"SELECT * FROM {self._table_name};")
        self._column_name_to_index = { tup[0] : index for index, tup in enumerate(cursor.description) }
        self._column_names = [tup[0] for tup in cursor.description]
        self._column_names_set = set(self._column_names)
        self._tuning_column_names = [cn for cn in self._column_names if cn.startswith('tuned_kernel$') or cn.startswith('compiler_options$')]
        for fsel in fsels:
            mfsel = fsel.meta
            if mfsel.nchoices <= 1:
                continue
            key_detected = None
            if mfsel.is_tensor:
                for tensor_name in fsel.argument_names:
                    tensor_key = f'inputs${tensor_name}_dtype'
                    if tensor_key in self._column_names_set:
                        key_detected = tensor_key
                        break
            elif mfsel.is_type:
                key_detected = None # TODO
            elif mfsel.is_feature:
                for aname in fsel.argument_names:
                    if f'inputs${aname}' in self._column_names_set:
                        key_detected = f'inputs${aname}'
                        break
                if key_detected is None:
                    key_detected = f'{self.UNDETECTED_COLUMN_PREFIX}{mfsel.argument_names[0]}'
            if not key_detected.startswith(self.UNDETECTED_COLUMN_PREFIX):
                self._input_column_names.append(key_detected)
                self._fsel_index_to_column_name[fsel.meta.first_apperance] = key_detected
        stmt_all_columns = ', '.join(self._column_names)
        self._select_all_stmt_base = f'SELECT {stmt_all_columns} from {self._table_name} '
        stmt_tune_columns = ', '.join(self._tuning_column_names)
        self._select_tune_stmt_base = f'SELECT DISTINCT {stmt_tune_columns} from {self._table_name} '
        # print(f'{self._input_column_names=}')
        # print(f'{self._tuning_column_names=}')

    '''
    Unlike the json version, this one needs perf_meta for deduplication
    '''
    def _lookup_tuning_info(self, fsels, perf_meta, with_duplicates=True):
        mfsels, where_columns, where_values = self._extract_colunm_and_values(fsels)
        target_values = where_values
        selected_columns, selected_rows = self._select_from_table(where_columns, target_values, with_inputs=with_duplicates)
        if not selected_rows:
            patched_values = self._apply_fallback(mfsels, where_columns, where_values)
            target_values = patched_values
            selected_columns, selected_rows = self._select_from_table(where_columns, target_values, with_inputs=with_duplicates)
        # print(f'{selected_columns=}')
        # print(f'{selected_rows=}')
        if not selected_rows:
            def fmtsql(value):
                if isinstance(value, bool):
                    return int(value)
                if isinstance(value, str):
                    return "'" + value + "'"
                return value
            selection = ' and '.join([f'{colname}={fmtsql(value)}' for colname, value in zip(where_columns, where_values)])
            fb_selection = ' and '.join([f'{colname}={fmtsql(value)}' for colname, value in zip(where_columns, patched_values)])
            assert selected_rows, f"Cannot find any rows from select * from {self._table_name} where arch='{self.arch}' and {selection} (fallback to {fb_selection})"
        # TODO: Support KernelDescription.DOWNGRADER
        # return columns, values, self._downgrade(rows)

        # Note: do NOT return patched_values.
        # get_lut needs this to identify the LUT
        return where_columns, tuple(where_values), selected_columns, selected_rows

    def _select_from_db(self,
                        fsels : 'list[ArgumentSelection]',
                        perf_meta : 'list[ArgumentMetadata]',
                        no_duplicate=True):
        _, _, selected_columns, selected_rows = self._lookup_tuning_info(fsels, perf_meta, with_duplicates=not no_duplicate)
        assert selected_rows
        for row in selected_rows:
            yield self.craft_perf_selection(selected_columns, row, perf_meta)

    def craft_perf_selection(self,
                             columns,
                             row,
                             perf_meta: 'list[ArgumentSelection]') -> 'list[TunedArgument], compiler_options':
        if row is None:  # default value when tuning db does not contain the kernel
            return [TunedArgument(meta, meta.default_value) for meta in perf_meta], None
        ps = self._row_to_dict(columns, row, prefix='tuned_kernel')
        co = self._row_to_dict(columns, row, prefix='compiler_options')
        if 'waves_per_eu' in ps:
            co['waves_per_eu'] = ps['waves_per_eu']
            del ps['waves_per_eu']
        return [TunedArgument(meta, ps[meta.argument_names[0]]) for meta in perf_meta], co

    def _row_to_dict(self, columns, row, prefix):
        d = {}
        for cname, cdata in zip(columns, row):
            if cname.startswith(prefix):
                key = cname[len(prefix) + 1:] # '$' separator
                d[key] = cdata
        return d

    def _row_to_list(self, columns, row, prefix, fields=None):
        l = []
        for cname, cdata in zip(columns, row):
            if cname.startswith(prefix):
                if fields is not None:
                    key = cname[len(prefix) + 1:] # '$' separator
                    if key in fields:
                        l.append(data)
                else:
                    l.append(cdata)
        return l

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
        # print(f'{autotune_keys=}')
        self._build_db_index(fsels)
        where_columns, where_values, selected_columns, selected_rows = self._lookup_tuning_info(fsels, perf_meta, with_duplicates=True)
        # print(f'SQLite.get_lut {fsels=}')
        # print(f'SQLite.get_lut {where_columns=}')
        # print(f'SQLite.get_lut {where_values=}')
        lut_key = tuple([s.compact_signature for s in fsels])
        if lut_key not in self._lut:
            # print(f'{selected_rows=}')
            assert selected_rows
            self._lut[lut_key] = KernelTuningEntryForFunctionalOnGPU(kdesc, self, fsels,
                                                                     selected_columns, selected_rows,
                                                                     autotune_keys, perf_meta)
        return self._lut[lut_key]

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


    '''
    Unlike Json (which is schemaless), with SQLite we can handle partially
    tuned database by checking the existance of columns from schemas
    '''
    def _extract_colunm_and_values(self,
                                   fsels : 'list[ArgumentSelection]'):
        columns = []
        values = []
        mfsels = []
        # print(f'{len(self._fsel_positions)=}')
        for fsel in fsels:
            try:
                # print(f'{fsel=}')
                # print(f'_extract_keys_from_fsels {fsel.argument_names} {fsel.meta.first_apperance}')

                colunm_name = self._fsel_index_to_column_name[fsel.meta.first_apperance]
                value = fsel.argument_value
                value = self._cast_argument_selection_to_sqlite3(fsel.meta, value)

                columns.append(colunm_name)
                values.append(value)
                mfsels.append(fsel.meta)
                if value is None:
                    assert use_fallback_for_partially_tuned
                    assert fsel.meta.incomplete_tuning
                # print(f'keys[{offset}] = {value} {fsel=}')
            except KeyError:
                pass
        return mfsels, columns, values

    def _apply_fallback(self, mfsels, columns, values):
        fallback_values = []
        for mfsel, c, v in zip(mfsels, columns, values):
            if mfsel.incomplete_tuning:
                v = mfsel.fallback_tuning_value
            fallback_values.append(v)
        return fallback_values

    def _select_from_table(self, columns, values, with_inputs):
        conds = [ 'arch = ?' ]
        # print(f'{columns=} {values=}')
        # Check value is not None in case falling back to any value
        conds += [f'{column} = ?' for column, v in zip(columns, values) if v is not None]
        select_vals = [self._arch]
        select_vals += [v for v in values if v is not None]
        # print(f'{conds=}')
        if with_inputs:
            stmt_base = self._select_all_stmt_base
            selected_columns = self._column_names
        else:
            stmt_base = self._select_tune_stmt_base
            selected_columns = self._tuning_column_names
        select_stmt = stmt_base + ' WHERE ' + ' AND '.join(conds)
        # print(f'{select_stmt=}')
        # print(f'{select_vals=}')
        return selected_columns, self._conn.execute(select_stmt, select_vals).fetchall()

    def extract_inputs(self, columns, row):
        return self._row_to_dict(columns, row, prefix='inputs')
