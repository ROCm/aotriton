# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import json
from collections import defaultdict
from .common_tuning_database import CommonKernelTuningDatabaseForArch

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

    def craft_perf_selection(self,
                             columns,
                             row,
                             perf_meta: 'list[ArgumentSelection]') -> 'list[TunedArgument], compiler_options':
        tinfo = row
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
            yield self.craft_perf_selection(None, tinfo, perf_meta)

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

    def extract_inputs(self, columns, tinfo):
        assert columns is None
        return tinfo['inputs']
